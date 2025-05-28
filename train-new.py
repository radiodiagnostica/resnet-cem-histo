#!/usr/bin/env python3
"""
Hormone-receptor (+/–) classification ─ 2024-06
────────────────────────────────────────────────
Key upgrades compared with the previous version
• Folder “1” → class 1 (HR-positive); Folder “2” → class 0 (HR-negative)
  realised by the HRDataset wrapper.

• Threshold optimisation = arg-max Matthews correlation coefficient (MCC)
  on a fine grid (0.01 … 0.99, step 0.01).  This avoids the “all positive”
  pathology we saw with F1 on an imbalanced validation set.

• Optional focal-loss with class weights (γ parameter configurable
  from CLI, default γ = 0 ⇒ plain weighted CE).

• Confusion matrix + class-wise metrics printed every epoch.

• Vectorised bootstrap (2 000 reps by default) for 95 % CIs;
  permutation testing removed to keep the log readable.

Directory layout expected:

data_root/
 ├── train/         (sub-folders 1, 2)
 ├── val/
 └── external_val/
"""
from __future__ import annotations
import argparse, random, time, warnings, sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score, roc_auc_score,
    confusion_matrix
)

# ───────────────────────── Hyper-parameters ─────────────────────────
IMG_SIZE      = 384
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 30
BATCH_SIZE    = 16
N_BOOT        = 2_000
SEED          = 42
EARLY_STOP    = 0            # set >0 for patience
# ────────────────────────────────────────────────────────────────────

# ───────── reproducibility ─────────
def set_seed(seed:int=SEED)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
set_seed()

DEVICE = (
    torch.device("cuda")    if torch.cuda.is_available() else
    torch.device("mps")     if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print("Running on", DEVICE)
if DEVICE.type=="mps":
    torch.set_float32_matmul_precision('high')

# ──────────────────────── Data set with label remap ─────────────────────────
class HRDataset(datasets.ImageFolder):
    """Force folder '1'→label 1 (positive), '2'→label 0 (negative)."""
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # original labels: '1':0, '2':1  (alphabetic)
        # flip them:
        self.samples = [(fp, 1 - lbl) for fp,lbl in self.samples]
        self.targets = [lbl for _,lbl in self.samples]
        self.class_to_idx = {'2':0,'1':1}     # cosmetic

# ─────────────────────── transforms ────────────────────────────────
def build_transforms(img_size:int)->Tuple[transforms.Compose,transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    eval_tf  = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def _seed_worker(worker_id:int)->None:
    np.random.seed(SEED+worker_id)
    random.seed(SEED+worker_id)

def make_loader(root:str, split:str, tf, batch:int,
                balance:bool=False, workers:int=2)->Tuple[DataLoader,HRDataset]:
    ds = HRDataset(Path(root)/split, transform=tf)

    # weighted sampling (train only)
    if balance and split=="train":
        targets = ds.targets
        class_counts = np.bincount(targets)
        weights = 1./class_counts[targets]
        sampler = WeightedRandomSampler(weights,len(weights),replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = (split=="train")

    loader = DataLoader(ds,
                        batch_size=batch,
                        shuffle=shuffle,
                        sampler=sampler,
                        num_workers=workers,
                        worker_init_fn=_seed_worker,
                        pin_memory = (DEVICE.type=="cuda"),
                        persistent_workers = (workers>0))
    return loader, ds

# ───────────────────────── focal loss (optional) ───────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha:torch.Tensor, gamma:float=2.0, reduction:str="mean"):
        super().__init__()
        self.alpha = alpha          # weight per class
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")
    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        p_t = torch.exp(-ce_loss)   # prob. of the true class
        focal = (self.alpha[target] * (1-p_t)**self.gamma * ce_loss)
        if self.reduction=="mean":
            return focal.mean()
        return focal.sum()

# ─────────────────────────── model ────────────────────────────────
def build_model(pretrained:bool=True)->nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, 2)
    nn.init.xavier_uniform_(m.fc.weight); nn.init.zeros_(m.fc.bias)
    return m

# ──────────── metric utils (vectorised bootstrap for CIs) ─────────
def _confusion(y_true,y_pred):
    tp = np.sum((y_true==1)&(y_pred==1))
    tn = np.sum((y_true==0)&(y_pred==0))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    return tp,fp,fn,tn

def metrics_from_preds(y_true,y_pred,y_prob)->Dict[str,float]:
    tp,fp,fn,tn = _confusion(y_true,y_pred)
    acc  = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred,zero_division=0)
    rec  = recall_score(y_true,y_pred)
    f1   = f1_score(y_true,y_pred)
    mcc  = matthews_corrcoef(y_true,y_pred)
    bal  = balanced_accuracy_score(y_true,y_pred)
    auc  = roc_auc_score(y_true,y_prob)
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,
                accuracy=acc,precision=prec,recall=rec,specificity=spec,
                f1=f1,mcc=mcc,balanced_accuracy=bal,roc_auc=auc)

def bootstrap_ci(y_true,y_prob,thr:float, n:int=N_BOOT, alpha:float=0.05):
    y_true = np.asarray(y_true); y_prob=np.asarray(y_prob)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0,len(y_true), size=(n,len(y_true)))
    metrics = []
    for b in idx:
        y_b = y_true[b]; p_b = y_prob[b]
        y_pred = (p_b>=thr).astype(int)
        metrics.append(metrics_from_preds(y_b,y_pred,p_b))
    out = {}
    for k in metrics[0]:
        dist = np.array([m[k] for m in metrics])
        out[k]=(np.percentile(dist,100*alpha/2),
                np.percentile(dist,100*(1-alpha/2)))
    return out

# ───────────── threshold tuning (max MCC) ─────────────
def best_mcc_threshold(y_true,y_prob)->float:
    thrs = np.linspace(0.01,0.99,99)
    mccs = [matthews_corrcoef(y_true,(y_prob>=t).astype(int)) for t in thrs]
    return float(thrs[int(np.argmax(mccs))])

# ─────────────────────── train / eval loops ───────────────────────
def train_one_epoch(model,loader,criterion,optim,device,scaler=None)->float:
    model.train(); running=0.0
    pbar = tqdm(loader,leave=False,desc="train")
    for x,y in pbar:
        x,y = x.to(device), y.to(device)
        optim.zero_grad()
        if scaler is None:
            out = model(x); loss = criterion(out,y); loss.backward(); optim.step()
        else:
            with torch.cuda.amp.autocast():
                out = model(x); loss = criterion(out,y)
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        running += loss.item()*x.size(0)
        pbar.set_postfix(loss=loss.item())
    return running/len(loader.dataset)

@torch.no_grad()
def inference(model,loader,device):
    model.eval(); y_true=[]; y_prob=[]
    for x,y in loader:
        x=x.to(device)
        with torch.cuda.amp.autocast(enabled=False):
            logits = model(x)
        prob = torch.softmax(logits,1)[:,1]
        y_true.append(y.numpy()); y_prob.append(prob.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)

# ─────────────────────────────── main ────────────────────────────
def main(args):
    train_tf, eval_tf = build_transforms(args.img_size)

    train_loader, train_ds = make_loader(args.data_root,'train',train_tf,
                                         args.batch_size, balance=True)
    val_loader,   _        = make_loader(args.data_root,'val',  eval_tf,
                                         args.batch_size)
    ext_loader,   _        = make_loader(args.data_root,'external_val',eval_tf,
                                         args.batch_size)

    # class weights (inverse frequency)
    counts = np.bincount(train_ds.targets)
    class_w = torch.tensor(1./counts, dtype=torch.float32, device=DEVICE)

    # choose loss
    if args.focal_gamma>0.0:
        criterion = FocalLoss(alpha=class_w, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_w)

    model = build_model(pretrained=not args.no_pretrain).to(DEVICE)
    if args.torch_compile and hasattr(torch,"compile"):
        model = torch.compile(model,mode="reduce-overhead")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type=="cuda" else None

    best_val_mcc=-1.0; best_thr=0.5; history=[]; epochs_no_gain=0

    for epoch in range(args.epochs):
        t0=time.time()
        train_loss = train_one_epoch(model,train_loader,criterion,optim,DEVICE,scaler)

        # ─── validation & ext
        y_val,p_val = inference(model,val_loader,DEVICE)
        best_thr    = best_mcc_threshold(y_val,p_val)

        y_ext,p_ext = inference(model,ext_loader,DEVICE)

        # compute metrics
        y_val_pred  = (p_val>=best_thr).astype(int)
        y_ext_pred  = (p_ext>=best_thr).astype(int)

        val_met = metrics_from_preds(y_val,y_val_pred,p_val)
        ext_met = metrics_from_preds(y_ext,y_ext_pred,p_ext)

        # bootstrap CIs (only for major metrics to keep run time low)
        val_ci = bootstrap_ci(y_val,p_val,best_thr)
        ext_ci = bootstrap_ci(y_ext,p_ext,best_thr)

        # checkpoint
        if val_met['mcc']>best_val_mcc:
            best_val_mcc=val_met['mcc']
            torch.save({'state_dict':model.state_dict(),'thr':best_thr},
                       'best_model.pt')
            epochs_no_gain=0
        else: epochs_no_gain+=1

        # ─── logging
        def _fmt(score,ci): return f"{score:5.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
        print(f"\nEpoch {epoch+1:02d}/{args.epochs} | "
              f"loss {train_loss:.4f} | thr={best_thr:.2f} | "
              f"{time.time()-t0:5.1f}s")

        for split,met,ci in [("VAL",val_met,val_ci),("EXT",ext_met,ext_ci)]:
            print(f"\n{split}  confusion  TP:{met['tp']} FP:{met['fp']} "
                  f"FN:{met['fn']} TN:{met['tn']}")
            print("{:18s} {:>18s}".format("metric","score [95% CI]"))
            for k in ['accuracy','precision','recall','specificity',
                      'balanced_accuracy','f1','mcc','roc_auc']:
                print("{:18s} {:>18s}".format(k,_fmt(met[k],ci[k])))

        # save CSV log
        row={'epoch':epoch,'thr':best_thr,**val_met,**{f"ext_{k}":v for k,v in ext_met.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv("training_log.csv",index=False)

        if EARLY_STOP and epochs_no_gain>=EARLY_STOP:
            print("Early stopping triggered.")
            break

    print("\nFinished. Best MCC (val):",best_val_mcc,
          "\nModel & threshold stored in best_model.pt")

# ───────────────────────────── CLI ───────────────────────────────
def parse_args():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data_root',default='.',type=str,
                    help='root folder with train/val/external_val')
    ap.add_argument('--epochs',default=EPOCHS,type=int)
    ap.add_argument('--batch_size',default=BATCH_SIZE,type=int)
    ap.add_argument('--img_size',default=IMG_SIZE,type=int)
    ap.add_argument('--lr',default=LR,type=float)
    ap.add_argument('--weight_decay',default=WEIGHT_DECAY,type=float)
    ap.add_argument('--no_pretrain',action='store_true',
                    help='start from random weights')
    ap.add_argument('--torch_compile',action='store_true')
    ap.add_argument('--focal_gamma',default=0.0,type=float,
                    help='γ for focal loss (0 → plain CE)')
    return ap.parse_args()

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=UserWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    main(parse_args())