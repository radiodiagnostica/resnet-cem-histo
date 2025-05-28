#!/usr/bin/env python3
"""
HR (+/–) classification — patient-level K-fold CV
──────────────────────────────────────────────────
• Any image file inside `train/` *or* `val/` is pooled together; splitting is
  done *per patient* with StratifiedKFold so every image of a patient lives
  in exactly one fold.

• After the K models are trained we concatenate the K validation predictions,
  pick the single threshold that maximises MCC on that pool, then re-score
  every fold + external set with that **fixed** threshold.

Folder layout (unchanged):

data_root/
 ├── train/          (1, 2)
 ├── val/            (1, 2)
 └── external_val/   (1, 2)
"""
from __future__ import annotations
import argparse, random, time, re, warnings, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score, roc_auc_score,
)

# ──────────────────── hyper-params you may tweak ───────────────────
IMG_SIZE      = 384
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 25
BATCH_SIZE    = 16
N_BOOT        = 2_000
SEED          = 42
N_FOLDS       = 5
# ───────────────────────────────────────────────────────────────────

def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print("Running on", DEVICE)

# ═══════════════════════ Dataset utilities ════════════════════════
PAT_RE = re.compile(r"^(?:train_|val_)?(.+?)_")   # strip possible prefix

def patient_id(fname: str) -> str:
    """Get patient id from basename."""
    return PAT_RE.match(fname).group(1) if PAT_RE.match(fname) else fname

def collect_images(root: Path) -> pd.DataFrame:
    """Return dataframe (path, label, patient) for train + val."""
    rows = []
    for split in ("train", "val"):
        for lbl_dir in (root/split/"1", root/split/"2"):
            if not lbl_dir.exists(): continue
            label = 1 if lbl_dir.name == "1" else 0
            for fp in lbl_dir.glob("*.jpg"):
                rows.append((fp.as_posix(), label, patient_id(fp.name)))
    df = pd.DataFrame(rows, columns=["path", "label", "pid"])
    return df

class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        return self.transform(img), row.label

def build_transforms(img_size: int) -> Tuple[Any, Any]:
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

# ═══════════════════════ Model + loss ═════════════════════════════
def build_model(pretrained: bool = True) -> nn.Module:
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    nn.init.xavier_uniform_(m.fc.weight); nn.init.zeros_(m.fc.bias)
    return m

# ─────────────────── metric helpers (same as your old) ────────────
def confusion(y_true, y_pred):
    tp = np.sum((y_true==1)&(y_pred==1))
    tn = np.sum((y_true==0)&(y_pred==0))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    return tp,fp,fn,tn

def mtrx(y_true, y_pred, y_prob):
    tp,fp,fn,tn = confusion(y_true,y_pred)
    spec = tn/(tn+fp) if (tn+fp) else np.nan
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy = accuracy_score(y_true,y_pred),
        precision = precision_score(y_true,y_pred, zero_division=0),
        recall = recall_score(y_true,y_pred),
        specificity = spec,
        balanced_accuracy = balanced_accuracy_score(y_true,y_pred),
        f1 = f1_score(y_true,y_pred),
        mcc = matthews_corrcoef(y_true,y_pred),
        roc_auc = roc_auc_score(y_true,y_prob) if (len(np.unique(y_true))==2) else np.nan
    )

def best_thr_mcc(y_true, y_prob) -> float:
    thrs = np.linspace(0.01,0.99,99)
    mccs = [matthews_corrcoef(y_true,(y_prob>=t).astype(int)) for t in thrs]
    return float(thrs[int(np.argmax(mccs))])

@torch.no_grad()
def inference(model, loader):
    model.eval()
    ys, ps = [], []
    for x,y in loader:
        x = x.to(DEVICE); y=y.numpy()
        with torch.cuda.amp.autocast(enabled=False):
            logits = model(x)
        prob = torch.softmax(logits,1)[:,1].cpu().numpy()
        ys.append(y); ps.append(prob)
    return np.concatenate(ys), np.concatenate(ps)

def train_epoch(model, loader, crit, opt, scaler=None):
    model.train(); running=0.0
    pbar = tqdm(loader, leave=False)
    for x,y in pbar:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        if scaler is None:
            out = model(x); loss = crit(out,y); loss.backward(); opt.step()
        else:
            with torch.cuda.amp.autocast():
                out = model(x); loss = crit(out,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        running += loss.item()*x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running/len(loader.dataset)

# ═══════════════════════ main cross-val routine ═══════════════════
def main(args):
    root = Path(args.data_root)
    df_all = collect_images(root)
    print(f"Pooled data  : {len(df_all)} images, "
          f"{df_all.label.sum()} positive, {(df_all.label==0).sum()} negative")
    # -------------- external set (kept untouched) --------------
    ext_rows = []
    for lbl_dir in (root/"external_val"/"1", root/"external_val"/"2"):
        label = 1 if lbl_dir.name=="1" else 0
        for fp in lbl_dir.glob("*.jpg"):
            ext_rows.append((fp.as_posix(), label))
    df_ext = pd.DataFrame(ext_rows, columns=["path","label"])
    # -------------- transforms --------------
    train_tf, eval_tf = build_transforms(args.img_size)
    # -------------- K-fold split on patients --------------
    pids = df_all.groupby("pid").first().reset_index()
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof_y, oof_p = [], []          # out-of-fold pool for global threshold
    fold_metrics = []
    for fold, (train_pid, val_pid) in enumerate(skf.split(pids.pid, pids.label)):
        print(f"\n════════════ Fold {fold+1}/{args.folds} ════════════")
        val_pids = set(pids.pid.iloc[val_pid])
        df_train = df_all[~df_all.pid.isin(val_pids)].reset_index(drop=True)
        df_val   = df_all[df_all.pid.isin(val_pids)].reset_index(drop=True)
        # datasets / loaders
        ds_train = ImageDataset(df_train, train_tf)
        ds_val   = ImageDataset(df_val,   eval_tf)
        # weighted sampler
        counts = np.bincount(df_train.label)
        wts = 1./counts[df_train.label]
        sampler = WeightedRandomSampler(wts, len(wts), replacement=True)
        loader_tr = DataLoader(ds_train, batch_size=args.batch, sampler=sampler,
                               num_workers=2, pin_memory=(DEVICE.type=="cuda"))
        loader_val = DataLoader(ds_val, batch_size=args.batch,
                                num_workers=2, pin_memory=(DEVICE.type=="cuda"))
        # ---- model
        model = build_model(pretrained=not args.no_pretrain).to(DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
        class_w = torch.tensor(1./counts, dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        scaler = torch.cuda.amp.GradScaler() if DEVICE.type=="cuda" else None
        # ---- epochs
        for ep in range(args.epochs):
            _ = train_epoch(model, loader_tr, criterion, optim, scaler)
        # ---- validate
        y_val, p_val = inference(model, loader_val)
        oof_y.append(y_val); oof_p.append(p_val)
        # store for later per-fold metrics
        fold_metrics.append( (y_val, p_val, model.state_dict()) )
        print(f"Fold {fold} done: pos {y_val.sum()}/{len(y_val)}")
    # -------------- global threshold --------------
    oof_y = np.concatenate(oof_y); oof_p = np.concatenate(oof_p)
    thr_global = best_thr_mcc(oof_y, oof_p)
    print("\n>>>> Global threshold (max MCC on OOF) =", round(thr_global,3))
    # -------------- final fold metrics --------------
    results = []
    for fold,(y_val,p_val,sdict) in enumerate(fold_metrics):
        y_pred = (p_val>=thr_global).astype(int)
        res = mtrx(y_val, y_pred, p_val)
        res['fold']=fold; results.append(res)
    res_df = pd.DataFrame(results)
    print("\nPer-fold metrics (threshold =",round(thr_global,3),")")
    print(res_df[['fold','accuracy','specificity','recall','mcc']])
    print("\nMean ± SD MCC:", res_df.mcc.mean().round(3),
          "±", res_df.mcc.std(ddof=0).round(3))
    # -------------- external set --------------
    ds_ext = ImageDataset(df_ext, eval_tf)
    loader_ext = DataLoader(ds_ext, batch_size=args.batch,
                            num_workers=2, pin_memory=(DEVICE.type=="cuda"))
    # simple model ensemble = average probs from the K checkpoints
    all_probs = []
    for _,_,sdict in fold_metrics:
        m = build_model(pretrained=False).to(DEVICE)
        m.load_state_dict(sdict); m.eval()
        y_ext, p_ext = inference(m, loader_ext)   # y_ext identical every loop
        all_probs.append(p_ext)
    p_ext_mean = np.mean(all_probs, axis=0)
    y_pred_ext = (p_ext_mean >= thr_global).astype(int)
    ext_metrics = mtrx(y_ext, y_pred_ext, p_ext_mean)
    print("\nExternal-set metrics (ensemble, thr global)")
    for k in ['accuracy','precision','recall','specificity',
              'balanced_accuracy','f1','mcc','roc_auc']:
        print(f"{k:18s} {ext_metrics[k]:.3f}")

# ═══════════════════════ argparse & CLI ═══════════════════════════
def parse():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data_root', default='.', type=str)
    ap.add_argument('--epochs',    default=EPOCHS, type=int)
    ap.add_argument('--batch',     default=BATCH_SIZE, type=int)
    ap.add_argument('--img_size',  default=IMG_SIZE, type=int)
    ap.add_argument('--lr',        default=LR, type=float)
    ap.add_argument('--weight_decay', default=WEIGHT_DECAY, type=float)
    ap.add_argument('--folds',     default=N_FOLDS, type=int)
    ap.add_argument('--no_pretrain', action='store_true')
    return ap.parse_args()

if __name__=="__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main(parse())