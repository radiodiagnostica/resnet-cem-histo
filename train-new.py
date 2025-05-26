#!/usr/bin/env python3
"""
Hormone-receptor (+/–) classification on cropped mammograms
────────────────────────────────────────────────────────────
  • RadImageNet backbones (resnet18 / 50, densenet121)
  • Apple-Silicon friendly memory tweaks
        – fp16 autocast on MPS      (--fp16)
        – channels-last tensors     (--channels_last)
        – gradient checkpointing    (--grad_ckpt)
        – gradient accumulation     (--grad_accum N)
  • Optional backbone freezing      (--freeze_backbone)

Directory layout is unchanged:

data_root/
 ├── train/         (sub-folders 0, 1)
 ├── val/
 └── external_val/
"""
from __future__ import annotations
import argparse, random, time, warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint_sequential
from torchvision import transforms, datasets, models
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score, roc_auc_score
)

# ──────────────────────────────── Hyper-params ────────────────────────────────
IMG_SIZE, LR, WEIGHT_DECAY = 384, 1e-4, 1e-4
EPOCHS, BATCH_SIZE = 30, 16
N_BOOT, N_PERM, SEED = 2_000, 2_000, 42
EARLY_STOP = 0      # 0 → disabled
# ──────────────────────────────────────────────────────────────────────────────

# ────────────────────────────── Reproducibility ───────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark      = False
set_seed()

def get_device() -> torch.device:
    if torch.cuda.is_available():             return torch.device("cuda")
    if torch.backends.mps.is_available():     return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print("Device:", DEVICE)
if DEVICE.type == "mps":
    torch.set_float32_matmul_precision("high")

# ──────────────────────── Metric & statistical helpers ────────────────────────
def _to_pred(p: np.ndarray, thr: float) -> np.ndarray:
    return (p >= thr).astype(int)

def compute_metrics(y, p, t=0.5):
    ŷ = _to_pred(p, t)
    f = lambda *a, **kw: np.nan
    try: roc = roc_auc_score(y, p)
    except ValueError: roc = np.nan
    return dict(
        accuracy           = accuracy_score         (y, ŷ),
        precision          = precision_score        (y, ŷ, zero_division=0),
        recall             = recall_score           (y, ŷ, zero_division=0),
        f1                 = f1_score               (y, ŷ, zero_division=0),
        mcc                = matthews_corrcoef      (y, ŷ),
        balanced_accuracy  = balanced_accuracy_score(y, ŷ),
        roc_auc            = roc
    )

def bootstrap_ci(fn, y, p, t, n=N_BOOT, α=.05):
    rng, idx, vals = np.random.default_rng(SEED), np.arange(len(y)), []
    while len(vals)<n:
        s = rng.choice(idx, size=len(idx), replace=True)
        try: vals.append(fn(y[s], p[s], t))
        except ValueError: pass
    lo, hi = np.percentile(vals, [100*α/2, 100*(1-α/2)])
    return float(lo), float(hi)

def permutation_p(fn, y, p, t, n=N_PERM):
    rng, obs, cnt = np.random.default_rng(SEED), fn(y,p,t), 0
    for _ in range(n):
        try:
            if fn(rng.permutation(y), p, t) >= obs: cnt += 1
        except ValueError: pass
    return (cnt+1)/(n+1)

def full_metrics(y, p, t):
    met = [
        ('accuracy', lambda y,p,t: accuracy_score(y, _to_pred(p,t))),
        ('precision',lambda y,p,t: precision_score(y,_to_pred(p,t),zero_division=0)),
        ('recall',   lambda y,p,t: recall_score   (y,_to_pred(p,t),zero_division=0)),
        ('f1',       lambda y,p,t: f1_score       (y,_to_pred(p,t),zero_division=0)),
        ('mcc',      lambda y,p,t: matthews_corrcoef(y,_to_pred(p,t))),
        ('balanced_accuracy', lambda y,p,t: balanced_accuracy_score(y,_to_pred(p,t))),
        ('roc_auc',  lambda y,p,t: roc_auc_score(y,p)),
    ]
    out={}
    for n,f in met:
        try: s=f(y,p,t)
        except ValueError: s=np.nan
        lo,hi=bootstrap_ci(f,y,p,t); pval=permutation_p(f,y,p,t)
        out[n]={'score':s,'ci_low':lo,'ci_high':hi,'p':pval}
    return out

def best_f1_threshold(y,p):
    th=np.linspace(.05,.95,19);f=[f1_score(y,_to_pred(p,t),zero_division=0) for t in th]
    return float(th[int(np.argmax(f))])

def print_table(d:Dict[str,Any], title:str):
    print(f"\n{title}")
    print("{:20s} {:>8s} {:>18s} {:>12s}".format("metric","score","95% CI","p"))
    for k,v in d.items():
        print("{:20s} {:8.3f} [{:5.3f},{:5.3f}] {:12.4f}".format(
            k,v['score'],v['ci_low'],v['ci_high'],v['p']))

# ───────────────────────────── Data & transforms ──────────────────────────────
def build_transforms(sz):
    norm = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train = transforms.Compose([
        transforms.Resize(int(sz*1.1)),
        transforms.RandomResizedCrop(sz,scale=(.9,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(.1,.1,.1,.05),
        transforms.ToTensor(), norm])
    eval_ = transforms.Compose([
        transforms.Resize(int(sz*1.1)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(), norm])
    return train, eval_

def _seed_worker(wid): np.random.seed(SEED+wid); random.seed(SEED+wid)

def make_loader(root, split, tf, bs, balance=False, workers=2, prefetch=1):
    ds=datasets.ImageFolder(Path(root)/split, transform=tf)
    sampler=None; shuffle=(split=="train")
    if balance and split=="train":
        tgt=[s[1] for s in ds.samples]
        w=1./np.bincount(tgt)[tgt]
        sampler=WeightedRandomSampler(w,len(w),replacement=True); shuffle=False
    pin=DEVICE.type=="cuda"
    kw=dict(batch_size=bs,shuffle=shuffle,sampler=sampler,
            num_workers=workers,worker_init_fn=_seed_worker,pin_memory=pin)
    if workers>0: kw.update(prefetch_factor=prefetch,persistent_workers=True)
    return DataLoader(ds,**kw), ds

# ─────────────────────────── Backbone / model builder ─────────────────────────
def build_model(arch='resnet18', classes=2, imagenet=True,
                rad_ckpt:str|None=None, freeze=False):
    if arch=='resnet18':
        m=models.resnet18(weights=ResNet18_Weights.DEFAULT if imagenet and not rad_ckpt else None)
        in_f=m.fc.in_features; m.fc=nn.Linear(in_f,classes)
    elif arch=='resnet50':
        m=models.resnet50(weights=ResNet50_Weights.DEFAULT if imagenet and not rad_ckpt else None)
        in_f=m.fc.in_features; m.fc=nn.Linear(in_f,classes)
    elif arch=='densenet121':
        m=models.densenet121(weights=DenseNet121_Weights.DEFAULT if imagenet and not rad_ckpt else None)
        in_f=m.classifier.in_features; m.classifier=nn.Linear(in_f,classes)
    else: raise ValueError(arch)

    if rad_ckpt:
        print("Loading RadImageNet:",rad_ckpt)
        sd=torch.load(rad_ckpt,map_location='cpu')
        skip=('fc','classifier')
        sd={k:v for k,v in sd.items() if not any(k.startswith(s) for s in skip)}
        miss,unexp=m.load_state_dict(sd,strict=False)
        print(f"  loaded {len(sd)} params | missing={len(miss)}")
    if freeze:
        for n,p in m.named_parameters():
            if not (n.startswith('fc') or n.startswith('classifier')):
                p.requires_grad=False
        print("Backbone frozen.")

    def _init(l):
        if isinstance(l,nn.Linear):
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)
    m.apply(_init)
    return m

# ──────────────────── Gradient checkpointing utilities ────────────────────────
def apply_checkpoint(model, arch):
    if arch.startswith('resnet'):
        for blk in ['layer1','layer2','layer3','layer4']:
            seq = getattr(model, blk)
            setattr(model, blk, checkpoint_sequential(list(seq), len(seq)//2 or 1))
    elif arch=='densenet121':
        for i in range(1,5):
            blk=getattr(model.features,f"denseblock{i}")
            setattr(model.features,f"denseblock{i}",
                    checkpoint_sequential(list(blk), len(blk)//2 or 1))
    print("Gradient checkpointing enabled.")
    return model

# ───────────────────────────── Train / Evaluate ───────────────────────────────
def train_one_epoch(model, loader, crit, opt, fp16, accum, chan_last):
    model.train(); running=0
    pbar=tqdm(loader,desc='train',leave=False)
    opt.zero_grad(); dev=DEVICE.type
    autocast=lambda: torch.autocast(device_type=dev,dtype=torch.float16,enabled=fp16)
    for step,(x,y) in enumerate(pbar):
        x=x.to(DEVICE, memory_format=torch.channels_last if chan_last else torch.contiguous_format)
        y=y.to(DEVICE)
        with autocast():
            out=model(x); loss=crit(out,y)/accum
        loss.backward()
        if (step+1)%accum==0:
            opt.step(); opt.zero_grad()
        running+=loss.item()*x.size(0)*accum
        pbar.set_postfix(loss=loss.item()*accum)
    return running/len(loader.dataset)

@torch.no_grad()
def inference(model, loader, fp16, chan_last):
    model.eval(); y_true,y_prob=[],[]
    autocast=lambda: torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=fp16)
    for x,y in loader:
        x=x.to(DEVICE, memory_format=torch.channels_last if chan_last else torch.contiguous_format)
        with autocast():
            logits=model(x)
        prob=torch.softmax(logits,1)[:,1]
        y_true.append(y.numpy()); y_prob.append(prob.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)

# ─────────────────────────────────── main ─────────────────────────────────────
def main(a):
    tf_tr,tf_ev=build_transforms(a.img_size)
    train_ld,train_ds=make_loader(a.data_root,'train',tf_tr,a.batch_size,balance=True)
    val_ld, _        =make_loader(a.data_root,'val',tf_ev,a.batch_size)
    ext_ld, _        =make_loader(a.data_root,'external_val',tf_ev,a.batch_size)

    tgt=np.array([y for _,y in train_ds.samples])
    cw=1./torch.tensor(np.bincount(tgt),dtype=torch.float32)
    crit=nn.CrossEntropyLoss(weight=cw.to(DEVICE))

    model=build_model(a.arch,2,not a.no_pretrain,a.rad_ckpt or None,a.freeze_backbone).to(DEVICE)
    if a.channels_last: model=model.to(memory_format=torch.channels_last)
    if a.grad_ckpt: model=apply_checkpoint(model,a.arch)
    if a.torch_compile and hasattr(torch,'compile'): model=torch.compile(model,mode='reduce-overhead')

    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),
                          lr=a.lr,weight_decay=a.weight_decay)

    best_f1, best_thr, no_imp, hist = 0.,.5,0,[]
    for epoch in range(a.epochs):
        t0=time.time()
        tr_loss=train_one_epoch(model,train_ld,crit,opt,a.fp16,a.grad_accum,a.channels_last)
        yv,pv=inference(model,val_ld,a.fp16,a.channels_last); best_thr=best_f1_threshold(yv,pv)
        vm=full_metrics(yv,pv,best_thr)
        ye,pe=inference(model,ext_ld,a.fp16,a.channels_last); em=full_metrics(ye,pe,best_thr)

        cur_f1=vm['f1']['score']
        if cur_f1>best_f1:
            best_f1=cur_f1; torch.save({'state_dict':model.state_dict(),'thr':best_thr},'best_model.pt'); no_imp=0
        else: no_imp+=1

        row={'epoch':epoch,'train_loss':tr_loss,'thr':best_thr}
        for k,v in vm.items(): row[f'val_{k}']=v['score']
        for k,v in em.items(): row[f'ext_{k}']=v['score']
        hist.append(row); pd.DataFrame(hist).to_csv('training_log.csv',index=False)

        print(f"\nEpoch {epoch+1:02d}/{a.epochs} | loss {tr_loss:.4f} | thr {best_thr:.2f} | time {time.time()-t0:4.1f}s")
        print_table(vm,"VAL"); print_table(em,"EXT")

        if EARLY_STOP and no_imp>=EARLY_STOP:
            print("Early stopping."); break
    print("\nBest F1(val):",best_f1,"  model → best_model.pt")

# ──────────────────────────────── CLI ─────────────────────────────────────────
def parse():
    F=argparse.ArgumentDefaultsHelpFormatter
    ap=argparse.ArgumentParser(formatter_class=F)
    ap.add_argument('--data_root',default='.',type=str)
    ap.add_argument('--epochs',default=EPOCHS,type=int)
    ap.add_argument('--batch_size',default=BATCH_SIZE,type=int)
    ap.add_argument('--img_size',default=IMG_SIZE,type=int)
    ap.add_argument('--lr',default=LR,type=float)
    ap.add_argument('--weight_decay',default=WEIGHT_DECAY,type=float)
    ap.add_argument('--no_pretrain',action='store_true')
    ap.add_argument('--torch_compile',action='store_true')

    # backbone / RadImageNet
    ap.add_argument('--arch',default='resnet18',choices=['resnet18','resnet50','densenet121'])
    ap.add_argument('--rad_ckpt',default='',type=str)
    ap.add_argument('--freeze_backbone',action='store_true')

    # memory-saving flags
    ap.add_argument('--fp16',action='store_true',help='autocast float16 on MPS/CPU')
    ap.add_argument('--channels_last',action='store_true',help='NHWC tensors')
    ap.add_argument('--grad_ckpt',action='store_true',help='gradient checkpoint backbone')
    ap.add_argument('--grad_accum',default=1,type=int,help='steps to accumulate gradients')

    return ap.parse_args()

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=UserWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    main(parse())