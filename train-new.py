#!/usr/bin/env python3
"""
Hormone-receptor (+/–) classification – CV + calibration
──────────────────────────────────────────────────────────
Main features:
  ✓ 5-fold CV with early-stopping (patience=3)
  ✓ freeze-then-unfreeze schedule
  ✓ optional focal-loss
  ✓ optional temperature-scaling calibration
  ✓ single threshold per fold (F1-optimised), median used on external set
  ✓ external evaluation with ensemble of calibrated models
"""

from __future__ import annotations
import argparse, copy, os, random, time, warnings
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ─────────────────────────── Hyper-parameters ────────────────────────────
IMG_SIZE       = 384
LR_HEAD        = 1e-4
LR_BACKBONE    = 1e-5
WEIGHT_DECAY   = 1e-4
EPOCHS         = 30
BATCH_SIZE     = 16
N_BOOT         = 2_000
N_PERM         = 2_000
SEED           = 42
PATIENCE       = 3
N_FOLDS        = 5
FROZEN_EPOCHS  = 2
# ──────────────────────────────────────────────────────────────────────────

# ───────── reproducibility & device ──────────────────────────────────────
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = get_device()
if DEVICE.type == 'mps':
    torch.set_float32_matmul_precision('high')
print("Running on:", DEVICE)

# ───────────────────────── stats utilities ───────────────────────────────
def _to_pred(p: np.ndarray, thr: float):
    return (p >= thr).astype(int)

def _confusion_mtx(y_true, y_pred):
    tp = np.sum((y_true==1)&(y_pred==1), axis=1)
    fp = np.sum((y_true==0)&(y_pred==1), axis=1)
    fn = np.sum((y_true==1)&(y_pred==0), axis=1)
    tn = np.sum((y_true==0)&(y_pred==0), axis=1)
    return tp,fp,fn,tn

def _metrics_from_conf(tp,fp,fn,tn):
    n   = tp+fp+fn+tn
    acc = (tp+tn)/n
    prec= np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)!=0)
    rec = np.divide(tp,tp+fn,out=np.zeros_like(tp,dtype=float),where=(tp+fn)!=0)
    f1  = np.divide(2*prec*rec,prec+rec,out=np.zeros_like(tp,dtype=float),where=(prec+rec)!=0)
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = np.divide(tp*tn-fp*fn,denom,out=np.zeros_like(tp,dtype=float),where=denom!=0)
    bal = (rec + np.divide(tn,tn+fp,out=np.zeros_like(tp,dtype=float),where=(tn+fp)!=0))/2
    return acc,prec,rec,f1,mcc,bal

def bootstrap_and_perm(y_true,y_prob,thr,n_boot=N_BOOT,n_perm=N_PERM,alpha=0.05):
    rng    = np.random.default_rng(SEED)
    y_true = np.asarray(y_true); y_prob=np.asarray(y_prob)
    y_pred = (y_prob>=thr).astype(int)

    # observed
    obs_tp,obs_fp,obs_fn,obs_tn=_confusion_mtx(y_true[np.newaxis,:],y_pred[np.newaxis,:])
    obs_metrics=_metrics_from_conf(obs_tp,obs_fp,obs_fn,obs_tn)
    obs_metrics=[m.item() for m in obs_metrics]
    obs_roc = roc_auc_score(y_true,y_prob)

    # bootstrap
    boot_idx=rng.integers(0,len(y_true),(n_boot,len(y_true)))
    tp,fp,fn,tn=_confusion_mtx(y_true[boot_idx],y_pred[boot_idx])
    boot_metrics=_metrics_from_conf(tp,fp,fn,tn)
    ci_low=[np.percentile(b,100*alpha/2) for b in boot_metrics]
    ci_high=[np.percentile(b,100*(1-alpha/2)) for b in boot_metrics]
    roc_boot=[roc_auc_score(y_true[i],y_prob[i]) for i in boot_idx]
    roc_ci_low,roc_ci_high=np.percentile(roc_boot,[100*alpha/2,100*(1-alpha/2)])

    # permutation
    perm_idx=np.argsort(rng.random((n_perm,len(y_true))),axis=1)
    perm_y=y_true[perm_idx]
    tp,fp,fn,tn=_confusion_mtx(perm_y,y_pred[np.newaxis,:])
    perm_metrics=_metrics_from_conf(tp,fp,fn,tn)
    p_vals=[((np.sum(p>=o)+1)/(n_perm+1)) for p,o in zip(perm_metrics,obs_metrics)]
    roc_perm=[(roc_auc_score(py,y_prob) if np.unique(py).size==2 else -np.inf) for py in perm_y]
    roc_p=(np.sum(np.array(roc_perm)>=obs_roc)+1)/(n_perm+1)

    names=['accuracy','precision','recall','f1','mcc','balanced_accuracy']
    out={}
    for n,s,lo,hi,p in zip(names,obs_metrics,ci_low,ci_high,p_vals):
        out[n]={'score':float(s),'ci_low':float(lo),'ci_high':float(hi),'p':float(p)}
    out['roc_auc']={'score':float(obs_roc),'ci_low':float(roc_ci_low),
                    'ci_high':float(roc_ci_high),'p':float(roc_p)}
    return out

def best_f1_threshold(y_true,y_prob):
    thrs=np.linspace(0.05,0.95,19)
    f1s =[f1_score(y_true,_to_pred(y_prob,t),zero_division=0) for t in thrs]
    return float(thrs[int(np.argmax(f1s))])

# ─────────────────────────── loss options ────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self,gamma=2,alpha=None,reduction='mean'):
        super().__init__()
        self.gamma,self.alpha,self.reduction=gamma,alpha,reduction
        self.ce=nn.CrossEntropyLoss(weight=alpha,reduction='none')
    def forward(self,logits,targets):
        ce=self.ce(logits,targets)
        pt=torch.exp(-ce)
        loss=(1-pt)**self.gamma*ce
        return loss.mean() if self.reduction=='mean' else loss.sum()

# ─────────────────────────── transforms / data ───────────────────────────
def build_transforms(img_size:int):
    train_tf=transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf=transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf,eval_tf

def _seed_worker(wid):
    np.random.seed(SEED+wid); random.seed(SEED+wid)

def _subset_targets(ds)->list[int]:
    if hasattr(ds,'samples'):
        return [y for _,y in ds.samples]
    if isinstance(ds,Subset):
        return [ds.dataset.samples[i][1] for i in ds.indices]
    raise TypeError("Unsupported dataset type")

def make_loader(dataset,split,tf,batch_size,balance,workers=2,prefetch=1)->DataLoader:
    ds=copy.deepcopy(dataset)
    if isinstance(ds,Subset): ds.dataset.transform=tf
    else: ds.transform=tf

    if balance and split=='train':
        targets=_subset_targets(ds)
        class_counts=np.bincount(targets)
        weights=1.0/class_counts[targets]
        sampler=WeightedRandomSampler(weights,len(weights),replacement=True)
        shuffle=False
    else:
        sampler=None; shuffle=(split=='train')
    pin_mem=DEVICE.type=='cuda'
    kwargs=dict(batch_size=batch_size,shuffle=shuffle,sampler=sampler,
                num_workers=workers,worker_init_fn=_seed_worker,
                pin_memory=pin_mem)
    if workers>0: kwargs.update(prefetch_factor=prefetch,persistent_workers=True)
    return DataLoader(ds,**kwargs)

# ───────────────────────────── model utils ───────────────────────────────
def build_model(num_classes=2,pretrained=True):
    w=ResNet18_Weights.DEFAULT if pretrained else None
    m=models.resnet18(weights=w)
    m.fc=nn.Linear(m.fc.in_features,num_classes)
    nn.init.xavier_uniform_(m.fc.weight); nn.init.zeros_(m.fc.bias)
    return m

def freeze_backbone(model:nn.Module,freeze:bool=True):
    for n,p in model.named_parameters():
        if not n.startswith('fc.'): p.requires_grad_(not freeze)

# ───────────────────── temperature scaling helper ────────────────────────
class _TempScaler(nn.Module):
    def __init__(self): super().__init__(); self.temp=nn.Parameter(torch.ones([])*1.5)
    def forward(self,logits): return logits/self.temp

def temperature_scale(logits:torch.Tensor,labels:torch.Tensor)->float:
    logits,labels=logits.to(DEVICE),labels.to(DEVICE)
    scaler=_TempScaler().to(DEVICE)
    nll=nn.CrossEntropyLoss()
    optim=torch.optim.LBFGS([scaler.temp],lr=0.01,max_iter=50)
    def _closure():
        optim.zero_grad()
        loss=nll(scaler(logits),labels); loss.backward(); return loss
    optim.step(_closure)
    return scaler.temp.detach().cpu().item()

# ───────────────────────────── dataset loader ────────────────────────────
def load_internal_ds(root:Path)->datasets.ImageFolder:
    if (root/'train_val').is_dir():
        return datasets.ImageFolder(root/'train_val')
    if (root/'train').is_dir() and (root/'val').is_dir():
        ds_tr=datasets.ImageFolder(root/'train')
        ds_va=datasets.ImageFolder(root/'val')
        if ds_tr.class_to_idx!=ds_va.class_to_idx:
            raise ValueError("Class mapping mismatch train vs val")
        merged=copy.deepcopy(ds_tr)
        merged.samples=ds_tr.samples+ds_va.samples
        merged.imgs   =merged.samples
        return merged
    raise FileNotFoundError("Expected train_val/ or train/ + val/ inside data_root")

# ───────────────────────────── training loop ─────────────────────────────
def train_one_epoch(model,loader,criterion,optim,epoch):
    model.train(); running=0.0
    pbar=tqdm(loader,leave=False,desc=f"train(e{epoch})")
    for x,y in pbar:
        x,y=x.to(DEVICE),y.to(DEVICE)
        optim.zero_grad()
        loss=criterion(model(x),y)
        loss.backward(); optim.step()
        running+=loss.item()*x.size(0)
        pbar.set_postfix(loss=loss.item())
    return running/len(loader.dataset)

@torch.no_grad()
def inference(model,loader):
    model.eval(); ys,ps=[],[]
    for x,y in loader:
        x=x.to(DEVICE)
        logits=model(x)
        prob=torch.softmax(logits,1)[:,1]
        ys.append(y.numpy()); ps.append(prob.cpu().numpy())
    return np.concatenate(ys),np.concatenate(ps)

# ─────────────────────── single-fold routine ─────────────────────────────
def run_fold(fold:int,idx_tr:list[int],idx_v:list[int],full_ds,args)->tuple[float,float]:
    tf_train,tf_eval=build_transforms(args.img_size)
    tr_set,va_set=Subset(full_ds,idx_tr),Subset(full_ds,idx_v)
    tr_loader=make_loader(tr_set,'train',tf_train,args.batch_size,True)
    va_loader=make_loader(va_set,'val',tf_eval,args.batch_size,False)

    # weighted loss
    targets=[full_ds.samples[i][1] for i in idx_tr]
    cw=1./torch.tensor(np.bincount(targets),dtype=torch.float32).to(DEVICE)
    criterion=FocalLoss(gamma=2,alpha=cw) if args.focal else nn.CrossEntropyLoss(weight=cw)

    model=build_model(pretrained=not args.no_pretrain).to(DEVICE)
    freeze_backbone(model,True)

    head=[p for n,p in model.named_parameters() if n.startswith('fc.')]
    base=[p for n,p in model.named_parameters() if not n.startswith('fc.')]
    optim=torch.optim.AdamW([{'params':head,'lr':LR_HEAD},
                             {'params':base,'lr':LR_BACKBONE}],
                            weight_decay=WEIGHT_DECAY)

    best_f1, no_imp, best_wts = 0.0, 0, None
    for epoch in range(EPOCHS):
        if epoch==FROZEN_EPOCHS: freeze_backbone(model,False)
        train_one_epoch(model,tr_loader,criterion,optim,epoch)
        y_val,p_val=inference(model,va_loader)
        f1_now=f1_score(y_val,_to_pred(p_val,0.5),zero_division=0)
        if f1_now>best_f1:
            best_f1, best_wts, no_imp = f1_now, copy.deepcopy(model.state_dict()),0
        else:
            no_imp+=1
        if no_imp>=PATIENCE:
            print(f"Fold {fold} – early stop at epoch {epoch+1}"); break

    model.load_state_dict(best_wts)

    # logits collection for calibration
    logits_val,y_val=[],[]
    model.eval()
    with torch.no_grad():
        for x,y in va_loader:
            logits_val.append(model(x.to(DEVICE)).cpu())
            y_val.append(y.numpy())
    logits_val=torch.cat(logits_val); y_val=np.concatenate(y_val)

    temp_value=1.0
    if args.calibrate:
        temp_value=temperature_scale(logits_val,torch.tensor(y_val))
        print(f"Fold {fold}: temperature = {temp_value:.2f}")

    p_val=torch.softmax(logits_val/temp_value,1)[:,1].numpy()
    best_thr=best_f1_threshold(y_val,p_val)

    torch.save({'state_dict':model.state_dict(),
                'temp':temp_value,
                'thr':best_thr},
               f"best_fold{fold}.pt")
    print(f"Fold {fold}: best-F1={best_f1:.3f}, thr={best_thr:.2f}")
    return best_f1,best_thr

# ───────────────────────────────── Main ──────────────────────────────────
def main(args):
    root=Path(args.data_root)
    full_ds=load_internal_ds(root)
    y_full=np.array([y for _,y in full_ds.samples])
    skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)

    fold_thrs,fold_f1s=[],[]
    for f,(tr,va) in enumerate(skf.split(np.zeros(len(y_full)),y_full)):
        f1,thr=run_fold(f,list(tr),list(va),full_ds,args)
        fold_f1s.append(f1); fold_thrs.append(thr)

    median_thr=float(np.median(fold_thrs))
    print("\nCV done. Fold F1s:",[f"{x:.3f}" for x in fold_f1s])
    print("Median threshold =",median_thr)

    # external evaluation
    tf_eval=build_transforms(args.img_size)[1]
    ext_ds=datasets.ImageFolder(root/'external_val',transform=tf_eval)
    ext_loader=make_loader(ext_ds,'ext',tf_eval,args.batch_size,False)

    prob_agg,y_ext=[],None
    for f in range(N_FOLDS):
        ckpt=torch.load(f"best_fold{f}.pt",map_location=DEVICE)
        temp=ckpt.get('temp',1.0)
        model=build_model(pretrained=False).to(DEVICE)
        model.load_state_dict(ckpt['state_dict']); model.eval()

        ps_fold,ys=[],[]
        with torch.no_grad():
            for x,y in ext_loader:
                logits=model(x.to(DEVICE))/temp
                prob=torch.softmax(logits,1)[:,1]
                ps_fold.append(prob.cpu().numpy())
                ys.append(y.numpy())
        prob_agg.append(np.concatenate(ps_fold))
        if y_ext is None: y_ext=np.concatenate(ys)

    p_ext=np.mean(prob_agg,axis=0)
    ext_metrics=bootstrap_and_perm(y_ext,p_ext,median_thr)
    print("\nExternal-test metrics (median thr = %.2f)"%median_thr)
    for k,v in ext_metrics.items():
        print(f"{k:20s} {v['score']:.3f} "
              f"[{v['ci_low']:.3f},{v['ci_high']:.3f}]  p={v['p']:.4f}")

# ───────────────────────────── CLI parsing ───────────────────────────────
def parse_args():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data_root',default='.',type=str)
    ap.add_argument('--batch_size',default=BATCH_SIZE,type=int)
    ap.add_argument('--img_size',default=IMG_SIZE,type=int)
    ap.add_argument('--no_pretrain',action='store_true')
    ap.add_argument('--focal',action='store_true',help='Use focal loss')
    ap.add_argument('--calibrate',action='store_true',help='Temperature scaling')
    return ap.parse_args()

# ──────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    warnings.filterwarnings("ignore",category=UserWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    main(parse_args())