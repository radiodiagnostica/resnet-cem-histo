#!/usr/bin/env python3
"""
Hormone-receptor (+/–) mammogram classification
───────────────────────────────────────────────
• patient-level, stratified k-fold CV
• focal loss for class imbalance (HR– = minority)
• early stop / checkpoint by PR-AUC
• out-of-fold isotonic calibration
• threshold picked for recall ≥ R* & max-F1
• external ensemble evaluation
"""

from __future__ import annotations
import argparse, random, warnings
from pathlib import Path
from typing   import List, Tuple, Dict, Any

import numpy  as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision      import transforms, models, datasets
from torchvision.models import ResNet18_Weights
from sklearn.metrics  import (average_precision_score as pr_auc,
                              precision_score, recall_score,
                              f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from tqdm.auto import tqdm

# ────────────────────────────── hyper-params ─────────────────────────────
IMG_SIZE      = 384
BATCH_SIZE    = 16
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
SEED          = 42
N_BOOT        = 2_000
N_PERM        = 2_000

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
if DEVICE.type == "mps":
    torch.set_float32_matmul_precision('high')
print("Running on:", DEVICE)

# ══════════════════ metrics block (unchanged) ════════════════════════════
def _to_pred(p:np.ndarray, thr:float)->np.ndarray: return (p>=thr).astype(int)

def _confusion_mtx(y,yhat):
    tp=((y==1)&(yhat==1)).sum(1); fp=((y==0)&(yhat==1)).sum(1)
    fn=((y==1)&(yhat==0)).sum(1); tn=((y==0)&(yhat==0)).sum(1)
    return tp,fp,fn,tn

def _metrics_from_conf(tp,fp,fn,tn):
    n=tp+fp+fn+tn
    acc=(tp+tn)/n
    prec=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)!=0)
    rec =np.divide(tp,tp+fn,out=np.zeros_like(tp,dtype=float),where=(tp+fn)!=0)
    f1  =np.divide(2*prec*rec,prec+rec,out=np.zeros_like(tp,dtype=float),
                   where=(prec+rec)!=0)
    denom=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc =np.divide(tp*tn-fp*fn,denom,out=np.zeros_like(tp,dtype=float),
                   where=denom!=0)
    bal =(rec + np.divide(tn,tn+fp,out=np.zeros_like(tp,dtype=float),
                          where=(tn+fp)!=0)) / 2
    return acc,prec,rec,f1,mcc,bal

def bootstrap_and_perm(y, p, thr,
                       n_boot=N_BOOT, n_perm=N_PERM, alpha=0.05):
    rng=np.random.default_rng(SEED)
    y,p=np.asarray(y),np.asarray(p); pred=_to_pred(p,thr)

    tp,fp,fn,tn=_confusion_mtx(y[None,:],pred[None,:])
    obs=list(m.item() for m in _metrics_from_conf(tp,fp,fn,tn))
    obs_auc=roc_auc_score(y,p) if len(np.unique(y))==2 else 0.5

    boot_idx=rng.integers(0,len(y),(n_boot,len(y)))
    tp,fp,fn,tn=_confusion_mtx(y[boot_idx],pred[boot_idx])
    boot=_metrics_from_conf(tp,fp,fn,tn)
    ci_lo=[np.percentile(b,100*alpha/2) for b in boot]
    ci_hi=[np.percentile(b,100*(1-alpha/2)) for b in boot]
    boot_auc=[roc_auc_score(y[i],p[i])
              if len(np.unique(y[i]))==2 else 0.5 for i in boot_idx]
    auc_lo,auc_hi=np.percentile(boot_auc,[100*alpha/2,100*(1-alpha/2)])

    perm_idx=np.argsort(rng.random((n_perm,len(y))),1)
    py=y[perm_idx]
    tp,fp,fn,tn=_confusion_mtx(py,pred[None,:])
    perm=_metrics_from_conf(tp,fp,fn,tn)
    pvals=[((perm_k>=obs_k).sum()+1)/(n_perm+1)
           for perm_k,obs_k in zip(perm,obs)]
    perm_auc=[roc_auc_score(py[i],p) if len(np.unique(py[i]))==2 else -np.inf
              for i in range(n_perm)]
    p_auc=((np.array(perm_auc)>=obs_auc).sum()+1)/(n_perm+1)

    names=['accuracy','precision','recall','f1','mcc','balanced_accuracy']
    out:Dict[str,Any]={}
    for n,s,lo,hi,pv in zip(names,obs,ci_lo,ci_hi,pvals):
        out[n]=dict(score=float(s),ci_low=float(lo),ci_high=float(hi),p=float(pv))
    out['roc_auc']=dict(score=float(obs_auc),ci_low=float(auc_lo),
                        ci_high=float(auc_hi),p=float(p_auc))
    return out
# ------------------------------------------------------------------------

def best_f1_threshold(y,p):                # vanilla max-F1
    th=np.linspace(0.05,0.95,19)
    f=[f1_score(y,_to_pred(p,t),zero_division=0) for t in th]
    return float(th[int(np.argmax(f))])

def print_table(d,title):
    print(f"\n{title}:")
    print("{:20s} {:>7s} {:>18s} {:>9s}".format("metric","score","95% CI","p"))
    for k,v in d.items():
        print("{:20s} {:7.3f} [{:5.3f},{:5.3f}] {:9.4f}".format(
            k,v['score'],v['ci_low'],v['ci_high'],v['p']))

# ═════════════════ data utilities ═══════════════════════════════════════
def get_pid(p:Path)->str:
    stem=p.stem
    for pref in("train_","val_"):
        if stem.startswith(pref): stem=stem[len(pref):]
    return stem.rsplit("_",1)[0]

class FileDS(Dataset):
    def __init__(self,samples,tf):
        self.samples=samples; self.tf=tf
        self.loader=datasets.folder.default_loader
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        path,lbl=self.samples[i]; img=self.loader(path); img=self.tf(img)
        return img,lbl

def make_loader(samples,tf,batch,train):
    ds=FileDS(samples,tf)
    shuffle=train
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,
                      num_workers=2,pin_memory=DEVICE.type=='cuda',
                      persistent_workers=True)

def build_tf(sz=IMG_SIZE):
    tr=transforms.Compose([
        transforms.Resize(int(sz*1.1)),
        transforms.RandomResizedCrop(sz,scale=(0.9,1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ev=transforms.Compose([
        transforms.Resize(int(sz*1.1)), transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return tr,ev

# ═════════════════ model & focal loss ═══════════════════════════════════
def build_model(pretrained=True):
    w=ResNet18_Weights.DEFAULT if pretrained else None
    m=models.resnet18(weights=w)
    m.fc=nn.Linear(m.fc.in_features,2)
    nn.init.xavier_uniform_(m.fc.weight); nn.init.zeros_(m.fc.bias)
    return m

def focal_loss(logits,targets,alpha,gamma=2.0):
    ce=nn.functional.cross_entropy(logits,targets,reduction='none',weight=alpha)
    pt=torch.softmax(logits,1)[range(len(targets)),targets]
    return torch.mean(((1-pt)**gamma)*ce)

@torch.no_grad()
def infer(model,loader)->Tuple[np.ndarray,np.ndarray]:
    model.eval(); ys,ps=[],[]
    for x,y in loader:
        x=x.to(DEVICE)
        prob=torch.softmax(model(x),1)[:,0]   # column-0 = HR-positive
        ys.append(y.numpy()); ps.append(prob.cpu().numpy())
    return np.concatenate(ys),np.concatenate(ps)

# ═════════════════ train one fold ═══════════════════════════════════════
def train_fold(fid,train_s,val_s,tf_tr,tf_ev,args):
    ld_tr=make_loader(train_s,tf_tr,args.batch_size,train=True)
    ld_va=make_loader(val_s, tf_ev,args.batch_size,train=False)

    # focal-loss α:  [HR+ weight, HR– weight]
    alpha=torch.tensor([1.0, args.neg_alpha],device=DEVICE)

    model=build_model(pretrained=not args.no_pretrain).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best_pr, no_imp=0.0,0
    for epoch in range(1,args.max_epochs+1):
        model.train(); running=0.0
        pbar=tqdm(ld_tr,leave=False,desc=f"[fold {fid}] {epoch:02d}")
        for x,y in pbar:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            out=model(x)
            loss=focal_loss(out,y,alpha,gamma=2.0)
            loss.backward(); opt.step()
            running+=loss.item()*x.size(0)
            pbar.set_postfix(loss=loss.item())

        yv,pv=infer(model,ld_va)
        pr=pr_auc(yv,pv)
        if pr>best_pr:
            best_pr,no_imp=pr,0
            torch.save({'state_dict':model.state_dict()},f"best_fold{fid}.pt")
        else: no_imp+=1
        if args.patience and no_imp>=args.patience:
            print(f"fold {fid}: early-stop @ epoch {epoch}")
            break
    print(f"fold {fid} done.  VAL PR-AUC={best_pr:.3f}")
    return best_pr

# ═════════════════════════ main ════════════════════════════════════════
def main(a):
    tf_tr,tf_ev=build_tf(a.img_size)

    # collect train+val samples
    samples=[]
    for sp in("train","val"):
        ds=datasets.ImageFolder(Path(a.data_root)/sp)
        samples+=ds.samples
    paths=np.array([p for p,_ in samples])
    labels=np.array([l for _,l in samples])
    groups=np.array([get_pid(Path(p)) for p in paths])

    cv=StratifiedGroupKFold(a.n_folds,shuffle=True,random_state=SEED)
    fold_prs=[]; oof_y,oof_p=[],[]

    for fid,(tr,va) in enumerate(cv.split(paths,labels,groups),1):
        tr_s=[samples[i] for i in tr]; va_s=[samples[i] for i in va]
        pr=train_fold(fid,tr_s,va_s,tf_tr,tf_ev,a)
        fold_prs.append(pr)

        # OOF preds
        va_loader=make_loader(va_s,tf_ev,a.batch_size,train=False)
        ck=torch.load(f"best_fold{fid}.pt",map_location=DEVICE)
        model=build_model(False).to(DEVICE); model.load_state_dict(ck['state_dict'])
        y,p=infer(model,va_loader)
        oof_y.append(y); oof_p.append(p)

    oof_y=np.concatenate(oof_y); oof_p=np.concatenate(oof_p)

    # isotonic calibration
    iso=IsotonicRegression(out_of_bounds='clip').fit(oof_p,oof_y)
    oof_p_cal=iso.transform(oof_p)

    # grid search for thr (recall ≥ min_recall & max F1)
    best_t,best_f1=0.05,0.0
    for t in np.linspace(0.05,0.95,19):
        yp=_to_pred(oof_p_cal,t)
        rec=recall_score(oof_y,yp,zero_division=0)
        if rec>=a.min_recall:
            f1=f1_score(oof_y,yp,zero_division=0)
            if f1>best_f1: best_f1,best_t=f1,t
    print(f"\nGlobal threshold = {best_t:.2f} (rec≥{a.min_recall} & max-F1={best_f1:.3f})")

    # ── external evaluation
    ext_ds=datasets.ImageFolder(Path(a.data_root)/"external_val",tf_ev)
    ext_loader=DataLoader(ext_ds,batch_size=a.batch_size,shuffle=False,
                          num_workers=2,pin_memory=DEVICE.type=='cuda')
    probs=[]
    for fid in range(1,a.n_folds+1):
        ck=torch.load(f"best_fold{fid}.pt",map_location=DEVICE)
        m=build_model(False).to(DEVICE); m.load_state_dict(ck['state_dict'])
        _,p=infer(m,ext_loader); probs.append(p)
    p_ext_mean=iso.transform(np.mean(probs,0))
    y_ext=np.array([l for _,l in ext_ds.samples])
    ext_metrics=bootstrap_and_perm(y_ext,p_ext_mean,best_t)

    # ── report
    print("\n────────── CV summary (PR-AUC) ──────────")
    for i,pr in enumerate(fold_prs,1): print(f"fold {i}: {pr:.3f}")
    print(f"CV mean±std PR-AUC: {np.mean(fold_prs):.3f} ± {np.std(fold_prs):.3f}")
    print_table(ext_metrics,"EXTERNAL (HR-positive as positive class)")

# ═════════════════════ argparse CLI ═════════════════════════════════════
def cli():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data_root',default='.',type=str)
    ap.add_argument('--n_folds',default=5,type=int)
    ap.add_argument('--max_epochs',default=30,type=int)
    ap.add_argument('--patience',default=10,type=int,
                    help='early-stop patience on PR-AUC')
    ap.add_argument('--batch_size',default=BATCH_SIZE,type=int)
    ap.add_argument('--img_size',default=IMG_SIZE,type=int)
    ap.add_argument('--lr',default=LR,type=float)
    ap.add_argument('--weight_decay',default=WEIGHT_DECAY,type=float)
    ap.add_argument('--neg_alpha',default=4.0,type=float,
                    help='class weight for HR-negative in focal loss')
    ap.add_argument('--min_recall',default=0.60,type=float,
                    help='minimum recall when choosing threshold')
    ap.add_argument('--no_pretrain',action='store_true')
    return ap.parse_args()

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=UserWarning)
    main(cli())