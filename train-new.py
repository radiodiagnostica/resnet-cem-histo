#!/usr/bin/env python3
"""
Hormone-receptor (+/–) classification
─────────────────────────────────────
• patient-level, stratified k-fold cross-validation
• fully-vectorised bootstrap CIs & permutation p-values
• external hold-out test
Folder layout (unchanged):
data_root/
 ├── train/1 … 2
 ├── val/1 … 2
 └── external_val/1 … 2
"""
from __future__ import annotations
import argparse, random, time, warnings
from pathlib import Path
from typing   import List, Tuple, Dict, Any

import numpy  as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision      import transforms, models, datasets
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics  import f1_score, roc_auc_score
from tqdm.auto import tqdm

# ─────────────────────────── globals ────────────────────────────
IMG_SIZE      = 384
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 16
SEED          = 42
N_BOOT        = 2_000
N_PERM        = 2_000
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

DEVICE = (torch.device("cuda")   if torch.cuda.is_available()      else
          torch.device("mps")    if torch.backends.mps.is_available() else
          torch.device("cpu"))
if DEVICE.type == "mps":
    torch.set_float32_matmul_precision('high')
print("Running on:", DEVICE)

# ════════════════════ 1.  Metric utilities ══════════════════════
def _to_pred(prob: np.ndarray, thr: float) -> np.ndarray:
    return (prob >= thr).astype(int)

def _confusion_mtx(y, yhat):
    tp = np.sum((y==1)&(yhat==1),1); fp = np.sum((y==0)&(yhat==1),1)
    fn = np.sum((y==1)&(yhat==0),1); tn = np.sum((y==0)&(yhat==0),1)
    return tp,fp,fn,tn

def _metrics_from_conf(tp,fp,fn,tn):
    n  = tp+fp+fn+tn
    acc = (tp+tn)/n
    prec = np.divide(tp, tp+fp, out=np.zeros_like(tp,dtype=float), where=(tp+fp)!=0)
    rec  = np.divide(tp, tp+fn, out=np.zeros_like(tp,dtype=float), where=(tp+fn)!=0)
    f1   = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(tp,dtype=float), where=(prec+rec)!=0)
    denom= np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc  = np.divide(tp*tn-fp*fn, denom, out=np.zeros_like(tp,dtype=float), where=denom!=0)
    bal  = (rec + np.divide(tn, tn+fp, out=np.zeros_like(tp,dtype=float), where=(tn+fp)!=0))/2
    return acc,prec,rec,f1,mcc,bal

def bootstrap_and_perm(y, prob, thr,
                       n_boot=N_BOOT, n_perm=N_PERM, alpha=0.05):
    rng = np.random.default_rng(SEED)
    y, prob = np.asarray(y), np.asarray(prob)
    pred    = _to_pred(prob, thr)

    # observed
    tp,fp,fn,tn = _confusion_mtx(y[None,:], pred[None,:])
    obs         = [m.item() for m in _metrics_from_conf(tp,fp,fn,tn)]
    obs_auc     = roc_auc_score(y, prob)

    # bootstrap
    boot_idx = rng.integers(0,len(y), (n_boot,len(y)))
    tp,fp,fn,tn = _confusion_mtx(y[boot_idx], pred[boot_idx])
    boot = _metrics_from_conf(tp,fp,fn,tn)
    ci_lo = [np.percentile(b,100*alpha/2) for b in boot]
    ci_hi = [np.percentile(b,100*(1-alpha/2)) for b in boot]
    boot_auc = [roc_auc_score(y[i],prob[i]) for i in boot_idx]
    auc_lo, auc_hi = np.percentile(boot_auc,[100*alpha/2,100*(1-alpha/2)])

    # permutation p
    perm_idx = np.argsort(rng.random((n_perm,len(y))),1)
    py       = y[perm_idx]
    tp,fp,fn,tn = _confusion_mtx(py, pred[None,:])
    perm = _metrics_from_conf(tp,fp,fn,tn)
    pvals= [((np.sum(p>=o)+1)/(n_perm+1)) for p,o in zip(perm,obs)]
    perm_auc=[roc_auc_score(py[i],prob) if np.unique(py[i]).size==2 else -np.inf
              for i in range(n_perm)]
    p_auc=(np.sum(np.array(perm_auc)>=obs_auc)+1)/(n_perm+1)

    keys=['accuracy','precision','recall','f1','mcc','balanced_accuracy']
    out:Dict[str,Any] = {}
    for k,s,lo,hi,p in zip(keys,obs,ci_lo,ci_hi,pvals):
        out[k]=dict(score=float(s),ci_low=float(lo),ci_high=float(hi),p=float(p))
    out['roc_auc']=dict(score=float(obs_auc),ci_low=float(auc_lo),ci_high=float(auc_hi),p=float(p_auc))
    return out

def best_f1_threshold(y, prob):
    thrs = np.linspace(0.05,0.95,19)
    f1s  = [f1_score(y,_to_pred(prob,t),zero_division=0) for t in thrs]
    return float(thrs[int(np.argmax(f1s))])

def print_table(d,title):
    print(f"\n{title}:")
    print("{:20s} {:>8s} {:>18s} {:>10s}".format("metric","score","95% CI","p"))
    for k,v in d.items():
        print("{:20s} {:8.3f} [{:5.3f},{:5.3f}] {:10.4f}".format(
            k,v['score'],v['ci_low'],v['ci_high'],v['p']))

# ════════════════════ 2.  Data helpers ════════════════════════
def get_pid(p:Path)->str:
    stem=p.stem
    for pref in("train_","val_"):
        if stem.startswith(pref): stem=stem[len(pref):]
    return stem.rsplit("_",1)[0]

class FileDS(Dataset):
    def __init__(self,samples,tf): self.samples=samples; self.tf=tf; self.loader=datasets.folder.default_loader
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        path,lbl=self.samples[i]; img=self.loader(path); img=self.tf(img)
        return img,lbl

def make_loader(samples,tf,batch,balance,train):
    ds=FileDS(samples,tf)
    if balance and train:
        lbls=np.array([l for _,l in samples]); w=1./np.bincount(lbls)[lbls]
        sampler=WeightedRandomSampler(w,len(w),replacement=True); shuffle=False
    else:sampler=None; shuffle=train
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,sampler=sampler,
                      num_workers=2,pin_memory=DEVICE.type=="cuda",persistent_workers=True)

def build_tf(sz):
    tr=transforms.Compose([
        transforms.Resize(int(sz*1.1)),
        transforms.RandomResizedCrop(sz,scale=(0.9,1.0)),
        transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ev=transforms.Compose([
        transforms.Resize(int(sz*1.1)),transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return tr,ev

# ════════════════════ 3.  Model ═══════════════════════════════
def build_model(pretrained=True):
    w=ResNet18_Weights.DEFAULT if pretrained else None
    m=models.resnet18(weights=w)
    m.fc=nn.Linear(m.fc.in_features,2)
    nn.init.xavier_uniform_(m.fc.weight); nn.init.zeros_(m.fc.bias)
    return m

@torch.no_grad()
def infer(model,loader):
    model.eval(); ys,ps=[],[]
    for x,y in loader:
        x=x.to(DEVICE); logits=model(x)
        prob=torch.softmax(logits,1)[:,1]
        ys.append(y.numpy()); ps.append(prob.cpu().numpy())
    return np.concatenate(ys),np.concatenate(ps)

# ════════════════════ 4.  Train one fold ══════════════════════
def train_fold(fid,train_s,val_s,tf_tr,tf_ev,args):
    tr_loader=make_loader(train_s,tf_tr,args.batch_size,True,True)
    va_loader=make_loader(val_s,  tf_ev,args.batch_size,False,False)

    cw=1./torch.tensor(np.bincount([l for _,l in train_s]),dtype=torch.float32)
    loss_fn=nn.CrossEntropyLoss(weight=cw.to(DEVICE))
    model=build_model(pretrained=not args.no_pretrain).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best_auc,best_thr=0.0,0.5; no_imp=0
    for ep in range(1,args.max_epochs+1):
        model.train(); run=0.0
        pbar=tqdm(tr_loader,leave=False,desc=f"[fold {fid}] {ep:02d}")
        for x,y in pbar:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(); out=model(x); loss=loss_fn(out,y); loss.backward(); opt.step()
            run+=loss.item()*x.size(0); pbar.set_postfix(loss=loss.item())
        yv,pv=infer(model,va_loader)
        thr=best_f1_threshold(yv,pv)          # for statistics
        auc=roc_auc_score(yv,pv)
        if auc>best_auc:
            best_auc,best_thr=auc,thr; no_imp=0
            torch.save({'state_dict':model.state_dict(),'thr':best_thr},f"best_fold{fid}.pt")
        else:no_imp+=1
        if args.patience and no_imp>=args.patience:
            print(f"fold {fid}: early-stop @ epoch {ep}")
            break

    # final metrics on VAL for logging
    val_metrics=bootstrap_and_perm(yv,pv,best_thr)
    print(f"\nfold {fid} finished. VAL ROC-AUC={best_auc:.3f}")
    print_table(val_metrics,f"VAL fold {fid}")
    return best_auc,best_thr,val_metrics

# ──────────────────────────────────────────────────────────────
#  NEW  main()  –  uses OOF predictions for a global threshold
# ──────────────────────────────────────────────────────────────
def main(args):
    tf_train, tf_eval = build_tf(args.img_size)

    # 1) collect *all* internal samples (train + val directories)
    samples_int: list[tuple[str, int]] = []
    for split in ("train", "val"):
        ds = datasets.ImageFolder(Path(args.data_root) / split)
        samples_int += ds.samples

    paths = np.array([p for p, _ in samples_int])
    labels = np.array([l for _, l in samples_int])
    groups = np.array([get_pid(Path(p)) for p in paths])

    cv = StratifiedGroupKFold(
        n_splits=args.n_folds, shuffle=True, random_state=SEED
    )

    fold_aucs, oof_y, oof_p = [], [], []          # ← will hold out-of-fold preds

    for fold_id, (tr_idx, va_idx) in enumerate(
        cv.split(paths, labels, groups), 1
    ):
        tr_samples = [samples_int[i] for i in tr_idx]
        va_samples = [samples_int[i] for i in va_idx]

        # ----- train this fold ------------------------------------------------
        auc, _, _ = train_fold(
            fold_id, tr_samples, va_samples,
            tf_train, tf_eval, args
        )
        fold_aucs.append(auc)

        # ----- generate OOF predictions with the *saved* best model ----------
        # inside the for-fold loop (build the VAL DataLoader)
        va_loader = make_loader(
            va_samples,          # samples
            tf_eval,             # transform
            args.batch_size,     # batch
            False,               # balance
            False                # train  (was: train_split=False)
        )
        ckpt = torch.load(f"best_fold{fold_id}.pt", map_location=DEVICE)
        model = build_model(pretrained=False).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        y_val, p_val = infer(model, va_loader)
        oof_y.append(y_val)
        oof_p.append(p_val)

    # 2) concatenate all OOF predictions and pick ONE global threshold
    oof_y = np.concatenate(oof_y)
    oof_p = np.concatenate(oof_p)
    global_thr = best_f1_threshold(oof_y, oof_p)
    print(f"\nGlobal threshold from OOF optimisation: {global_thr:.2f}")

    # 3) evaluate the external set with an ensemble of k models
    ext_ds = datasets.ImageFolder(
        Path(args.data_root) / "external_val", transform=tf_eval
    )
    ext_loader = DataLoader(
        ext_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=DEVICE.type == "cuda",
    )

    ensemble_probs = []
    for fold_id in range(1, args.n_folds + 1):
        ckpt = torch.load(f"best_fold{fold_id}.pt", map_location=DEVICE)
        model = build_model(pretrained=False).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        _, p_ext = infer(model, ext_loader)
        ensemble_probs.append(p_ext)

    p_ext_mean = np.mean(ensemble_probs, axis=0)
    y_ext = np.array([lbl for _, lbl in ext_ds.samples])

    ext_metrics = bootstrap_and_perm(y_ext, p_ext_mean, global_thr)

    # 4) final report ---------------------------------------------------------
    print("\n────────── CV summary ──────────")
    for i, auc in enumerate(fold_aucs, 1):
        print(f"fold {i}: VAL ROC-AUC = {auc:.3f}")
    print(
        f"CV mean ± std ROC-AUC: {np.mean(fold_aucs):.3f} "
        f"± {np.std(fold_aucs):.3f}"
    )
    print_table(ext_metrics, "EXTERNAL (ensemble)")

# ════════════════════ 6.  CLI ════════════════════════════════
def cli():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data_root',default='.',type=str)
    ap.add_argument('--n_folds',default=5,type=int)
    ap.add_argument('--max_epochs',default=30,type=int)
    ap.add_argument('--patience',default=4,type=int)
    ap.add_argument('--batch_size',default=BATCH_SIZE,type=int)
    ap.add_argument('--img_size',default=IMG_SIZE,type=int)
    ap.add_argument('--lr',default=LR,type=float)
    ap.add_argument('--weight_decay',default=WEIGHT_DECAY,type=float)
    ap.add_argument('--no_pretrain',action='store_true')
    return ap.parse_args()

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=UserWarning)
    main(cli())