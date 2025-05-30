#!/usr/bin/env python3
# ------------------------------------------------------------
#  Hormone-receptor (HR) prediction from CEM crops – compact v2
#  PyTorch ≥2.0  |  Apple-silicon (MPS) friendly
#  -----------------------------------------------------------

from __future__ import annotations
import os, copy, time, random, argparse, warnings, itertools
from dataclasses import dataclass, asdict

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models

from sklearn.metrics import *
from sklearn.utils import resample

import matplotlib, matplotlib.pyplot as plt, seaborn as sns
matplotlib.use("Agg")                                    # head-less speed

# ---------- CONFIG ------------------------------------------------------------------
@dataclass
class CFG:                               # defaults are cli-over-writable
    data_dir          : str   = './'
    tag               : str   = 'resnet18_rep'
    epochs            : int   = 30
    bs                : int   = 4
    lr                : float = 1e-5
    wd                : float = 5e-4
    patience          : int   = 7
    img_sz            : tuple = (224,224)
    n_classes         : int   = 2
    best_metric       : str   = 'pr_auc'               # or balanced_accuracy
    pos_cls           : int   = 1                      # for PR-AUC
    seed              : int   = 86
    boot              : int   = 1_000                  # bootstrap N
    ci                : float = .95
    loss_weights      : tuple = (1., 2.5)              # HR+ , HR-
CFG = CFG()  # will be overwritten by CLI ------------------------------------------------

# ---------- SPEED TWEAKS -------------------------------------------------------------
torch.set_float32_matmul_precision('high')
if torch.backends.mps.is_available():  device = torch.device('mps')
elif torch.cuda.is_available():        device = torch.device('cuda')
else:                                  device = torch.device('cpu')
torch.manual_seed(CFG.seed); np.random.seed(CFG.seed); random.seed(CFG.seed)
if device.type == 'cuda': torch.cuda.manual_seed_all(CFG.seed)

# optional torch.compile (safe for CPU / CUDA; skipped on current MPS):
compile_ok = (device.type != 'mps') and hasattr(torch, 'compile')
def _compile(m): return torch.compile(m) if compile_ok else m

# ---------- TRANSFORMS --------------------------------------------------------------
norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
aug  = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomResizedCrop(CFG.img_sz),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(.1,.1),
        transforms.ToTensor(), norm ])
val_tf = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize(256),
        transforms.CenterCrop(CFG.img_sz), transforms.ToTensor(), norm ])
dtrans = {'train':aug, 'val':val_tf, 'external_val':val_tf}

# ---------- METRICS HELPERS ---------------------------------------------------------
def _pr_auc(y,p):
    if len(np.unique(y))<2: return .0
    pr,rc,_=precision_recall_curve(y,p,pos_label=CFG.pos_cls)
    return auc(rc,pr)

def _all_metrics(y,ŷ,p):
    prec,rec,f1,_=precision_recall_fscore_support(y,ŷ,labels=[0,1],zero_division=0)
    met=dict(accuracy=accuracy_score(y,ŷ),
             balanced_accuracy=balanced_accuracy_score(y,ŷ),
             mcc=matthews_corrcoef(y,ŷ) if len(np.unique(y))>1 else 0,
             specificity=rec[0],               # HR+ recall
             roc_auc=roc_auc_score(y,p) if len(np.unique(y))>1 else .5,
             pr_auc=_pr_auc(y,p),
             precision_hr_plus =prec[0], recall_hr_plus =rec[0], f1_hr_plus =f1[0],
             precision_hr_minus=prec[1], recall_hr_minus=rec[1], f1_hr_minus=f1[1])
    return met

def ci(values):                                # percentile CI
    l,u=np.percentile(values,[100*CFG.ci/2,100*(1-CFG.ci/2)])
    return (l,u)

# ---------- DATA --------------------------------------------------------------------
def get_dls():
    imgs={k:datasets.ImageFolder(os.path.join(CFG.data_dir,k),dtrans[k])
          for k in dtrans}
    trg=np.array(imgs['train'].targets); cnt=np.bincount(trg)
    wts=torch.DoubleTensor([1/cnt[t] for t in trg])
    samp=WeightedRandomSampler(wts,len(wts))
    dl=lambda ds,**kw:DataLoader(ds,batch_size=CFG.bs,**kw,
                                 worker_init_fn=lambda _:np.random.seed(CFG.seed))
    return { 'train':dl(imgs['train'],sampler=samp),
             'val'  :dl(imgs['val'],shuffle=False),
             'external_val':dl(imgs['external_val'],shuffle=False) }, imgs

# ---------- TRAIN -------------------------------------------------------------------
def train(model, dls, sizes):
    crit = nn.CrossEntropyLoss(weight=torch.tensor(CFG.loss_weights,
                                 device=device))
    opt  = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    sch  = ReduceLROnPlateau(opt,mode='max',factor=.1,patience=CFG.patience)
    best,best_w=0,copy.deepcopy(model.state_dict())
    hist={'tr_loss':[],'va_loss':[],'tr_met':[],'va_met':[]}
    for ep in range(1,CFG.epochs+1):
        for phase in ['train','val']:
            model.train(phase=='train')
            run_loss, y, ŷ, p = 0,[],[],[]
            for x,l in dls[phase]:
                x,l=x.to(device),l.to(device); opt.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    o=model(x); loss=crit(o,l)
                    if phase=='train': loss.backward(); opt.step()
                run_loss+=loss.item()*x.size(0)
                probs=torch.softmax(o,1)[:,CFG.pos_cls].detach().cpu().numpy()
                p.extend(probs); y.extend(l.cpu().numpy())
                ŷ.extend(o.argmax(1).cpu().numpy())
            ep_loss=run_loss/sizes[phase]
            met=_pr_auc(y,p) if CFG.best_metric=='pr_auc' else balanced_accuracy_score(y, ŷ)
            if phase=='train':   hist['tr_loss'].append(ep_loss); hist['tr_met'].append(met)
            else:
                hist['va_loss'].append(ep_loss); hist['va_met'].append(met)
                sch.step(met)
                if met>best: best, best_w=met,copy.deepcopy(model.state_dict())
        print(f'Epoch {ep}/{CFG.epochs}  |  val {CFG.best_metric}: {best:.4f}')
        if opt.param_groups[0]['lr'] < CFG.lr*1e-3: break
    model.load_state_dict(best_w); torch.save(best_w,f'best_model_hr_{CFG.tag}.pth')
    return model,hist

# ---------- EVAL --------------------------------------------------------------------
def eval_phase(model,dl,phase='val',th=.5):
    model.eval(); y, p=[], []
    with torch.no_grad():
        for x,l in dl: 
            o=model(x.to(device)); p.extend(torch.softmax(o,1)[:,CFG.pos_cls].cpu().numpy())
            y.extend(l.numpy())
    p=np.array(p); y=np.array(y)
    ŷ = (p>=th).astype(int) if CFG.pos_cls==1 else (p<=th).astype(int)
    pt=_all_metrics(y, ŷ, p)
    if len(y)<10: return {k:(v,(np.nan,np.nan)) for k,v in pt.items()}
    boots={k:[] for k in pt}
    for i in range(CFG.boot):
        idx=resample(range(len(y)), n_samples=len(y),
                     random_state=CFG.seed+i)
        yb,pb=y[idx],p[idx]
        ŷb=(pb>=th).astype(int) if CFG.pos_cls==1 else (pb<=th).astype(int)
        mb=_all_metrics(yb, ŷb, pb)
        for k in mb: boots[k].append(mb[k])
    return {k:(pt[k],ci(v)) for k,v in boots.items()}

def get_preds(model, dl):
    model.eval(); y, p = [], []
    with torch.no_grad():
        for x, l in dl:
            o = model(x.to(device))
            p.extend(torch.softmax(o, 1)[:, CFG.pos_cls].cpu().numpy())
            y.extend(l.numpy())
    return np.asarray(y), np.asarray(p)
# --------------------------------------------------------------------

# ---------- THRESHOLD TUNING --------------------------------------------------------
def optimal_th(y,p):
    uniq=np.unique(p);                
    if len(uniq)<2: return .5
    cand=np.clip(np.r_[.001,.5,.999,(uniq[:-1]+uniq[1:])/2],0,1)
    best,bth = -1,.5
    for t in cand:
        ŷ=(p>=t).astype(int) if CFG.pos_cls==1 else (p<=t).astype(int)
        _,_,f1,_=precision_recall_fscore_support(y, ŷ, labels=[0,1],zero_division=0)
        if f1[CFG.pos_cls]>best: best,bth = f1[CFG.pos_cls],t
    return bth

# ---------- PLOTS -------------------------------------------------------------------
def plot_hist(h):
    e=range(1,len(h['tr_loss'])+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(e,h['tr_loss'],label='train'); plt.plot(e,h['va_loss'],label='val')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2); plt.plot(e,h['tr_met'],label='train'); plt.plot(e,h['va_met'],label='val')
    plt.legend(); plt.title(CFG.best_metric)
    plt.tight_layout(); plt.savefig(f'train_hist_{CFG.tag}.png')

def cm_plot(y,ŷ,name):
    cm=confusion_matrix(y, ŷ,labels=[0,1])
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
        xticklabels=[f'Pred {c}' for c in ['HR+','HR-']],
        yticklabels=[f'True {c}' for c in ['HR+','HR-']])
    plt.title(name); plt.tight_layout()
    plt.savefig(f'cm_{name}_{CFG.tag}.png')

# ---------- MAIN --------------------------------------------------------------------
def main():
    global CFG
    p=argparse.ArgumentParser(); [p.add_argument(f'--{k}',type=type(v),default=v) for k,v in asdict(CFG).items()]
    CFG = CFG.__class__(**vars(p.parse_args()))
    print('CONFIG:',CFG)

    dls,imgs = get_dls(); sizes={k:len(v) for k,v in imgs.items()}
    model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, CFG.n_classes)
    model=_compile(model.to(device))

    model,hist = train(model,dls,sizes); plot_hist(hist)

    # ----- validation set
    val_res=eval_phase(model,dls['val'],'val');        print('\nVAL:',val_res['balanced_accuracy'])
    y=p=dls['val'].dataset.targets
    y_val, p_val = get_preds(model, dls['val'])   # both are 1-D np.arrays
    th = optimal_th(y_val, p_val)
    val_res_opt=eval_phase(model,dls['val'],'val_opt',th)
    # ----- external
    ext_res=eval_phase(model,dls['external_val'],'ext')
    ext_res_opt=eval_phase(model,dls['external_val'],'ext_opt',th)
    # ----- summaries
    for name,res in [('VAL 0.5',val_res),('VAL opt',val_res_opt),
                     ('EXT 0.5',ext_res),('EXT opt',ext_res_opt)]:
        print(f'\n{name}')
        for k,(v,(l,u)) in res.items():
            print(f' {k:20s}: {v:.4f}  ({l:.4f},{u:.4f})')

if __name__=='__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    main()