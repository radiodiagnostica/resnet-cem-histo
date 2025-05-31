#!/usr/bin/env python3
# ------------------------------------------------------------
#  Hormone-receptor (HR) prediction – compact v5.2  +  Grad-CAM
# ------------------------------------------------------------
from __future__ import annotations
import os, copy, random, argparse, warnings
from dataclasses import dataclass, asdict

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from PIL import Image

import matplotlib, matplotlib.pyplot as plt, seaborn as sns
matplotlib.use("Agg")

from sklearn.metrics import *
from sklearn.utils import resample

# ---------- CONFIG ------------------------------------------------------------------
@dataclass
class CFG:
    data_dir : str   = './';        tag          : str   = 'resnet18_rep'
    epochs   : int   = 30;          bs           : int   = 4
    lr       : float = 1e-5;        wd           : float = 5e-4
    patience : int   = 7;           img_sz       : tuple = (224,224)
    n_classes: int   = 2;           best_metric  : str   = 'pr_auc'
    pos_cls  : int   = 1;           seed         : int   = 86
    boot     : int   = 1_000;       ci           : float = .95
    loss_weights:tuple=(1.,2.5)
CFG = CFG()

# ---------- SPEED -------------------------------------------------------------------
torch.set_float32_matmul_precision('high')
device=torch.device('mps' if torch.backends.mps.is_available()
                    else ('cuda' if torch.cuda.is_available() else 'cpu'))

compile_ok=(device.type!='mps') and hasattr(torch,'compile')
def _compile(m): return torch.compile(m) if compile_ok else m

# ---------- SEED --------------------------------------------------------------------
def set_seed(seed:int):
    """(Re)seed every RNG we rely on."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)

# ---------- TRANSFORMS --------------------------------------------------------------
def build_transforms():
    """Build data-augmentation / validation transforms based on *current* CFG."""
    norm  = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    aug   = transforms.Compose([
                transforms.Grayscale(3),
                transforms.RandomResizedCrop(CFG.img_sz),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(.1,.1),
                transforms.ToTensor(), norm
            ])
    val_tf = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(256),
                transforms.CenterCrop(CFG.img_sz),
                transforms.ToTensor(), norm
             ])
    return norm, aug, val_tf, {'train': aug, 'val': val_tf, 'external_val': val_tf}

norm, aug, val_tf, dtrans = build_transforms()

# ---------- METRICS -----------------------------------------------------------------
def _pr_auc(y,p):
    if len(np.unique(y))<2: return .0
    pr,rc,_=precision_recall_curve(y,p,pos_label=CFG.pos_cls); return auc(rc,pr)

def _all_metrics(y,ŷ,p):
    prec,rec,f1,_=precision_recall_fscore_support(y,ŷ,labels=[0,1],zero_division=0)
    return dict(accuracy=accuracy_score(y,ŷ),
                balanced_accuracy=balanced_accuracy_score(y,ŷ),
                mcc=matthews_corrcoef(y,ŷ) if len(np.unique(y))>1 else 0,
                specificity=rec[0],
                roc_auc=roc_auc_score(y,p) if len(np.unique(y))>1 else .5,
                pr_auc=_pr_auc(y,p),
                precision_hr_plus =prec[0], recall_hr_plus =rec[0], f1_hr_plus =f1[0],
                precision_hr_minus=prec[1], recall_hr_minus=rec[1], f1_hr_minus=f1[1])

def ci(v): l,u=np.percentile(v,[100*(1-CFG.ci)/2,100*(1+CFG.ci)/2]); return (l,u)

# ---------- DATA --------------------------------------------------------------------
def get_dls():
    imgs={k:datasets.ImageFolder(os.path.join(CFG.data_dir,k),dtrans[k]) for k in dtrans}
    trg=np.array(imgs['train'].targets)
    wts=torch.DoubleTensor([1/np.bincount(trg)[t] for t in trg])
    samp=WeightedRandomSampler(wts,len(wts))
    mk=lambda ds,**kw:DataLoader(ds,batch_size=CFG.bs,**kw,
                       worker_init_fn=lambda _:np.random.seed(CFG.seed))
    return {'train':mk(imgs['train'],sampler=samp),
            'val':mk(imgs['val'],shuffle=False),
            'external_val':mk(imgs['external_val'],shuffle=False)}, imgs

# ---------- TRAIN -------------------------------------------------------------------
def train(model,dls,sizes):
    crit=nn.CrossEntropyLoss(weight=torch.tensor(CFG.loss_weights,device=device))
    opt=optim.Adam(model.parameters(),lr=CFG.lr,weight_decay=CFG.wd)
    sch=ReduceLROnPlateau(opt,mode='max',factor=.1,patience=CFG.patience)
    best,bw=0,copy.deepcopy(model.state_dict())
    hist={'tr_loss':[],'va_loss':[],'tr_met':[],'va_met':[]}
    for ep in range(1,CFG.epochs+1):
        for phase in ['train','val']:
            model.train(phase=='train'); run_loss,y,ŷ,p=0,[],[],[]
            for x,l in dls[phase]:
                x,l=x.to(device),l.to(device); opt.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    o=model(x); loss=crit(o,l); 
                    if phase=='train': loss.backward(); opt.step()
                run_loss+=loss.item()*x.size(0)
                probs=torch.softmax(o,1)[:,CFG.pos_cls].detach().cpu().numpy()
                p.extend(probs); y.extend(l.cpu().numpy()); ŷ.extend(o.argmax(1).cpu().numpy())
            ep_loss=run_loss/sizes[phase]
            met=_pr_auc(y,p) if CFG.best_metric=='pr_auc' else balanced_accuracy_score(y,ŷ)
            if phase=='train': hist['tr_loss'].append(ep_loss); hist['tr_met'].append(met)
            else: hist['va_loss'].append(ep_loss); hist['va_met'].append(met); sch.step(met)
            if phase=='val' and met>best: best,bw=met,copy.deepcopy(model.state_dict())
        
        print(f'E{ep:02d}/{CFG.epochs} | tr_loss {hist["tr_loss"][-1]:.4f} '
              f'va_loss {hist["va_loss"][-1]:.4f} | tr_{CFG.best_metric} '
              f'{hist["tr_met"][-1]:.4f} va_{CFG.best_metric} '
              f'{hist["va_met"][-1]:.4f} | best {best:.4f}')

        if opt.param_groups[0]['lr']<CFG.lr*1e-3: break
    model.load_state_dict(bw); torch.save(bw,f'best_model_{CFG.tag}.pth')
    
    pd.DataFrame({'epoch':range(1,len(hist['tr_loss'])+1),
                  'tr_loss':hist['tr_loss'],'va_loss':hist['va_loss'],
                  f'tr_{CFG.best_metric}':hist['tr_met'],
                  f'va_{CFG.best_metric}':hist['va_met']}
                ).to_csv(f'epoch_metrics_{CFG.tag}.csv',index=False)
    
    return model,hist

# ---------- EVAL --------------------------------------------------------------------
def eval_phase(model,dl,th=.5):
    model.eval(); y,p=[],[]
    with torch.no_grad():
        for x,l in dl:
            o=model(x.to(device))
            p.extend(torch.softmax(o,1)[:,CFG.pos_cls].cpu().numpy()); y.extend(l.numpy())
    y,p=np.asarray(y),np.asarray(p)
    ŷ=(p>=th).astype(int) if CFG.pos_cls==1 else (p<=th).astype(int)
    base=_all_metrics(y, ŷ, p)
    if len(y)<10: return {k:(v,(np.nan,np.nan)) for k,v in base.items()},y,ŷ
    boots={k:[] for k in base}
    for i in range(CFG.boot):
        idx=resample(range(len(y)),random_state=CFG.seed+i)
        yb,pb=y[idx],p[idx]
        ŷb=(pb>=th).astype(int) if CFG.pos_cls==1 else (pb<=th).astype(int)
        mb=_all_metrics(yb, ŷb, pb)
        for k in mb: boots[k].append(mb[k])
    return {k:(base[k],ci(v)) for k,v in boots.items()},y,ŷ

def get_preds(model,dl):
    model.eval(); y,p=[],[]
    with torch.no_grad():
        for x,l in dl:
            o=model(x.to(device))
            p.extend(torch.softmax(o,1)[:,CFG.pos_cls].cpu().numpy()); y.extend(l.numpy())
    return np.asarray(y),np.asarray(p)

# ---------- THRESHOLD ---------------------------------------------------------------
def optimal_th(y,p):
    cand=np.clip(np.r_[.001,.5,.999,(np.unique(p)[:-1]+np.unique(p)[1:])/2],0,1)
    best,bth=-1,.5
    for t in cand:
        ŷ=(p>=t).astype(int) if CFG.pos_cls==1 else (p<=t).astype(int)
        f1=precision_recall_fscore_support(y, ŷ,labels=[0,1],zero_division=0)[2][CFG.pos_cls]
        if f1>best: best,bth=f1,t
    return bth

# ---------- PLOTS -------------------------------------------------------------------
def plot_hist(h):
    e=range(1,len(h['tr_loss'])+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(e,h['tr_loss'],label='train'); plt.plot(e,h['va_loss'],label='val'); plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2); plt.plot(e,h['tr_met'],label='train'); plt.plot(e,h['va_met'],label='val'); plt.legend(); plt.title(CFG.best_metric)
    plt.tight_layout(); plt.savefig(f'train_hist_{CFG.tag}.png',dpi=300); plt.close()

def cm_plot(y,ŷ,name):
    cm=confusion_matrix(y, ŷ,labels=[0,1])
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                xticklabels=['Pred HR+','Pred HR-'],
                yticklabels=['True HR+','True HR-'])
    plt.title(name); plt.tight_layout()
    plt.savefig(f'cm_{name}_{CFG.tag}.png',dpi=300); plt.close()

# ---------- ACTIVATION HEATMAPS (Grad-CAM) ------------------------------------------
def save_heatmaps(model,dataset,n=50,save_dir='heatmaps'):
    os.makedirs(save_dir,exist_ok=True)
    blk=model.layer4[-1]

    acts,grads=[],[]
    def fw_hook(_, __, output):
        acts.append(output.detach())
        output.register_hook(lambda g: grads.append(g))
    h=blk.register_forward_hook(fw_hook)
    model.eval()

    for idx in random.sample(range(len(dataset)),min(n,len(dataset))):
        acts.clear(); grads.clear()
        pth,_ = dataset.samples[idx]
        img   = Image.open(pth).convert('RGB')
        x     = val_tf(img).unsqueeze(0).to(device)

        model.zero_grad()
        logits = model(x)
        logits[0, CFG.pos_cls].backward()

        A, G  = acts[0], grads[0]
        w     = G.mean(dim=(2,3), keepdim=True)
        cam   = (w*A).sum(1, keepdim=True).relu()
        cam   = F.interpolate(cam,(img.size[1],img.size[0]),
                              mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        cam   = (cam-cam.min())/(cam.max()+1e-9)

        plt.figure(figsize=(3,3)); plt.imshow(img)
        plt.imshow(cam,cmap='jet',alpha=.5); plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f'{save_dir}/{os.path.basename(pth)}',dpi=300); plt.close()
    h.remove()
    print(f'Activation heatmaps saved to \"{save_dir}/\"')

# ---------- MAIN --------------------------------------------------------------------
def main():
    global CFG, norm, aug, val_tf, dtrans
    p=argparse.ArgumentParser(); [p.add_argument(f'--{k}',type=type(v),default=v) for k,v in asdict(CFG).items()]
    CFG = CFG.__class__(**vars(p.parse_args())); print('CONFIG:',CFG)

    set_seed(CFG.seed)
    norm, aug, val_tf, dtrans = build_transforms()

    dls,imgs=get_dls(); sizes={k:len(v) for k,v in imgs.items()}
    model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc=nn.Linear(model.fc.in_features,CFG.n_classes); model=_compile(model.to(device))

    model,hist=train(model,dls,sizes); plot_hist(hist)
    yv,pv=get_preds(model,dls['val']); th=optimal_th(yv,pv)
    print(f'Optimal threshold = {th:.3f}')

    rows=[]; sets=[('TRN',dls['train']),('VAL',dls['val']),('EXT',dls['external_val'])]
    for lbl,dl in sets:
        for tname,t in [('0.5',.5),('opt',th)]:
            res,y,ŷ=eval_phase(model,dl,t); name=f'{lbl}_{tname}'
            rows.extend([dict(set=name,metric=m,value=v,ci_low=l,ci_high=u)
                         for m,(v,(l,u)) in res.items()])
            cm_plot(y, ŷ, name)
            print(f'\n{name}')
            for k,(v,(l,u)) in res.items(): print(f' {k:20s}: {v:.4f} ({l:.4f},{u:.4f})')

    pd.DataFrame(rows).to_csv(f'metrics_{CFG.tag}.csv',index=False)
    print(f'\nMetrics saved to metrics_{CFG.tag}.csv')
    print('All confusion matrices saved (one image per split & threshold).')

    save_heatmaps(model, imgs['val'])

if __name__=='__main__':
    warnings.filterwarnings("ignore",category=UserWarning)
    main()