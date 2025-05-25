#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#   Hormone-Receptor classifier  (v5-fix – patient-balanced α, correct import)
#   2024-xx-xx
# ---------------------------------------------------------------------------

import re, argparse, warnings, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models
import torchvision.transforms.v2 as T2
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                             roc_auc_score, roc_curve, confusion_matrix)
from tabulate import tabulate

# ------------------------- 0.  DEVICE ----------------------------------------
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
print("Running on", DEVICE)

# ------------------------- 1.  DATASET ---------------------------------------
PID_RX = re.compile(r"^(?:train_|val_)?(.+?)_\d+$", re.I)

def pid_from(fname: str) -> str:
    m = PID_RX.match(Path(fname).stem)
    if m is None:
        raise RuntimeError(f"Cannot parse patient id from {fname}")
    return m.group(1).lower()

class CropSet(Dataset):
    """root/1/*.jpg   root/2/*.jpg"""
    def __init__(self, root, tf):
        self.root = Path(root)
        self.tf = tf
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.cls2idx = {c:i for i,c in enumerate(self.classes)}
        self.samples = self._gather()

    def _gather(self):
        out = []
        for cls in self.classes:
            for p in (self.root/cls).glob("*.[jp][pn]g"):
                out.append((p, self.cls2idx[cls], pid_from(p.name)))
        return out

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path,lbl,pid = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.tf: img = self.tf(img)
        return img.float(), lbl, pid

# ------------------------- 2.  AUGMENTATION ----------------------------------
def build_tf(sz):
    mean,std = [0.5],[0.5]
    train = T2.Compose([
        T2.RandomResizedCrop(sz, scale=(0.8,1.0)),
        T2.RandomHorizontalFlip(),
        T2.RandomRotation(10),
        T2.RandomAffine(0, translate=(.05,.05), scale=(.9,1.1)),
        T2.RandomAdjustSharpness(.3),
        T2.GaussianBlur(3, sigma=(.1,2.0)),
        T2.ToTensor(),
        T2.Normalize(mean,std),
        T2.Lambda(lambda x: x.repeat(3,1,1))
    ])
    val = T2.Compose([
        T2.Resize(int(sz*1.05)),
        T2.CenterCrop(sz),
        T2.ToTensor(), T2.Normalize(mean,std),
        T2.Lambda(lambda x: x.repeat(3,1,1))
    ])
    return train, val

# ------------------------- 3.  SAMPLER ---------------------------------------
class OneCropPerPatient(Sampler):
    """Each epoch: one random crop per patient, shuffled."""
    def __init__(self, dataset):
        self.pid2idx = defaultdict(list)
        for idx,(*_,pid) in enumerate(dataset.samples):
            self.pid2idx[pid].append(idx)
        self.pids = list(self.pid2idx)

    def __iter__(self):
        idxs = [random.choice(self.pid2idx[pid]) for pid in self.pids]
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self): return len(self.pids)

# ------------------------- 4.  MODEL -----------------------------------------
def make_model(nc):
    net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for p in net.parameters(): p.requires_grad = False
    in_f = net.classifier.in_features
    net.classifier = nn.Sequential(
        nn.Dropout(.4),
        nn.Linear(in_f,256),
        nn.ReLU(inplace=True),
        nn.Dropout(.3),
        nn.Linear(256,nc)
    )
    return net

# ------------------------- 5.  LOSS ------------------------------------------
class Focal(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.a = torch.tensor(alpha)
        self.g = gamma
    def forward(self, logit, tgt):
        a = self.a.to(logit.device)
        logp = torch.log_softmax(logit,1)
        p    = torch.exp(logp)
        foc  = (1-p)**self.g
        oneh = torch.zeros_like(logit).scatter_(1,tgt.unsqueeze(1),1)
        loss = -(a.unsqueeze(0)*foc*oneh*logp).sum(1)
        return loss.mean()

# ------------------------- 6.  METRICS ---------------------------------------
def best_thr(y,s):
    fpr,tpr,thr = roc_curve(y,s)
    bal = (tpr + (1-fpr))/2
    return thr[np.argmax(bal)]

def pat_stats(y,prob,pid,thr):
    bag = defaultdict(list)
    for yt,p,pp in zip(y,prob,pid): bag[pp].append((yt,p))
    y_true,y_pred,y_score = [],[],[]
    for lst in bag.values():
        ys,ps = zip(*lst)
        y_true.append(ys[0])
        sc = np.mean(ps)
        y_score.append(sc)
        y_pred.append(int(sc>=thr))
    return (balanced_accuracy_score(y_true,y_pred),
            matthews_corrcoef(y_true,y_pred),
            roc_auc_score(y_true,y_score))

def show(d): print(tabulate([(k,f"{v:.4f}") for k,v in d.items()],
                            headers=["metric","value"],tablefmt="pipe"))

# ------------------------- 7.  MAIN ------------------------------------------
def run(a):
    tf_tr,tf_v = build_tf(a.img_size)
    r = Path(a.data_dir)
    ds = {'train': CropSet(r/'train',tf_tr),
          'val'  : CropSet(r/'val',tf_v)}
    if (r/'external_val').exists():
        ds['external_val'] = CropSet(r/'external_val',tf_v)

    dl = {
        'train': DataLoader(ds['train'], batch_size=a.bs,
                            sampler=OneCropPerPatient(ds['train']),
                            num_workers=0, pin_memory=True),
        'val'  : DataLoader(ds['val'], batch_size=a.bs,
                            shuffle=False, num_workers=0, pin_memory=True)
    }
    if 'external_val' in ds:
        dl['external_val'] = DataLoader(ds['external_val'], batch_size=a.bs,
                                        shuffle=False,num_workers=0,pin_memory=True)

    # ------- patient counts for α -------------------------------------
    pat_per_cls = defaultdict(set)
    for _,lbl,pid in ds['train'].samples:
        pat_per_cls[lbl].add(pid)
    n_pat = [len(pat_per_cls[i]) for i in range(len(ds['train'].classes))]   # e.g. [88,17]

    beta = 0.999
    raw  = [(1-beta)/(1-beta**n) for n in n_pat]   # effective-number formula
    scale = len(raw)/sum(raw)                      # rescale so mean=1
    alpha = [w*scale for w in raw]
    print("α for Focal-Loss (patient-balanced):", alpha)

    crit = Focal(alpha)

    # ---------------- model / optim / sched ---------------------------
    net = make_model(len(ds['train'].classes)).to(DEVICE)
    head,blk3 = [],[]
    for n,p in net.named_parameters():
        if n.startswith(("features.denseblock3","features.transition3")):
            p.requires_grad=False; blk3.append(p)
        elif p.requires_grad: head.append(p)
    opt = torch.optim.AdamW([
        {'params':head, 'lr':a.lr},
        {'params':blk3,'lr':a.lr*0.33}
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr*3,a.lr], epochs=a.epochs,
        steps_per_epoch=len(dl['train'])
    )

    best_bacc,best_state,wait = 0,None,0

    # ---------------- training loop -----------------------------------
    for ep in range(1,a.epochs+1):
        if ep==4:
            print(" -> denseblock3 unfrozen!")
            for p in blk3: p.requires_grad=True

        lr_now = sched.get_last_lr()[0]
        print(f"\nEpoch {ep}/{a.epochs} -- lr={lr_now:.3e}")

        # ----- TRAIN --------------------------------------------------
        net.train()
        seen,correct,tloss = 0,0,0
        for x,y,_ in dl['train']:
            x,y = x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            logit = net(x)
            loss  = crit(logit,y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.0)
            opt.step(); sched.step()

            bs = x.size(0)
            seen += bs
            correct += (logit.argmax(1)==y).sum().item()
            tloss += loss.item()*bs
        print(f"train loss={tloss/seen:.4f} acc={correct/seen:.4f}")

        # ----- VAL ----------------------------------------------------
        net.eval()
        v_lab,v_prob,v_pid = [],[],[]
        vseen,vcorr,vloss = 0,0,0
        with torch.no_grad():
            for x,y,p in dl['val']:
                x,y = x.to(DEVICE),y.to(DEVICE)
                logit = net(x)
                loss  = crit(logit,y)

                bs=x.size(0)
                vseen+=bs
                vcorr+=(logit.argmax(1)==y).sum().item()
                vloss+=loss.item()*bs
                v_lab.extend(y.cpu().numpy())
                v_prob.extend(torch.softmax(logit,1)[:,1].cpu().numpy())
                v_pid.extend(p)
        print(f"val   loss={vloss/vseen:.4f} acc_crop={vcorr/vseen:.4f}")

        thr = best_thr(v_lab,v_prob)
        bacc,mcc,auc = pat_stats(v_lab,v_prob,v_pid,thr)
        print(f"val PATIENT  bAcc={bacc:.3f} MCC={mcc:.3f} AUC={auc:.3f} thr={thr:.3f}")

        if bacc>best_bacc:
            best_bacc,wait=bacc,0
            best_state={'model':net.state_dict(),'thr':thr,'classes':ds['train'].classes}
        else:
            wait+=1
            if wait>=a.patience:
                print("Early stopping."); break

    torch.save(best_state,a.out)

    # ------------- external ------------------------------------------
    if 'external_val' in dl:
        print("\n--- External validation ---")
        net.load_state_dict(best_state['model']); net.eval()
        e_lab,e_prob,e_pid=[],[],[]
        with torch.no_grad():
            for x,y,p in dl['external_val']:
                x=x.to(DEVICE)
                logit=net(x)
                e_lab.extend(y.numpy())
                e_prob.extend(torch.softmax(logit,1)[:,1].cpu().numpy())
                e_pid.extend(p)
        eb,emcc,eau = pat_stats(e_lab,e_prob,e_pid,best_state['thr'])
        show(dict(balanced_acc=eb,mcc=emcc,auc=eau))
        # ConfMat on patient level
        bag=defaultdict(list)
        for yt,pr,p in zip(e_lab,e_prob,e_pid): bag[p].append((yt,pr))
        yt,yp=[],[]
        for lst in bag.values():
            yt.append(lst[0][0])
            yp.append(int(np.mean([pr for _,pr in lst])>=best_state['thr']))
        print("Confusion matrix (patient level):\n",confusion_matrix(yt,yp))

# ------------------------- 8.  CLI -------------------------------------------
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="dataset root with train/ val/ external_val/")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="best_mammo.pth")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    run(args)