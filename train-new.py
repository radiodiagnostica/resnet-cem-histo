#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────────────
#  Hormone-Receptor mammography classifier
#  v8 – class-balanced sampler • warm-start head • γ=4 focal loss
#       full metric block (value + 95 % CI + permutation p-val)
#  2024-xx-xx
# ────────────────────────────────────────────────────────────────────────────

import re, argparse, warnings, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import numpy.random as npr
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models
import torchvision.transforms.v2 as T2
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, roc_curve,
                             confusion_matrix)
from tabulate import tabulate


# ════════════════════════════════════════════════════════════════════════════
#  0.  DEVICE
# ════════════════════════════════════════════════════════════════════════════
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("Running on", DEVICE)


# ════════════════════════════════════════════════════════════════════════════
#  1.  DATASET
# ════════════════════════════════════════════════════════════════════════════
PID_RX = re.compile(r"^(?:train_|val_)?(.+?)_\d+$", re.I)

def pid_from(fname: str) -> str:
    m = PID_RX.match(Path(fname).stem)
    if m is None:
        raise RuntimeError(f"Cannot parse patient id from {fname}")
    return m.group(1).lower()

class CropSet(Dataset):
    """root/1/*.jpg …  root/2/*.jpg"""
    def __init__(self, root, tf):
        self.root  = Path(root)
        self.tf    = tf
        self.classes  = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.cls2idx  = {c: i for i, c in enumerate(self.classes)}
        self.samples  = self._gather()

    def _gather(self):
        out = []
        for cls in self.classes:
            for p in (self.root/cls).glob("*.[jp][pn]g"):
                out.append((p, self.cls2idx[cls], pid_from(p.name)))
        return out

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, lbl, pid = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.tf: img = self.tf(img)
        return img.float(), lbl, pid


# ════════════════════════════════════════════════════════════════════════════
#  2.  AUGMENTATION
# ════════════════════════════════════════════════════════════════════════════
def build_tf(sz):
    mean, std = [0.5], [0.5]
    train = T2.Compose([
        T2.RandomResizedCrop(sz, scale=(0.8, 1.0)),
        T2.RandomHorizontalFlip(),
        T2.RandomRotation(10),
        T2.RandomAffine(0, translate=(.05, .05), scale=(.9, 1.1)),
        T2.RandomAdjustSharpness(.3),
        T2.GaussianBlur(3, sigma=(.1, 2.0)),
        T2.ToTensor(), T2.Normalize(mean, std),
        T2.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    val = T2.Compose([
        T2.Resize(int(sz*1.05)), T2.CenterCrop(sz),
        T2.ToTensor(), T2.Normalize(mean, std),
        T2.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    return train, val


# ════════════════════════════════════════════════════════════════════════════
#  3.  PATIENT-BALANCED SAMPLER
# ════════════════════════════════════════════════════════════════════════════
class OneCropPerPatientBalanced(Sampler):
    """
    Each epoch:
        • 1 random crop for every majority-class patient
        • k random crops for every minority-class patient
      where k ≈ ceil(#maj / #min) so that the stream is class-balanced.
    """
    def __init__(self, dataset):
        pid2idx, pid2lbl = defaultdict(list), {}
        for idx, (_, lbl, pid) in enumerate(dataset.samples):
            pid2idx[pid].append(idx)
            pid2lbl[pid] = lbl

        cls0 = [p for p, l in pid2lbl.items() if l == 0]
        cls1 = [p for p, l in pid2lbl.items() if l == 1]

        self.maj, self.min_ = (cls0, cls1) if len(cls0) >= len(cls1) else (cls1, cls0)
        self.pid2idx = pid2idx
        self.k = int(np.ceil(len(self.maj) / len(self.min_)))

    def __iter__(self):
        idxs = []
        for pid in self.maj:
            idxs.append(random.choice(self.pid2idx[pid]))
        for pid in self.min_:
            for _ in range(self.k):
                idxs.append(random.choice(self.pid2idx[pid]))
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.maj) + len(self.min_) * self.k


# ════════════════════════════════════════════════════════════════════════════
#  4.  MODEL
# ════════════════════════════════════════════════════════════════════════════
def make_model(nc):
    net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for p in net.parameters(): p.requires_grad = False
    in_f = net.classifier.in_features
    net.classifier = nn.Sequential(
        nn.Dropout(.4),
        nn.Linear(in_f, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(.3),
        nn.Linear(256, nc)
    )
    return net


# ════════════════════════════════════════════════════════════════════════════
#  5.  LOSS
# ════════════════════════════════════════════════════════════════════════════
class Focal(nn.Module):
    def __init__(self, alpha, gamma=4.0):          #  ← γ increased
        super().__init__()
        self.a = torch.tensor(alpha)
        self.g = gamma
    def forward(self, logit, tgt):
        a    = self.a.to(logit.device)
        logp = torch.log_softmax(logit, 1)
        p    = torch.exp(logp)
        foc  = (1 - p) ** self.g
        oneh = torch.zeros_like(logit).scatter_(1, tgt.unsqueeze(1), 1)
        loss = -(a.unsqueeze(0) * foc * oneh * logp).sum(1)
        return loss.mean()


# ════════════════════════════════════════════════════════════════════════════
#  6.  METRICS  (bootstrap CI + permutation p-val)
# ════════════════════════════════════════════════════════════════════════════
def best_thr(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    bal = (tpr + (1 - fpr)) / 2
    return thr[np.argmax(bal)]

def bootstrap_ci(stat_fn, y_true, y_pred=None, y_score=None,
                 n_boot=2_000, alpha=.95, seed=0):
    rng, idx = npr.RandomState(seed), np.arange(len(y_true))
    stats = []
    for _ in range(n_boot):
        s  = rng.choice(idx, size=len(idx), replace=True)
        yt = [y_true[i] for i in s]
        if y_pred is not None:
            yp = [y_pred[i] for i in s]
            stats.append(stat_fn(yt, yp))
        else:
            ys = [y_score[i] for i in s]
            stats.append(stat_fn(yt, ys))
    lo = np.percentile(stats, (1-alpha)/2*100)
    hi = np.percentile(stats, (1+alpha)/2*100)
    return lo, hi

def permutation_p(stat_fn, y_true, y_pred=None, y_score=None,
                  n_perm=2_000, alternative='greater', seed=0):
    rng  = npr.RandomState(seed)
    obs  = stat_fn(y_true, y_pred) if y_pred is not None else stat_fn(y_true, y_score)
    hits, total = 0, 0
    for _ in range(n_perm):
        yt_perm = rng.permutation(y_true)
        try:
            perm = stat_fn(yt_perm, y_pred) if y_pred is not None else stat_fn(yt_perm, y_score)
        except ValueError:
            continue
        total += 1
        if alternative == 'greater':
            if perm >= obs: hits += 1
        else:
            if abs(perm) >= abs(obs): hits += 1
    return (hits + 1) / (total + 1)

def all_metrics(y_true, y_pred, y_score, alpha=.95):
    f_pr = lambda yt, yp: precision_score(yt, yp, zero_division=0)
    f_re = lambda yt, yp: recall_score   (yt, yp, zero_division=0)
    f_f1 = lambda yt, yp: f1_score       (yt, yp, zero_division=0)

    metrics = {
        'accuracy'      : accuracy_score,
        'balanced_acc'  : balanced_accuracy_score,
        'precision'     : f_pr,
        'recall'        : f_re,
        'f1'            : f_f1,
        'mcc'           : matthews_corrcoef,
        'auc'           : lambda yt, ys: roc_auc_score(yt, ys)
    }
    out = {}
    for nm, fn in metrics.items():
        if nm == 'auc':
            est = fn(y_true, y_score)
            lo, hi = bootstrap_ci(fn, y_true, y_score=y_score, alpha=alpha)
            pval   = permutation_p(fn, y_true, y_score=y_score)
        else:
            est = fn(y_true, y_pred)
            lo, hi = bootstrap_ci(fn, y_true, y_pred=y_pred, alpha=alpha)
            pval   = permutation_p(fn, y_true, y_pred=y_pred)
        out[nm] = (est, lo, hi, pval)
    return out

def show_metrics(block):
    rows = [(k,
             f"{v[0]:.4f}",
             f"[{v[1]:.4f} … {v[2]:.4f}]",
             f"{v[3]:.4f}")
            for k, v in block.items()]
    print(tabulate(rows,
                   headers=["metric", "value", "95 % CI", "p-val"],
                   tablefmt="pipe"))

def pat_aggregate(y, prob, pid, thr):
    bag = defaultdict(list)
    for yt, p, pp in zip(y, prob, pid): bag[pp].append((yt, p))
    y_true, y_pred, y_score = [], [], []
    for lst in bag.values():
        ys, ps = zip(*lst)
        y_true.append(ys[0])
        sc = np.mean(ps)
        y_score.append(sc)
        y_pred.append(int(sc >= thr))
    return y_true, y_pred, y_score


# ════════════════════════════════════════════════════════════════════════════
#  7.  MAIN
# ════════════════════════════════════════════════════════════════════════════
def run(a):
    tf_tr, tf_v = build_tf(a.img_size)
    root = Path(a.data_dir)
    ds = {'train': CropSet(root/'train', tf_tr),
          'val'  : CropSet(root/'val',   tf_v)}
    if (root/'external_val').exists():
        ds['external_val'] = CropSet(root/'external_val', tf_v)

    dl = {
        'train': DataLoader(ds['train'], batch_size=a.bs,
                            sampler=OneCropPerPatientBalanced(ds['train']),
                            num_workers=0, pin_memory=True),
        'val': DataLoader(ds['val'], batch_size=a.bs,
                          shuffle=False, num_workers=0, pin_memory=True)
    }
    if 'external_val' in ds:
        dl['external_val'] = DataLoader(ds['external_val'], batch_size=a.bs,
                                        shuffle=False, num_workers=0, pin_memory=True)

    # α (effective number, patient-level)
    pat_per_cls = defaultdict(set)
    for _, lbl, pid in ds['train'].samples:
        pat_per_cls[lbl].add(pid)
    n_pat  = [len(pat_per_cls[i]) for i in range(len(ds['train'].classes))]
    beta   = 0.999
    raw    = [(1-beta)/(1-beta**n) for n in n_pat]
    alpha  = [w * len(raw)/sum(raw) for w in raw]
    print("α for Focal-Loss (patient-balanced):", alpha)

    crit = Focal(alpha)

    # model
    net = make_model(len(ds['train'].classes)).to(DEVICE)

    # 1) warm-start: only last Linear (classifier.4)
    for n, p in net.named_parameters():
        p.requires_grad = n.endswith('classifier.4.weight') or n.endswith('classifier.4.bias')
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                            lr=a.lr_head, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr_head*3, epochs=a.warm_epochs,
        steps_per_epoch=len(dl['train'])
    )

    best_bacc, best_state = 0, None

    # ─────────────────────────── TRAINING LOOP ──────────────────────────────
    total_epochs = a.warm_epochs + a.fine_epochs
    for ep in range(1, total_epochs + 1):
        # unfreeze after warm-up
        if ep == a.warm_epochs + 1:
            print(" → unfreezing classifier (full head)")
            for n, p in net.named_parameters():
                if n.startswith("classifier"): p.requires_grad = True
            # add smaller-LR conv block
            for n, p in net.named_parameters():
                if n.startswith(("features.denseblock3", "features.transition3")):
                    p.requires_grad = True
            opt = torch.optim.AdamW([
                {'params': [p for n,p in net.named_parameters() if n.startswith("classifier")],
                 'lr': a.lr},
                {'params': [p for n,p in net.named_parameters()
                            if n.startswith(("features.denseblock3","features.transition3"))],
                 'lr': a.lr * 0.33}
            ], weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=[a.lr*3, a.lr], epochs=a.fine_epochs,
                steps_per_epoch=len(dl['train'])
            )

        lr_now = sched.get_last_lr()[0]
        print(f"\nEpoch {ep}/{total_epochs} — lr={lr_now:.3e}")

        # TRAIN
        net.train()
        seen, correct, tloss = 0, 0, 0
        for x, y, _ in dl['train']:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logit = net(x)
            loss  = crit(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()

            bs = x.size(0)
            seen += bs
            correct += (logit.argmax(1) == y).sum().item()
            tloss   += loss.item()*bs
        print(f"train loss={tloss/seen:.4f}   acc={correct/seen:.4f}")

        # VALIDATION
        net.eval()
        v_lab, v_prob, v_pid = [], [], []
        vseen, vcorr, vloss  = 0, 0, 0
        with torch.no_grad():
            for x, y, p in dl['val']:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logit = net(x)
                loss  = crit(logit, y)

                bs     = x.size(0)
                vseen += bs
                vcorr += (logit.argmax(1) == y).sum().item()
                vloss += loss.item()*bs

                v_lab.extend(y.cpu().numpy())
                v_prob.extend(torch.softmax(logit, 1)[:,1].cpu().numpy())
                v_pid.extend(p)
        print(f"val   loss={vloss/vseen:.4f}   acc_crop={vcorr/vseen:.4f}")

        # histogram to monitor collapse
        with torch.no_grad():
            preds = []
            for x,_,_ in dl['val']:
                preds.extend(net(x.to(DEVICE)).argmax(1).cpu().numpy())
        print("VAL crop prediction histogram:", np.bincount(preds, minlength=2))

        # PATIENT-level metrics
        thr = best_thr(v_lab, v_prob)
        yt, yp, ys = pat_aggregate(v_lab, v_prob, v_pid, thr)
        met = all_metrics(yt, yp, ys)
        print("\nPatient-level validation metrics:")
        show_metrics(met)

        # early stopping on bAcc
        bacc = met['balanced_acc'][0]
        improved = bacc > best_bacc + 1e-4   # tiny margin
        if improved:
            best_bacc = bacc
            best_state = {'model': net.state_dict(), 'thr': thr,
                          'classes': ds['train'].classes}
            wait = 0
        else:
            wait += 1
            if wait >= a.patience:
                print("Early stopping."); break

    torch.save(best_state, a.out)
    print(f"\nBest model saved to '{a.out}'  (bAcc={best_bacc:.3f})")

    # ───────────────────────── external validation ──────────────────────────
    if 'external_val' in dl:
        print("\n— External validation —")
        net.load_state_dict(best_state['model']); net.eval()
        e_lab, e_prob, e_pid = [], [], []
        with torch.no_grad():
            for x, y, p in dl['external_val']:
                x = x.to(DEVICE)
                logit = net(x)
                e_lab.extend(y.numpy())
                e_prob.extend(torch.softmax(logit,1)[:,1].cpu().numpy())
                e_pid.extend(p)
        yt, yp, ys = pat_aggregate(e_lab, e_prob, e_pid, best_state['thr'])
        met_ext = all_metrics(yt, yp, ys)
        print("\nExternal-PATIENT metrics:")
        show_metrics(met_ext)

        print("\nConfusion matrix (patient level):")
        print(confusion_matrix(yt, yp))


# ════════════════════════════════════════════════════════════════════════════
#  8.  CLI
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="dataset root with train/ val/ external_val/")
    ap.add_argument("--img_size",   type=int,   default=256)
    ap.add_argument("--bs",         type=int,   default=32)
    ap.add_argument("--warm_epochs",type=int,   default=3,
                    help="epochs with only last linear layer trainable")
    ap.add_argument("--fine_epochs",type=int,   default=40,
                    help="epochs after unfreezing head/backbone block")
    ap.add_argument("--patience",   type=int,   default=10)
    ap.add_argument("--lr_head",    type=float, default=1e-3,
                    help="LR for warm-start linear layer")
    ap.add_argument("--lr",         type=float, default=3e-4,
                    help="LR for fine-tuning head / block3")
    ap.add_argument("--out",        type=str,   default="best_mammo.pth")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    run(args)