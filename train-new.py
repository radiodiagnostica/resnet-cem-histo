#!/usr/bin/env python3
"""

Hormone-receptor (+ / –) classification on cropped mammograms
with proper imbalance handling, bootstrap CIs, permutation p-values
and data-driven threshold tuning.

Directory layout (unchanged):

data_root/
 ├── train/         (sub-folders 1, 2)
 ├── val/
 └── external_val/

"""
import argparse, os, random, time, copy, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, balanced_accuracy_score,
                             roc_auc_score)

# ───────────────────────────── Globals / Hyper-params ──────────────────────────
IMG_SIZE      = 384
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 30
BATCH_SIZE    = 16
N_BOOT        = 2000
N_PERM        = 2000
SEED          = 42
# ────────────────────────────────────────────────────────────────────────────────

# ------------------------- Reproducibility helpers -----------------------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
set_seed()

# ------------------------ Metric & statistics utilities ------------------------
def _to_pred(y_prob, thr):  # np.ndarray
    return (y_prob >= thr).astype(int)

def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = _to_pred(y_prob, thr)
    res = dict(
        accuracy           = accuracy_score         (y_true, y_pred),
        precision          = precision_score        (y_true, y_pred, zero_division=0),
        recall             = recall_score           (y_true, y_pred, zero_division=0),
        f1                 = f1_score               (y_true, y_pred, zero_division=0),
        mcc                = matthews_corrcoef      (y_true, y_pred),
        balanced_accuracy  = balanced_accuracy_score(y_true, y_pred)
    )
    try:
        res['roc_auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        res['roc_auc'] = np.nan
    return res

def bootstrap_ci(metric_func, y_true, y_prob, thr, n_boot=N_BOOT, alpha=0.05):
    rng  = np.random.default_rng(SEED)
    idx  = np.arange(len(y_true))
    vals = []
    while len(vals) < n_boot:
        s = rng.choice(idx, size=len(idx), replace=True)
        try:
            vals.append(metric_func(y_true[s], y_prob[s], thr))
        except ValueError:
            pass                 # skip resamples with single class for roc_auc
    lo, hi = np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

def permutation_p(metric_func, y_true, y_prob, thr, n_perm=N_PERM):
    rng  = np.random.default_rng(SEED)
    obs  = metric_func(y_true, y_prob, thr)
    cnt  = 0
    for _ in range(n_perm):
        perm = rng.permutation(y_true)
        try:
            stat = metric_func(perm, y_prob, thr)
            if stat >= obs: cnt += 1
        except ValueError:
            pass
    return (cnt + 1)/(n_perm + 1)

def full_metrics(y_true, y_prob, thr):
    out = {}
    metric_list = [
        ('accuracy',          lambda y,p,t: accuracy_score(y, _to_pred(p,t))),
        ('precision',         lambda y,p,t: precision_score(y, _to_pred(p,t), zero_division=0)),
        ('recall',            lambda y,p,t: recall_score   (y, _to_pred(p,t), zero_division=0)),
        ('f1',                lambda y,p,t: f1_score       (y, _to_pred(p,t), zero_division=0)),
        ('mcc',               lambda y,p,t: matthews_corrcoef(y, _to_pred(p,t))),
        ('balanced_accuracy', lambda y,p,t: balanced_accuracy_score(y, _to_pred(p,t))),
        ('roc_auc',           lambda y,p,t: roc_auc_score(y, p))
    ]
    for name, fn in metric_list:
        try:
            score = fn(y_true, y_prob, thr)
        except ValueError:
            score = np.nan
        ci_lo, ci_hi = bootstrap_ci(fn, y_true, y_prob, thr)
        p_val        = permutation_p(fn, y_true, y_prob, thr)
        out[name] = {'score':score, 'ci_low':ci_lo, 'ci_high':ci_hi, 'p':p_val}
    return out

def print_table(metrics_dict, title):
    print(f"\n{title} metrics:")
    print("{:20s} {:>8s} {:>18s} {:>12s}".format("metric","score","95% CI","p-value"))
    for k,v in metrics_dict.items():
        print("{:20s} {:8.3f} [{:5.3f},{:5.3f}] {:12.4f}".format(
            k, v['score'], v['ci_low'], v['ci_high'], v['p']))

# ------------------------ Threshold optimisation helper ------------------------
def best_f1_threshold(y_true, y_prob):
    thrs = np.linspace(0.05, 0.95, 19)
    f1s  = [f1_score(y_true, _to_pred(y_prob,t), zero_division=0) for t in thrs]
    return thrs[int(np.argmax(f1s))]

# ------------------------ Data & augmentation ---------------------------------
def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.9,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def make_loader(root,                # dataset root
                split,               # 'train' | 'val' | 'external_val'
                tf,                  # torchvision transforms
                batch_size,
                balance   = False,
                workers   = 2,       # ↓ reduced from 4
                prefetch  = 1):      # ↓ reduced from default 2
    """
    Build a DataLoader with sensible defaults for macOS
    (low file-descriptor limit).  Uses a weighted sampler when
    balance=True and split=='train'.
    Can be combined with setting 'ulimit -Sn 4096' on system shell.
    """
    ds       = datasets.ImageFolder(Path(root) / split, transform=tf)
    pin_mem  = torch.cuda.is_available()          # pin only if CUDA GPU present

    if balance and split == 'train':
        # ----- class-balanced sampling -----
        targets      = [s[1] for s in ds.samples]
        class_counts = np.bincount(targets)
        weights      = 1.0 / class_counts[targets]
        sampler      = WeightedRandomSampler(weights,
                                             num_samples=len(weights),
                                             replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = (split == 'train')

    loader = DataLoader(
        ds,
        batch_size        = batch_size,
        shuffle           = shuffle,
        sampler           = sampler,
        num_workers       = workers,
        prefetch_factor   = prefetch,
        persistent_workers= workers > 0,    # keep worker FDs open between epochs
        pin_memory        = pin_mem
    )
    return loader, ds

# ----------------------------- Model & loss -----------------------------------
def build_model(num_classes=2, pretrained=True):
    m = models.resnet18(pretrained=pretrained)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ---------------------------- Train / Eval loops ------------------------------
def train_one_epoch(model, loader, criterion, optim, device):
    model.train(); running = 0.0
    for x,y in tqdm(loader, leave=False):
        x,y = x.to(device), y.to(device)
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        running += loss.item()*x.size(0)
    return running/len(loader.dataset)

@torch.no_grad()
def inference(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for x,y in loader:
        x = x.to(device)
        logits = model(x)
        prob   = torch.softmax(logits,1)[:,1]    # P(class==1)
        y_true.append(y.cpu().numpy())
        y_prob.append(prob.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)

# ----------------------------------- Main -------------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    tf_train, tf_eval = build_transforms(args.img_size)

    train_loader, train_ds = make_loader(args.data_root, 'train', tf_train,
                                         args.batch_size, balance=True)
    val_loader,   _        = make_loader(args.data_root, 'val',   tf_eval, args.batch_size)
    ext_loader,   _        = make_loader(args.data_root, 'external_val', tf_eval, args.batch_size)

    # ---------- class-weighted CrossEntropy ----------
    train_targets = np.array([y for _,y in train_ds.samples])
    class_counts  = np.bincount(train_targets)     # e.g. [213, 36] (folder 0/1 default)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = build_model(pretrained=not args.no_pretrain).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1, best_thr = 0.0, 0.5
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optim, device)

        # ---------------- RAW probabilities on val -----------------
        y_val, p_val = inference(model, val_loader, device)
        best_thr     = best_f1_threshold(y_val, p_val)

        # ---------------- Metrics with tuned threshold --------------
        val_metrics = full_metrics(y_val, p_val, best_thr)

        y_ext, p_ext = inference(model, ext_loader, device)
        ext_metrics  = full_metrics(y_ext, p_ext, best_thr)

        if val_metrics['f1']['score'] > best_val_f1:
            best_val_f1 = val_metrics['f1']['score']
            torch.save({'state_dict':model.state_dict(), 'thr':best_thr},
                       'best_model.pt')

        # ------------------------- Logging --------------------------
        row = {'epoch':epoch, 'train_loss':train_loss, 'thr':best_thr}
        for k,v in val_metrics.items(): row[f'val_{k}'] = v['score']
        for k,v in ext_metrics.items(): row[f'ext_{k}'] = v['score']
        history.append(row)
        pd.DataFrame(history).to_csv('training_log.csv', index=False)

        # ------------------------- Console --------------------------
        print(f"\nEpoch {epoch+1}/{args.epochs} | train-loss {train_loss:.4f} | thr={best_thr:.2f} | time {time.time()-t0:5.1f}s")
        print_table(val_metrics,  "   VAL")
        print_table(ext_metrics, "EXTERNAL")

    print("\nFinished. Best F1 (val):", best_val_f1, "  |  best model -> best_model.pt")

# --------------------------------- CLI ----------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='.', type=str)
    ap.add_argument('--epochs',    default=EPOCHS,    type=int)
    ap.add_argument('--batch_size',default=BATCH_SIZE,type=int)
    ap.add_argument('--img_size',  default=IMG_SIZE,  type=int)
    ap.add_argument('--lr',        default=LR,        type=float)
    ap.add_argument('--weight_decay', default=WEIGHT_DECAY, type=float)
    ap.add_argument('--no_pretrain', action='store_true')
    args = ap.parse_args()
    main(args)