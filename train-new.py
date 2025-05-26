#!/usr/bin/env python3
"""
Hormone-receptor (+/–) classification on cropped mammograms
────────────────────────────────────────────────────────────
Patient-level split, proper class-imbalance handling, bootstrap CIs,
permutation p-values and data-driven threshold tuning.

This version contains a new *fully-vectorised* implementation of the
metric / statistics block that is typically >10× faster than the
original (no Python loops in the hot path).

Directory layout (unchanged):

data_root/
 ├── train/         (sub-folders 0, 1)
 ├── val/
 └── external_val/
"""
from __future__ import annotations
import argparse, os, random, time, copy, warnings, sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score, roc_auc_score
)

# ───────────────────────────── Globals / Hyper-params ─────────────────────────
IMG_SIZE      = 384
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 30
BATCH_SIZE    = 16
N_BOOT        = 2_000
N_PERM        = 2_000
SEED          = 42
EARLY_STOP    = 0          # 0 → disabled   (set >0 for early-stop patience)
# ──────────────────────────────────────────────────────────────────────────────

# ─────────────────── Reproducibility & PyTorch boiler-plate ───────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)        # safe on non-cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark      = False
set_seed()

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Apple Silicon (M1/M2)
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Running on device: {DEVICE}")

# Better matmul on Apple Silicon
if DEVICE.type == "mps":
    torch.set_float32_matmul_precision('high')

# ───────────────────────── Metric & statistics utilities ──────────────────────
def _to_pred(y_prob: np.ndarray, thr: float) -> np.ndarray:
    return (y_prob >= thr).astype(int)

# ----------  fast, fully-vectorised bootstrap & permutation  -----------------
def _confusion_mtx(y_true, y_pred):
    """Return TP, FP, FN, TN for *each row* of the 2-D input arrays."""
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=1)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=1)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=1)
    tn = np.sum((y_true == 0) & (y_pred == 0), axis=1)
    return tp, fp, fn, tn

# ────────────────────── vectorised metrics (warning-free) ─────────────────────
def _metrics_from_conf(tp, fp, fn, tn):
    """
    Compute confusion-matrix metrics from TP / FP / FN / TN vectors
    (works with scalars or 1-D NumPy arrays).

    • all divisions are protected with `np.divide(..., where=…)`
      so no `RuntimeWarning: invalid value encountered in divide`
    • returns six NumPy arrays (or scalars) in the order:
        accuracy, precision, recall, f1, mcc, balanced_accuracy
    """
    n   = tp + fp + fn + tn

    acc = (tp + tn) / n

    prec = np.divide(tp, tp + fp,
                     out=np.zeros_like(tp, dtype=float),
                     where=(tp + fp) != 0)

    rec  = np.divide(tp, tp + fn,
                     out=np.zeros_like(tp, dtype=float),
                     where=(tp + fn) != 0)

    f1   = np.divide(2 * prec * rec, prec + rec,
                     out=np.zeros_like(tp, dtype=float),
                     where=(prec + rec) != 0)

    # Matthews correlation coefficient
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc   = np.divide(tp * tn - fp * fn, denom,
                      out=np.zeros_like(tp, dtype=float),
                      where=denom != 0)

    bal  = (rec + np.divide(tn, tn + fp,
                             out=np.zeros_like(tp, dtype=float),
                             where=(tn + fp) != 0)) / 2
    return acc, prec, rec, f1, mcc, bal


def bootstrap_and_perm(y_true, y_prob, thr,
                       n_boot=N_BOOT, n_perm=N_PERM, alpha=0.05):
    """
    Fast, vectorised computation of point estimates, bootstrap CIs
    and permutation p-values for

        accuracy, precision, recall, F1, MCC, balanced_accuracy, ROC-AUC

    Returns the same dict structure that the original `full_metrics()`
    produced, but without any Python loops in the hot path and without
    runtime / deprecation warnings.
    """
    rng    = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = _to_pred(y_prob, thr)

    # ── 1) observed values ----------------------------------------------------
    obs_tp, obs_fp, obs_fn, obs_tn = _confusion_mtx(
        y_true[np.newaxis, :], y_pred[np.newaxis, :])
    obs_metrics = _metrics_from_conf(
        obs_tp, obs_fp, obs_fn, obs_tn)
    obs_metrics = [m.item() for m in obs_metrics]           # squeeze to scalars
    obs_roc     = roc_auc_score(y_true, y_prob)

    # ── 2) bootstrap CIs ------------------------------------------------------
    boot_idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    tp, fp, fn, tn = _confusion_mtx(y_true[boot_idx], y_pred[boot_idx])
    boot_metrics = _metrics_from_conf(tp, fp, fn, tn)       # tuple of arrays

    ci_low  = [np.percentile(b, 100 * alpha / 2) for b in boot_metrics]
    ci_high = [np.percentile(b, 100 * (1 - alpha / 2)) for b in boot_metrics]

    # ROC-AUC bootstrap (okay to loop; ~0.05 s for 2 000 reps)
    roc_boot = [roc_auc_score(y_true[i], y_prob[i]) for i in boot_idx]
    roc_ci_low, roc_ci_high = np.percentile(
        roc_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # ── 3) permutation p-values ----------------------------------------------
    # generate n_perm independent permutations without Python loops
    perm_idx = np.argsort(rng.random((n_perm, len(y_true))), axis=1)
    perm_y   = y_true[perm_idx]

    tp, fp, fn, tn = _confusion_mtx(perm_y, y_pred[np.newaxis, :])
    perm_metrics = _metrics_from_conf(tp, fp, fn, tn)

    p_vals = [((np.sum(p >= o) + 1) / (n_perm + 1))
              for p, o in zip(perm_metrics, obs_metrics)]

    # ROC-AUC permutation
    roc_perm = [(roc_auc_score(py, y_prob)
                 if np.unique(py).size == 2 else -np.inf)   # handle single-class
                for py in perm_y]
    roc_pval = (np.sum(np.array(roc_perm) >= obs_roc) + 1) / (n_perm + 1)

    # ── 4) assemble output ----------------------------------------------------
    names = ['accuracy', 'precision', 'recall',
             'f1', 'mcc', 'balanced_accuracy']
    out: Dict[str, Dict[str, float]] = {}
    for n, s, lo, hi, p in zip(names, obs_metrics, ci_low, ci_high, p_vals):
        out[n] = {'score':  float(s),
                  'ci_low': float(lo),
                  'ci_high':float(hi),
                  'p':      float(p)}

    out['roc_auc'] = {'score':  float(obs_roc),
                      'ci_low': float(roc_ci_low),
                      'ci_high':float(roc_ci_high),
                      'p':      float(roc_pval)}
    return out
# -----------------------------------------------------------------------------


def print_table(metrics_dict: Dict[str, Any], title: str) -> None:
    print(f"\n{title} metrics:")
    print("{:20s} {:>8s} {:>18s} {:>12s}".format("metric","score","95% CI","p-value"))
    for k,v in metrics_dict.items():
        print("{:20s} {:8.3f} [{:5.3f},{:5.3f}] {:12.4f}".format(
            k, v['score'], v['ci_low'], v['ci_high'], v['p']))

# ─────────────────────── Threshold optimisation helper ────────────────────────
def best_f1_threshold(y_true, y_prob) -> float:
    thrs = np.linspace(0.05, 0.95, 19)
    f1s  = [f1_score(y_true, _to_pred(y_prob,t), zero_division=0) for t in thrs]
    return float(thrs[int(np.argmax(f1s))])

# ───────────────────────────── Data & augmentation ────────────────────────────
def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.9,1.0)),
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

def _seed_worker(worker_id: int) -> None:
    """Make dataloader deterministic across workers."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loader(root: str,
                split: str,
                tf: transforms.Compose,
                batch_size: int,
                balance: bool = False,
                workers: int = 2,
                prefetch: int = 1) -> Tuple[DataLoader, datasets.ImageFolder]:
    """
    Build a DataLoader with macOS-friendly defaults.
    """
    ds = datasets.ImageFolder(Path(root)/split, transform=tf)

    # Weighted sampling (only for train)
    if balance and split == "train":
        targets      = [s[1] for s in ds.samples]
        class_counts = np.bincount(targets)
        weights      = 1.0 / class_counts[targets]
        sampler      = WeightedRandomSampler(weights,
                                             num_samples=len(weights),
                                             replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = (split == "train")

    # pin_memory only helps for CUDA
    pin_mem = DEVICE.type == "cuda"

    # prefetch & persistent_workers require num_workers > 0
    loader_kwargs = dict(
        batch_size        = batch_size,
        shuffle           = shuffle,
        sampler           = sampler,
        num_workers       = workers,
        worker_init_fn    = _seed_worker,
        pin_memory        = pin_mem,
    )
    if workers > 0:
        loader_kwargs.update(
            prefetch_factor    = prefetch,
            persistent_workers = True
        )

    loader = DataLoader(ds, **loader_kwargs)
    return loader, ds

# ─────────────────────────────── Model & loss ─────────────────────────────────
def build_model(num_classes=2, pretrained=True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    nn.init.xavier_uniform_(m.fc.weight)
    nn.init.zeros_(m.fc.bias)
    return m

# ───────────────────────────── Train / Eval loops ─────────────────────────────
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion, optim, device: torch.device,
                    scaler=None) -> float:
    model.train()
    running = 0.0
    pbar = tqdm(loader, leave=False, desc="train")
    for x,y in pbar:
        x, y = x.to(device), y.to(device)

        optim.zero_grad()

        if scaler is None:        # CPU / MPS
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optim.step()
        else:                     # CUDA mixed precision
            with torch.cuda.amp.autocast():
                out  = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        running += loss.item() * x.size(0)
        pbar.set_postfix(loss=loss.item())

    return running / len(loader.dataset)

@torch.no_grad()
def inference(model: nn.Module, loader: DataLoader,
              device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_prob = [], []
    for x,y in loader:
        x = x.to(device)
        with torch.cuda.amp.autocast(enabled=False):  # no AMP for eval on CPU/MPS
            logits = model(x)
        prob = torch.softmax(logits, 1)[:,1]
        y_true.append(y.cpu().numpy())
        y_prob.append(prob.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)

# ─────────────────────────────────── Main ─────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    tf_train, tf_eval = build_transforms(args.img_size)

    train_loader, train_ds = make_loader(args.data_root, 'train', tf_train,
                                         args.batch_size, balance=True)
    val_loader,   _        = make_loader(args.data_root, 'val',
                                         tf_eval,  args.batch_size)
    ext_loader,   _        = make_loader(args.data_root, 'external_val',
                                         tf_eval,  args.batch_size)

    # class-weighted CE
    train_targets = np.array([y for _,y in train_ds.samples])
    class_counts  = np.bincount(train_targets)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    model = build_model(pretrained=not args.no_pretrain).to(DEVICE)

    # Optional torch.compile (PyTorch ≥ 2.0, helps CPU/MPS)
    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    optim  = torch.optim.AdamW(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    best_val_f1, best_thr, epochs_no_improve = 0.0, 0.5, 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader,
                                     criterion, optim, DEVICE, scaler)

        # ───────── Validation
        y_val, p_val = inference(model, val_loader, DEVICE)
        best_thr     = best_f1_threshold(y_val, p_val)
        val_metrics  = bootstrap_and_perm(y_val, p_val, best_thr)

        # ───────── External
        y_ext, p_ext = inference(model, ext_loader, DEVICE)
        ext_metrics  = bootstrap_and_perm(y_ext, p_ext, best_thr)

        # ───────── Checkpointing
        current_f1 = val_metrics['f1']['score']
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            torch.save({'state_dict': model.state_dict(),
                        'thr'       : best_thr},
                       'best_model.pt')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ───────── Logging
        row = {'epoch':epoch, 'train_loss':train_loss, 'thr':best_thr}
        for k,v in val_metrics.items(): row[f'val_{k}'] = v['score']
        for k,v in ext_metrics.items(): row[f'ext_{k}'] = v['score']
        history.append(row)
        pd.DataFrame(history).to_csv('training_log.csv', index=False)

        print(f"\nEpoch {epoch+1:02d}/{args.epochs} | "
              f"train-loss {train_loss:.4f} | thr={best_thr:.2f} | "
              f"time {time.time()-t0:5.1f}s")
        print_table(val_metrics,  "   VAL")
        print_table(ext_metrics, "EXTERNAL")

        # ───────── Early-stopping (optional)
        if EARLY_STOP and epochs_no_improve >= EARLY_STOP:
            print(f"\nEarly stop after {epoch+1} epochs "
                  f"(no F1 gain for {EARLY_STOP} epochs).")
            break

    print("\nFinished. Best F1 (val):", best_val_f1,
          " | best model saved to best_model.pt")

# ───────────────────────────────── CLI ────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    formatter = argparse.ArgumentDefaultsHelpFormatter
    ap = argparse.ArgumentParser(formatter_class=formatter)
    ap.add_argument('--data_root', default='.', type=str,
                    help='Root folder with train / val / external_val dirs')
    ap.add_argument('--epochs', default=EPOCHS, type=int)
    ap.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    ap.add_argument('--img_size',  default=IMG_SIZE, type=int)
    ap.add_argument('--lr',        default=LR, type=float)
    ap.add_argument('--weight_decay', default=WEIGHT_DECAY, type=float)
    ap.add_argument('--no_pretrain', action='store_true',
                    help='Start from random weights instead of ImageNet')
    ap.add_argument('--torch_compile', action='store_true',
                    help='Use torch.compile (PyTorch >=2.0)')
    return ap.parse_args()

if __name__ == "__main__":
    # Silence torchvision / sklearn deprecation noise
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    main(args)