#!/usr/bin/env python3
# ---------------------------------------------------------------
#   Hormone-Receptor classifier on lesion crops (patient-level eval)
#   2024-xx-xx
# ---------------------------------------------------------------

import os, re, argparse, warnings, math, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, balanced_accuracy_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from tabulate import tabulate

# ---------- 0.  device ---------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", DEVICE)

# ---------- 1.  dataset --------------------------------------------------------
PATIENT_REGEX = re.compile(r"^(?:train_|val_)?(.+?)_\d+$", re.I)   # capture patient id

def get_patient_id(fname: str) -> str:
    m = PATIENT_REGEX.match(Path(fname).stem)
    if m is None:
        raise RuntimeError(f"Cannot extract patient id from {fname}")
    return m.group(1).lower()

class CropDataset(Dataset):
    """
    root_dir/
        1/   (positive)
        2/   (negative)
            *.jpg
    """
    def __init__(self, root_dir, transform):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.samples = self._gather()

    def _gather(self):
        items = []
        for cls in self.classes:
            for img_path in (self.root_dir/cls).glob("*.[jp][pn]g"):
                pid = get_patient_id(img_path.name)
                items.append((img_path, self.class_to_idx[cls], pid))
        return items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, pid = self.samples[idx]
        img = Image.open(path).convert("L")        # single-channel
        if self.transform:
            img = self.transform(img)
        return img.float(), label, pid

# ---------- 2.  transforms -----------------------------------------------------
def build_transforms(img_size):
    mean, std = [0.5], [0.5]          # grayscale
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.9,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ColorJitter(0.15,0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(lambda x: x.repeat(3,1,1))  # convert 1-ch → 3-ch
    ])
    val_tf   = transforms.Compose([
        transforms.Resize(int(img_size*1.05)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
        transforms.Lambda(lambda x: x.repeat(3,1,1))
    ])
    return train_tf, val_tf

# ---------- 3.  model ----------------------------------------------------------
def create_model(num_classes):
    model = models.resnet18(weights='IMAGENET1K_V1')
    # Freeze everything except layer4 & fc
    for name,param in model.named_parameters():
        param.requires_grad = False
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

# ---------- 4.  helpers --------------------------------------------------------
def find_best_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    balacc = (tpr + (1-fpr))/2
    return thr[np.argmax(balacc)]

def aggregate_patient_metrics(labels, probs, patients, threshold):
    per_patient = defaultdict(list)
    for y,p,pid in zip(labels, probs, patients):
        per_patient[pid].append((y,p))
    y_true, y_pred, y_score = [], [], []
    for pid, lst in per_patient.items():
        ys, ps = zip(*lst)
        y_true.append(ys[0])                # should all be same
        score = np.mean(ps)                 # mean-pool
        y_score.append(score)
        y_pred.append(int(score>=threshold))
    return (balanced_accuracy_score(y_true, y_pred),
            matthews_corrcoef(y_true, y_pred),
            roc_auc_score(y_true, y_score))

def metric_table(res):
    rows = [(k,f"{v:.4f}") for k,v in res.items()]
    print(tabulate(rows, headers=["metric","value"], tablefmt="pipe"))

# ---------- 5.  main train/eval loop ------------------------------------------
def run(args):
    # 5.1 transforms & datasets
    train_tf, val_tf = build_transforms(args.img_size)
    data_dirs = {p:Path(args.data_dir)/p for p in ('train','val','external_val')}
    datasets = {
        'train': CropDataset(data_dirs['train'], train_tf),
        'val'  : CropDataset(data_dirs['val'],   val_tf)
    }
    if data_dirs['external_val'].exists():
        datasets['external_val'] = CropDataset(data_dirs['external_val'], val_tf)

    # 5.2 weighted sampler for TRAIN
    labels_train = [lbl for _,lbl,_ in datasets['train'].samples]
    class_cnt = Counter(labels_train)
    weights = [1/class_cnt[lbl] for lbl in labels_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loaders = {
        'train': DataLoader(datasets['train'], batch_size=args.bs,
                            sampler=sampler, num_workers=0, pin_memory=True),
        'val'  : DataLoader(datasets['val'],   batch_size=args.bs,
                            shuffle=False, num_workers=0, pin_memory=True)
    }
    if 'external_val' in datasets:
        loaders['external_val'] = DataLoader(datasets['external_val'],
                                             batch_size=args.bs, shuffle=False,
                                             num_workers=0, pin_memory=True)
    # 5.3 model / optim
    model = create_model(len(datasets['train'].classes)).to(DEVICE)
    # class-weighted loss (1/n_crops_per_class)
    cw = torch.tensor([1/class_cnt[i] for i in range(len(class_cnt))],
                      dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimiser = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min',
                                                           factor=0.2, patience=3)

    best_val_bacc = 0
    patience_ctr  = 0

    for epoch in range(1,args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        # ----  train  ---------------------------------------------------------
        model.train()
        tloss, tcorrect = 0, 0
        for imgs, lbls, _ in loaders['train']:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimiser.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tloss += loss.item()*imgs.size(0)
            tcorrect += (logits.argmax(1)==lbls).sum().item()
        tloss /= len(datasets['train'])
        tacc  = tcorrect/len(datasets['train'])
        print(f"train loss={tloss:.4f} acc={tacc:.4f}")

        # ----  validation (crop level)  --------------------------------------
        model.eval()
        with torch.no_grad():
            val_labels, val_probs, val_pat = [], [], []
            vloss, vcorrect = 0, 0
            for imgs,lbls,pid in loaders['val']:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                vloss += loss.item()*imgs.size(0)
                preds = logits.argmax(1)
                vcorrect += (preds==lbls).sum().item()
                val_labels.extend(lbls.cpu().numpy())
                val_probs.extend(torch.softmax(logits,1)[:,1].cpu().numpy())
                val_pat.extend(pid)
            vloss /= len(datasets['val'])
            vacc  = vcorrect/len(datasets['val'])
        print(f"val   loss={vloss:.4f} acc_crop={vacc:.4f}")

        # ----  patient-level threshold selection -----------------------------
        best_thr = find_best_threshold(val_labels, val_probs)
        v_bacc, v_mcc, v_auc = aggregate_patient_metrics(val_labels, val_probs,
                                                         val_pat, best_thr)
        print(f"val PATIENT-level  bAcc={v_bacc:.3f}  MCC={v_mcc:.3f}  AUC={v_auc:.3f}  thr={best_thr:.3f}")

        scheduler.step(vloss)

        # ----  early stop on patient balanced-accuracy -----------------------
        if v_bacc > best_val_bacc:
            best_val_bacc = v_bacc
            patience_ctr = 0
            best_state   = {
                'model': model.state_dict(),
                'thr'  : best_thr,
                'classes': datasets['train'].classes
            }
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print("Early stopping.")
                break

    # save best
    torch.save(best_state, args.out)

    # -------- evaluate on external ------------------------------------------
    if 'external_val' in loaders:
        print("\n--- External validation ---")
        model.load_state_dict(best_state['model'])
        model.eval()
        ext_lab, ext_prob, ext_pat = [], [], []
        with torch.no_grad():
            for imgs,lbls,pid in loaders['external_val']:
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                ext_lab.extend(lbls.numpy())
                ext_prob.extend(torch.softmax(logits,1)[:,1].cpu().numpy())
                ext_pat.extend(pid)
        e_bacc, e_mcc, e_auc = aggregate_patient_metrics(ext_lab, ext_prob,
                                                         ext_pat, best_state['thr'])
        res = dict(balanced_acc=e_bacc, mcc=e_mcc, auc=e_auc)
        metric_table(res)
        # confusion matrix for patients
        pats_pred = defaultdict(list)
        for y,p,pid in zip(ext_lab,ext_prob, ext_pat):
            pats_pred[pid].append((y,p))
        y_true_pat, y_pred_pat = [], []
        for pid,lst in pats_pred.items():
            y_true_pat.append(lst[0][0])
            y_pred_pat.append(int(np.mean([p for _,p in lst])>=best_state['thr']))
        print("Confusion matrix (patient level):\n", confusion_matrix(y_true_pat,y_pred_pat))

# ---------- 6.  cli -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="dataset root with train/ val/ external_val/")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="best_mammo.pth")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    run(args)
