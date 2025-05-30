import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_curve, auc
)
from sklearn.utils import resample
from sklearn.model_selection import StratifiedGroupKFold # Changed from GroupKFold
from PIL import Image
from glob import glob
import time
import copy
import random
import argparse
import re

# --- Configuration (defaults for argparse) ---
DATA_DIR = './' # Default, will be updated by argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME_TAG = "resnet18_rep_sgkf" # Updated tag

NUM_CLASSES = 2
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-5 # Default from previous successful run
WEIGHT_DECAY = 5e-4  # Default from previous successful run
LR_SCHEDULER_PATIENCE = 7
IMAGE_SIZE = (224, 224)
BEST_METRIC_FOR_SAVING = "pr_auc"
POSITIVE_CLASS_LABEL_FOR_PR_AUC = 1 # Corresponds to class '2' if classes are '1', '2'

SEED = 86

N_BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

USE_CV = False # Default, can be overridden by CLI
CV_FOLDS = 5   # Default, can be overridden by CLI

# Global placeholders, will be defined in main
MODEL_SAVE_PATH = None
ALPHA = None
class_names = [] # Will be populated in main
class_to_idx = {} # Will be populated in main
# loss_weights_cpu defined globally, moved to device in main

# --- Helper Function to Extract Patient ID ---
def get_patient_id_from_filename(filename):
    name_part = os.path.basename(filename)
    if name_part.startswith("train_"):
        name_part = name_part[len("train_"):]
    elif name_part.startswith("val_"):
        name_part = name_part[len("val_"):]
    
    match = re.match(r'(.+)_([0-9]+)\.(jpg|jpeg|png|gif|bmp|tif|tiff)$', name_part, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        # print(f"Warning: Fallback Patient ID parsing for {filename}")
        return name_part.split('.')[0]

# --- Custom Dataset for CV ---
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L') 
        except FileNotFoundError:
            print(f"ERROR: Image file not found: {img_path}")
            return torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1])), torch.tensor(0) # Placeholder
        except Exception as e:
            print(f"ERROR: Could not load image {img_path}: {e}")
            return torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1])), torch.tensor(0)

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Reproducibility ---
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if torch.backends.mps.is_available():
        try: torch.mps.manual_seed(seed_value)
        except AttributeError: print("torch.mps.manual_seed not available.")

# --- Device Configuration ---
if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")

# --- Data Transformations ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'external_val': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

loss_weights_cpu = torch.tensor([1.0, 2.5], dtype=torch.float32) # Define on CPU first

# --- Training Function (Refactored for CV) ---
def train_model_core(model, criterion, optimizer, scheduler,
                     current_train_loader, current_val_loader,
                     current_train_size, current_val_size,
                     num_epochs_to_run, best_metric_name_to_monitor,
                     positive_class_label_for_pr_auc_local, device_local,
                     model_save_path_local, 
                     current_lr_for_stop_condition,
                     is_cv_fold_run=False):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric_score = -1.0 

    history_train_losses, history_val_losses = [], []
    history_train_metric, history_val_metric = [], []
    no_improvement_epochs_lr, min_lr_stop_patience = 0, 5

    metric_name_for_print = "PR AUC" if best_metric_name_to_monitor == "pr_auc" else "Balanced Acc"

    for epoch in range(num_epochs_to_run):
        epoch_print_prefix = "CV Fold Epoch" if is_cv_fold_run else "Epoch"
        print(f'{epoch_print_prefix} {epoch+1}/{num_epochs_to_run}\n' + '-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            all_labels_epoch, all_preds_epoch, all_probs_class_positive_epoch = [], [], []
            current_loader = current_train_loader if phase == 'train' else current_val_loader
            current_phase_size = current_train_size if phase == 'train' else current_val_size

            if current_phase_size == 0:
                if phase == 'train': history_train_losses.append(np.nan); history_train_metric.append(np.nan)
                else: history_val_losses.append(np.nan); history_val_metric.append(np.nan)
                continue

            for inputs, labels in current_loader:
                inputs, labels = inputs.to(device_local), labels.to(device_local)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward(); optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                all_preds_epoch.extend(preds.cpu().numpy())
                all_labels_epoch.extend(labels.cpu().numpy())
                if phase == 'val' or (phase == 'train' and best_metric_name_to_monitor == 'pr_auc'): 
                    probabilities = torch.softmax(outputs, dim=1)
                    all_probs_class_positive_epoch.extend(probabilities[:, positive_class_label_for_pr_auc_local].detach().cpu().numpy())
            
            epoch_loss = running_loss / current_phase_size if current_phase_size > 0 else 0
            all_labels_epoch_np = np.array(all_labels_epoch)
            all_preds_epoch_np = np.array(all_preds_epoch)
            
            epoch_bal_acc = balanced_accuracy_score(all_labels_epoch_np, all_preds_epoch_np) if len(all_labels_epoch_np) > 0 else 0.0
            epoch_pr_auc = 0.0
            if len(all_labels_epoch_np) > 0 and len(np.unique(all_labels_epoch_np)) > 1 and \
               len(all_probs_class_positive_epoch) == len(all_labels_epoch_np):
                all_probs_class_positive_epoch_np = np.array(all_probs_class_positive_epoch)
                try:
                    precision_p, recall_p, _ = precision_recall_curve(all_labels_epoch_np, all_probs_class_positive_epoch_np, pos_label=positive_class_label_for_pr_auc_local)
                    if len(recall_p) > 1 and len(precision_p) > 1 : # Need at least 2 points for auc
                         epoch_pr_auc = auc(recall_p, precision_p)
                except ValueError: epoch_pr_auc = 0.0
            
            current_epoch_metric_value = epoch_pr_auc if best_metric_name_to_monitor == "pr_auc" else epoch_bal_acc
            
            if phase == 'train':
                history_train_losses.append(epoch_loss); history_train_metric.append(current_epoch_metric_value) 
                print(f'{phase} Loss: {epoch_loss:.4f} {metric_name_for_print}: {current_epoch_metric_value:.4f} (Bal Acc: {epoch_bal_acc:.4f})')
            else: # val phase
                history_val_losses.append(epoch_loss); history_val_metric.append(current_epoch_metric_value) 
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(current_epoch_metric_value) 
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < old_lr: print(f"Epoch {epoch+1}: LR reduced from {old_lr} to {new_lr}."); no_improvement_epochs_lr = 0
                else: no_improvement_epochs_lr += 1
                
                if current_epoch_metric_value > best_metric_score:
                    best_metric_score = current_epoch_metric_value
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if model_save_path_local:
                        torch.save(model.state_dict(), model_save_path_local)
                        print(f"Best model for this run saved to {model_save_path_local} with {metric_name_for_print}: {best_metric_score:.4f}")
                    no_improvement_epochs_lr = 0 
                print(f'{phase} Loss: {epoch_loss:.4f} {metric_name_for_print}: {current_epoch_metric_value:.4f} (Bal Acc: {epoch_bal_acc:.4f})')
        print()
        current_lr_val = optimizer.param_groups[0]['lr']
        if current_lr_val <= (current_lr_for_stop_condition * 0.01 * 0.5) and no_improvement_epochs_lr >= min_lr_stop_patience:
             print(f"Early stopping triggered."); break

    time_elapsed = time.time() - since
    print(f'{"CV Fold Training" if is_cv_fold_run else "Training"} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val {metric_name_for_print} for this run: {best_metric_score:4f}')
    model.load_state_dict(best_model_wts)
    return model, history_train_losses, history_val_losses, history_train_metric, history_val_metric, best_metric_score

# --- Threshold Tuning Function ---
def find_optimal_threshold(labels_np, probs_class1_np, target_metric='f1_minority', minority_class_label=1):
    optimal_threshold, best_metric_value = 0.5, -1.0
    if len(labels_np) == 0 or len(probs_class1_np) == 0 or len(np.unique(labels_np)) < 2:
        print(f"Cannot tune threshold: Insufficient or imbalanced data. Labels unique: {np.unique(labels_np)}")
        return 0.5, -1.0 # Return default and no improvement

    unique_sorted_probs = np.sort(np.unique(probs_class1_np))
    if len(unique_sorted_probs) == 0: return 0.5, -1.0
    
    candidate_thresholds = (unique_sorted_probs[:-1] + unique_sorted_probs[1:]) / 2.0 if len(unique_sorted_probs) > 1 else unique_sorted_probs
    candidate_thresholds = np.clip(np.sort(np.unique(np.append(candidate_thresholds, [0.001, 0.5, 0.999]))), 0.001, 0.999) # Ensure valid range
    if not candidate_thresholds.size: return 0.5, -1.0

    for threshold_val in candidate_thresholds:
        preds_at_threshold = (probs_class1_np >= threshold_val).astype(int)
        if target_metric == 'f1_minority':
            _, _, f1, support = precision_recall_fscore_support(labels_np, preds_at_threshold, average=None, labels=[0,1], zero_division=0)
            if len(f1) > minority_class_label and support[minority_class_label] > 0 : # Check if minority class has support
                current_metric_value = f1[minority_class_label]
            else: # No instances of minority class predicted or present
                current_metric_value = 0.0
        elif target_metric == 'balanced_accuracy': current_metric_value = balanced_accuracy_score(labels_np, preds_at_threshold)
        else: raise ValueError(f"Unsupported target_metric: {target_metric}")
        
        if current_metric_value > best_metric_value:
            best_metric_value, optimal_threshold = current_metric_value, threshold_val
        elif current_metric_value == best_metric_value and abs(threshold_val - 0.5) < abs(optimal_threshold - 0.5):
            optimal_threshold = threshold_val # Prefer threshold closer to 0.5 if metric is tied
    print(f"Optimal threshold: {optimal_threshold:.4f} for '{target_metric}' (Value: {best_metric_value:.4f})")
    return optimal_threshold, best_metric_value

# --- Evaluation Functions ---
def calculate_metrics_bootstrap(y_true, y_pred, y_probs_class1):
    # (Uses global POSITIVE_CLASS_LABEL_FOR_PR_AUC)
    metrics = {}
    is_problematic_sample = len(np.unique(y_true)) < 2 
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    try: metrics['mcc'] = matthews_corrcoef(y_true, y_pred) if not is_problematic_sample else 0.0
    except ValueError: metrics['mcc'] = 0.0
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1], zero_division=0)
    metrics['precision_hr+'] = precision[0] if len(precision) > 0 else 0.0
    metrics['recall_hr+'] = recall[0] if len(recall) > 0 else 0.0
    metrics['f1_hr+'] = f1[0] if len(f1) > 0 else 0.0
    metrics['precision_hr-'] = precision[1] if len(precision) > 1 else 0.0
    metrics['recall_hr-'] = recall[1] if len(recall) > 1 else 0.0
    metrics['f1_hr-'] = f1[1] if len(f1) > 1 else 0.0
    metrics['specificity'] = recall[0] if len(recall) > 0 else 0.0 # Specificity = Recall of class 0
    
    try: metrics['roc_auc'] = roc_auc_score(y_true, y_probs_class1) if not is_problematic_sample and len(y_true) > 0 else 0.5
    except ValueError: metrics['roc_auc'] = 0.5

    try:
        if not is_problematic_sample and len(y_true) > 0 and len(y_probs_class1) > 0:
            precision_p_curve, recall_p_curve, _ = precision_recall_curve(y_true, y_probs_class1, pos_label=POSITIVE_CLASS_LABEL_FOR_PR_AUC)
            metrics['pr_auc'] = auc(recall_p_curve, precision_p_curve) if len(recall_p_curve) > 1 and len(precision_p_curve) > 1 else 0.0
        else: metrics['pr_auc'] = 0.0 
    except ValueError: metrics['pr_auc'] = 0.0
    return metrics

def evaluate_model(model, dataloader, phase_name="Test", fixed_threshold=None):
    # (Uses global device, POSITIVE_CLASS_LABEL_FOR_PR_AUC, N_BOOTSTRAP_SAMPLES, SEED, ALPHA, class_names, SCRIPT_DIR, MODEL_NAME_TAG)
    model.eval()
    all_preds_default_thresh, all_labels, all_probs_class1 = [], [], [] 
    with torch.no_grad():
        for inputs, labels_batch in dataloader: 
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = model(inputs)
            _, preds_default = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            all_probs_class1.extend(probabilities[:, POSITIVE_CLASS_LABEL_FOR_PR_AUC].cpu().numpy())
            all_preds_default_thresh.extend(preds_default.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    labels_np = np.array(all_labels)
    probs_class1_np = np.array(all_probs_class1)

    if len(labels_np) == 0:
        # print(f"No data to evaluate for {phase_name}.")
        empty_metrics_dict = {key: (np.nan, (np.nan, np.nan)) for key in ["accuracy", "balanced_accuracy", "mcc", "specificity", "pr_auc", "precision_hr+", "recall_hr+", "f1_hr+", "precision_hr-", "recall_hr-", "f1_hr-", "roc_auc"]}
        empty_metrics_dict["labels_np"] = labels_np; empty_metrics_dict["probs_class1_np"] = probs_class1_np
        return empty_metrics_dict

    current_threshold_for_print = f"{fixed_threshold:.4f}" if fixed_threshold is not None else "Default 0.5"
    preds_np = (probs_class1_np >= fixed_threshold).astype(int) if fixed_threshold is not None and POSITIVE_CLASS_LABEL_FOR_PR_AUC == 1 else \
               (probs_class1_np <= fixed_threshold).astype(int) if fixed_threshold is not None and POSITIVE_CLASS_LABEL_FOR_PR_AUC == 0 else \
               np.array(all_preds_default_thresh)

    print(f"\n--- Evaluation Metrics for {phase_name} (Threshold: {current_threshold_for_print}) ---")
    point_metrics = calculate_metrics_bootstrap(labels_np, preds_np, probs_class1_np)
    bootstrap_metrics_values = {key: [] for key in point_metrics.keys()} 
    n_samples = len(labels_np)

    if n_samples >= 10 and len(np.unique(labels_np)) >= NUM_CLASSES : # Ensure enough samples and both classes present for reliable bootstrapping
        for i in range(N_BOOTSTRAP_SAMPLES):
            indices = resample(np.arange(n_samples), n_samples=n_samples, random_state=SEED+i, stratify=labels_np if len(np.unique(labels_np)) > 1 else None)
            labels_boot = labels_np[indices]
            probs_class1_boot = probs_class1_np[indices]
            if len(np.unique(labels_boot)) < NUM_CLASSES: # Skip if bootstrap sample doesn't have all classes
                 for key_metric in bootstrap_metrics_values.keys(): bootstrap_metrics_values[key_metric].append(np.nan)
                 continue
            preds_boot = (probs_class1_boot >= fixed_threshold).astype(int) if fixed_threshold is not None and POSITIVE_CLASS_LABEL_FOR_PR_AUC == 1 else \
                         (probs_class1_boot <= fixed_threshold).astype(int) if fixed_threshold is not None and POSITIVE_CLASS_LABEL_FOR_PR_AUC == 0 else \
                         (np.array(all_preds_default_thresh)[indices]) 
            current_boot_metrics = calculate_metrics_bootstrap(labels_boot, preds_boot, probs_class1_boot)
            for key in bootstrap_metrics_values.keys(): bootstrap_metrics_values[key].append(current_boot_metrics.get(key, np.nan))
    else: # Not enough samples or classes for bootstrapping
        print(f"Skipping bootstrapping for {phase_name} (Samples: {n_samples}, Unique Labels: {np.unique(labels_np)})")

    metrics_with_ci = {}
    for key, values in bootstrap_metrics_values.items():
        valid_values = [v for v in values if not np.isnan(v)]
        point_val = point_metrics.get(key, np.nan)
        if not valid_values or not N_BOOTSTRAP_SAMPLES: # if no valid bootstrap values or bootstrapping was skipped
             metrics_with_ci[key] = (point_val, (np.nan, np.nan))
             continue
        lower_bound = np.percentile(valid_values, ALPHA * 100)
        upper_bound = np.percentile(valid_values, (1 - ALPHA) * 100)
        metrics_with_ci[key] = (point_val, (lower_bound, upper_bound))
    
    # If bootstrapping was skipped entirely, fill with point_metrics
    if not bootstrap_metrics_values or all(not v for v in bootstrap_metrics_values.values()):
        metrics_with_ci = {key: (value, (np.nan, np.nan)) for key, value in point_metrics.items()}

    # Print metrics (ensure class_names is populated)
    if class_names: # Check if class_names is populated
        metric_keys_to_print = ["accuracy", "balanced_accuracy", "specificity", "mcc",
                                f"precision_{class_names[0].lower()}", f"recall_{class_names[0].lower()}", f"f1_{class_names[0].lower()}",
                                f"precision_{class_names[1].lower()}", f"recall_{class_names[1].lower()}", f"f1_{class_names[1].lower()}",
                                "roc_auc", "pr_auc"] # Adapt if your metric keys are different
        # This print loop needs to map to the keys in metrics_with_ci correctly
        # Example for overall accuracy:
        acc_data = metrics_with_ci.get('accuracy', (np.nan, (np.nan, np.nan)))
        print(f"Overall Accuracy: {acc_data[0]:.4f} (95% CI: {acc_data[1][0]:.4f}-{acc_data[1][1]:.4f})")
        # ... (print other metrics similarly, mapping keys like 'precision_hr+' to your class names)
        bal_acc_data = metrics_with_ci.get('balanced_accuracy', (np.nan, (np.nan, np.nan)))
        print(f"Balanced Accuracy: {bal_acc_data[0]:.4f} (95% CI: {bal_acc_data[1][0]:.4f}-{bal_acc_data[1][1]:.4f})")
        spec_data = metrics_with_ci.get('specificity', (np.nan, (np.nan, np.nan))) # Specificity is recall of class 0
        print(f"Specificity (Recall {class_names[0]}): {spec_data[0]:.4f} (95% CI: {spec_data[1][0]:.4f}-{spec_data[1][1]:.4f})")
        mcc_data = metrics_with_ci.get('mcc', (np.nan, (np.nan, np.nan)))
        print(f"MCC: {mcc_data[0]:.4f} (95% CI: {mcc_data[1][0]:.4f}-{mcc_data[1][1]:.4f})")

        print(f"\nClass-wise metrics for {class_names[0]}:")
        prec_c0 = metrics_with_ci.get('precision_hr+', (np.nan, (np.nan, np.nan))) # Assuming hr+ maps to class 0
        rec_c0 = metrics_with_ci.get('recall_hr+', (np.nan, (np.nan, np.nan)))
        f1_c0 = metrics_with_ci.get('f1_hr+', (np.nan, (np.nan, np.nan)))
        print(f"  Precision: {prec_c0[0]:.4f} (CI: {prec_c0[1][0]:.4f}-{prec_c0[1][1]:.4f})")
        print(f"  Recall: {rec_c0[0]:.4f} (CI: {rec_c0[1][0]:.4f}-{rec_c0[1][1]:.4f})")
        print(f"  F1-score: {f1_c0[0]:.4f} (CI: {f1_c0[1][0]:.4f}-{f1_c0[1][1]:.4f})")

        print(f"Class-wise metrics for {class_names[1]}:")
        prec_c1 = metrics_with_ci.get('precision_hr-', (np.nan, (np.nan, np.nan))) # Assuming hr- maps to class 1
        rec_c1 = metrics_with_ci.get('recall_hr-', (np.nan, (np.nan, np.nan)))
        f1_c1 = metrics_with_ci.get('f1_hr-', (np.nan, (np.nan, np.nan)))
        print(f"  Precision: {prec_c1[0]:.4f} (CI: {prec_c1[1][0]:.4f}-{prec_c1[1][1]:.4f})")
        print(f"  Recall (Sensitivity): {rec_c1[0]:.4f} (CI: {rec_c1[1][0]:.4f}-{rec_c1[1][1]:.4f})") # Sensitivity is recall of positive class
        print(f"  F1-score: {f1_c1[0]:.4f} (CI: {f1_c1[1][0]:.4f}-{f1_c1[1][1]:.4f})")

        roc_auc_data = metrics_with_ci.get('roc_auc', (np.nan, (np.nan, np.nan)))
        pr_auc_data = metrics_with_ci.get('pr_auc', (np.nan, (np.nan, np.nan)))
        print(f"\nROC AUC (for {class_names[POSITIVE_CLASS_LABEL_FOR_PR_AUC]} as positive): {roc_auc_data[0]:.4f} (CI: {roc_auc_data[1][0]:.4f}-{roc_auc_data[1][1]:.4f})")
        print(f"PR AUC (for {class_names[POSITIVE_CLASS_LABEL_FOR_PR_AUC]} as positive): {pr_auc_data[0]:.4f} (CI: {pr_auc_data[1][0]:.4f}-{pr_auc_data[1][1]:.4f})")

    # Confusion Matrix
    if len(labels_np) > 0 and class_names:
        cm = confusion_matrix(labels_np, preds_np, labels=[0,1]) # Assuming labels are 0 and 1
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f"Pred {class_names[0]}", f"Pred {class_names[1]}"],
                    yticklabels=[f"True {class_names[0]}", f"True {class_names[1]}"])
        cm_title_thresh = "Def" if fixed_threshold is None else f"{fixed_threshold:.2f}"
        plt.title(f'CM - {phase_name} (Th: {cm_title_thresh})')
        plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.tight_layout()
        if SCRIPT_DIR and MODEL_NAME_TAG:
            plt.savefig(os.path.join(SCRIPT_DIR, f"cm_{MODEL_NAME_TAG}_{phase_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{cm_title_thresh.replace('.', 'p')}.png"))

    metrics_with_ci["labels_np"] = labels_np
    metrics_with_ci["probs_class1_np"] = probs_class1_np
    return metrics_with_ci

# --- Plotting training history ---
def plot_training_history(train_losses, val_losses, train_metric_scores, val_metric_scores, metric_name="Metric", suffix=""):
    # (Same as before)
    epochs_len = len(train_losses)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_len + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.legend(loc='upper right'); plt.title('Training and Validation Loss' + suffix); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs_len + 1), train_metric_scores, label=f'Training {metric_name}')
    plt.plot(range(1, len(val_metric_scores) + 1), val_metric_scores, label=f'Validation {metric_name}')
    plt.legend(loc='lower right'); plt.title(f'Training and Validation {metric_name}' + suffix); plt.xlabel('Epoch'); plt.ylabel(metric_name)
    plt.tight_layout()
    if SCRIPT_DIR and MODEL_NAME_TAG:
        plt.savefig(os.path.join(SCRIPT_DIR, f"training_history_{MODEL_NAME_TAG}_{metric_name.replace(' ', '_').lower()}{suffix.replace(' ', '_')}.png"))

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a ResNet18 model with optional StratifiedGroupKFold CV.")
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help="Root directory of the dataset.")
    parser.add_argument('--model_name_tag', type=str, default=MODEL_NAME_TAG, help="Tag for model and output files.")
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help="Initial learning rate.")
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help="Weight decay for optimizer.")
    parser.add_argument('--lr_scheduler_patience', type=int, default=LR_SCHEDULER_PATIENCE, help="Patience for LR scheduler.")
    parser.add_argument('--best_metric', type=str, default=BEST_METRIC_FOR_SAVING, choices=["balanced_accuracy", "pr_auc"], help="Metric to optimize for saving best model.")
    parser.add_argument('--positive_class_label', type=int, default=POSITIVE_CLASS_LABEL_FOR_PR_AUC, help="Integer label of the positive class (e.g., 0 or 1).")
    parser.add_argument('--seed', type=int, default=SEED, help="Random seed for reproducibility.")
    parser.add_argument('--n_bootstrap', type=int, default=N_BOOTSTRAP_SAMPLES, help="Number of bootstrap samples for CIs.")
    parser.add_argument('--confidence_level', type=float, default=CONFIDENCE_LEVEL, help="Confidence level for bootstrap CIs.")
    parser.add_argument('--use_cv', action='store_true', default=USE_CV, help="Enable StratifiedGroupKFold Cross-Validation.")
    parser.add_argument('--cv_folds', type=int, default=CV_FOLDS, help="Number of folds for Cross-Validation.")
    args = parser.parse_args()

    # Update Global Configuration Variables from Parsed Args
    DATA_DIR = args.data_dir
    MODEL_NAME_TAG = args.model_name_tag
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    LR_SCHEDULER_PATIENCE = args.lr_scheduler_patience
    BEST_METRIC_FOR_SAVING = args.best_metric
    POSITIVE_CLASS_LABEL_FOR_PR_AUC = args.positive_class_label
    SEED = args.seed
    N_BOOTSTRAP_SAMPLES = args.n_bootstrap
    CONFIDENCE_LEVEL = args.confidence_level
    USE_CV = args.use_cv
    CV_FOLDS = args.cv_folds

    MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, f'best_model_hr_{MODEL_NAME_TAG}.pth')
    ALPHA = (1 - CONFIDENCE_LEVEL) / 2.0
    set_seed(SEED)
    loss_weights = loss_weights_cpu.to(device) # Move to device

    # Print configuration
    print(f"--- Configuration ---")
    print(f"Script directory: {SCRIPT_DIR}") # Will be global
    # ... (all other print statements for config) ...
    print(f"Model save path (final model): {MODEL_SAVE_PATH}")
    if torch.backends.mps.is_available() and device.type == "mps": print(f"MPS seed set to {SEED}")
    print(f"Global random seed set to {SEED}")
    print(f"Using device: {device}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs (per fold/run): {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"LR scheduler patience: {LR_SCHEDULER_PATIENCE}")
    print(f"Best metric for saving/monitoring: {BEST_METRIC_FOR_SAVING}")
    print(f"Positive class label (for PR AUC etc.): {POSITIVE_CLASS_LABEL_FOR_PR_AUC}")
    print(f"Using Cross-Validation: {USE_CV}")
    if USE_CV: print(f"Number of CV folds: {CV_FOLDS}")
    print(f"Loss weights: {loss_weights.cpu().numpy()}")
    print("--- End Configuration ---")

    # Get class names and mapping (crucial for consistent labeling)
    # This needs to be done early, before data collection for CV
    try:
        # Assuming 'train' dir exists and reflects all classes
        temp_train_dir = os.path.join(DATA_DIR, 'train')
        if not os.path.isdir(temp_train_dir) or not os.listdir(temp_train_dir):
            raise FileNotFoundError(f"'train' directory ({temp_train_dir}) is missing or empty. Cannot determine class names.")
        
        # Create a temporary ImageFolder just to get class names and mapping
        _temp_ds = datasets.ImageFolder(temp_train_dir)
        class_names = _temp_ds.classes # Global assignment
        class_to_idx = _temp_ds.class_to_idx # Global assignment
        if len(class_names) != NUM_CLASSES:
            print(f"Warning: Discovered {len(class_names)} classes ({class_names}) but NUM_CLASSES is {NUM_CLASSES}.")
        print(f"Discovered class names: {class_names} (Mapping: {class_to_idx})")
        # Ensure POSITIVE_CLASS_LABEL_FOR_PR_AUC is valid if it's an index
        if not (0 <= POSITIVE_CLASS_LABEL_FOR_PR_AUC < NUM_CLASSES):
             raise ValueError(f"POSITIVE_CLASS_LABEL_FOR_PR_AUC ({POSITIVE_CLASS_LABEL_FOR_PR_AUC}) is out of bounds for {NUM_CLASSES} classes.")

    except Exception as e:
        print(f"Error determining class names: {e}")
        print("Please ensure your data directory structure is correct or define class_names manually.")
        exit()


    # --- Prepare data paths and labels for CV or standard run ---
    all_dev_image_paths = []
    all_dev_labels = []
    all_dev_patient_ids = []
    
    dev_splits_for_data_collection = ['train', 'val'] if USE_CV else ['train']
    
    for split_name in dev_splits_for_data_collection:
        for class_name_folder in class_names: # Use discovered class_names
            current_label = class_to_idx[class_name_folder]
            folder_path = os.path.join(DATA_DIR, split_name, class_name_folder)
            if not os.path.isdir(folder_path): continue
            for image_filename in os.listdir(folder_path):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
                    image_path = os.path.join(folder_path, image_filename)
                    patient_id = get_patient_id_from_filename(image_filename)
                    all_dev_image_paths.append(image_path)
                    all_dev_labels.append(current_label)
                    all_dev_patient_ids.append(patient_id)

    all_dev_image_paths = np.array(all_dev_image_paths)
    all_dev_labels = np.array(all_dev_labels)
    all_dev_patient_ids = np.array(all_dev_patient_ids)

    print(f"Total images collected for development pool: {len(all_dev_image_paths)}")
    if len(all_dev_patient_ids) > 0:
        print(f"Unique patients in development pool: {len(np.unique(all_dev_patient_ids))}")

    # External validation dataloader
    ext_val_path = os.path.join(DATA_DIR, 'external_val')
    external_val_dataloader = None
    if os.path.exists(ext_val_path) and any(f.is_dir() for f in os.scandir(ext_val_path)): # Check for subdirs
        try:
            external_val_dataset = datasets.ImageFolder(ext_val_path, data_transforms['external_val'])
            if len(external_val_dataset) > 0:
                external_val_dataloader = DataLoader(external_val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))
                print(f"External validation dataset size: {len(external_val_dataset)}")
            else: print("External validation directory is empty.")
        except Exception as e: print(f"Could not load external_val dataset: {e}")
    else: print("External validation directory not found or empty/no class subdirs.")

    final_trained_model = None
    avg_optimal_threshold_cv = 0.5 # Default if CV not run or fails to find thresholds

    if USE_CV:
        # ... (CV loop as in the previous full script, using StratifiedGroupKFold) ...
        print(f"\n--- Starting {CV_FOLDS}-Fold Cross-Validation (Using StratifiedGroupKFold) ---")
        if len(all_dev_image_paths) == 0:
            raise ValueError("No data for CV. Check DATA_DIR and 'train'/'val' subfolders.")
        if len(all_dev_patient_ids) > 0 and len(np.unique(all_dev_patient_ids)) < CV_FOLDS :
             print(f"Warning: Unique patients ({len(np.unique(all_dev_patient_ids))}) < CV_FOLDS ({CV_FOLDS}).")

        sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        cv_fold_metrics_summary = [] 
        cv_fold_optimal_thresholds = []

        for fold_idx, (train_indices, val_indices) in enumerate(sgkf.split(all_dev_image_paths, all_dev_labels, groups=all_dev_patient_ids)):
            print(f"\n--- CV Fold {fold_idx + 1}/{CV_FOLDS} ---")
            current_train_paths_fold = all_dev_image_paths[train_indices]
            current_train_labels_fold = all_dev_labels[train_indices]
            current_val_paths_fold = all_dev_image_paths[val_indices]
            current_val_labels_fold = all_dev_labels[val_indices]

            train_patients_this_fold = set(all_dev_patient_ids[train_indices])
            val_patients_this_fold = set(all_dev_patient_ids[val_indices])
            # Patient leakage check (optional, StratifiedGroupKFold should handle this by design for groups)
            # if train_patients_this_fold.intersection(val_patients_this_fold):
            #     print(f"Warning: Patient leakage detected in CV fold {fold_idx + 1}!")

            print(f"Fold {fold_idx+1}: Train images: {len(current_train_paths_fold)} (Patients: {len(train_patients_this_fold)}), Val images: {len(current_val_paths_fold)} (Patients: {len(val_patients_this_fold)})")
            val_fold_class_counts = np.bincount(current_val_labels_fold, minlength=NUM_CLASSES)
            print(f"Fold {fold_idx+1} Val Class Counts: {val_fold_class_counts}")

            if len(current_val_paths_fold) == 0:
                print(f"Skipping fold {fold_idx+1} due to empty validation set."); cv_fold_metrics_summary.append(np.nan); cv_fold_optimal_thresholds.append(np.nan); continue
            if np.any(val_fold_class_counts == 0) and BEST_METRIC_FOR_SAVING == 'pr_auc':
                 print(f"Warning: Fold {fold_idx+1} val set has a class with 0 samples. PR AUC issues likely.")

            train_dataset_fold = CustomImageDataset(current_train_paths_fold, current_train_labels_fold, transform=data_transforms['train'])
            val_dataset_fold = CustomImageDataset(current_val_paths_fold, current_val_labels_fold, transform=data_transforms['val'])

            fold_sampler = None
            if len(current_train_labels_fold) > 0:
                class_counts_train_fold = np.bincount(current_train_labels_fold, minlength=NUM_CLASSES)
                if not np.any(class_counts_train_fold == 0):
                    weight_per_class_fold = 1. / class_counts_train_fold
                    samples_weight_fold_list = [weight_per_class_fold[t] for t in current_train_labels_fold]
                    samples_weight_fold = torch.from_numpy(np.array(samples_weight_fold_list)).double()
                    fold_sampler = WeightedRandomSampler(samples_weight_fold, len(samples_weight_fold), replacement=True)

            train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, sampler=fold_sampler, worker_init_fn=lambda _: np.random.seed(SEED + fold_idx))
            val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED + fold_idx))

            model_cv_fold = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs_cv = model_cv_fold.fc.in_features; model_cv_fold.fc = nn.Linear(num_ftrs_cv, NUM_CLASSES); model_cv_fold = model_cv_fold.to(device)
            criterion_cv = nn.CrossEntropyLoss(weight=loss_weights)
            optimizer_cv = optim.Adam(model_cv_fold.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler_cv = ReduceLROnPlateau(optimizer_cv, mode='max', factor=0.1, patience=LR_SCHEDULER_PATIENCE)
            
            trained_model_for_fold, _, _, _, _, best_val_metric_for_fold = train_model_core(model_cv_fold, criterion_cv, optimizer_cv, scheduler_cv, train_loader_fold, val_loader_fold, len(current_train_paths_fold), len(current_val_paths_fold), NUM_EPOCHS, BEST_METRIC_FOR_SAVING, POSITIVE_CLASS_LABEL_FOR_PR_AUC, device, None, LEARNING_RATE, True)
            cv_fold_metrics_summary.append(best_val_metric_for_fold)

            fold_val_eval_results = evaluate_model(trained_model_for_fold, val_loader_fold, phase_name=f"CV Fold {fold_idx+1} Val Metrics")
            fold_val_labels = fold_val_eval_results.get("labels_np"); fold_val_probs = fold_val_eval_results.get("probs_class1_np")
            optimal_threshold_this_fold, _ = find_optimal_threshold(fold_val_labels, fold_val_probs, target_metric='f1_minority', minority_class_label=POSITIVE_CLASS_LABEL_FOR_PR_AUC)
            cv_fold_optimal_thresholds.append(optimal_threshold_this_fold)

        valid_cv_metrics = [m for m in cv_fold_metrics_summary if m is not None and not np.isnan(m)]
        if valid_cv_metrics:
            avg_cv_metric = np.mean(valid_cv_metrics); std_cv_metric = np.std(valid_cv_metrics)
            print(f"\n--- CV Perf Summary ({BEST_METRIC_FOR_SAVING}) ---")
            print(f"Scores per fold: {['{:.4f}'.format(m) for m in valid_cv_metrics]}")
            print(f"Avg {BEST_METRIC_FOR_SAVING} across {len(valid_cv_metrics)} folds: {avg_cv_metric:.4f} +/- {std_cv_metric:.4f}")
            valid_cv_thresholds = [t for t in cv_fold_optimal_thresholds if t is not None and not np.isnan(t)]
            if valid_cv_thresholds:
                avg_optimal_threshold_cv = np.mean(valid_cv_thresholds) # Store for external eval
                print(f"Optimal thresholds per fold: {['{:.4f}'.format(t) for t in valid_cv_thresholds]}")
                print(f"Avg optimal threshold from CV: {avg_optimal_threshold_cv:.4f}")
        else: print("No valid CV metrics collected.")

        print("\n--- Training Final Model on All Dev Data ---")
        if len(all_dev_image_paths) > 0:
            full_dev_dataset = CustomImageDataset(all_dev_image_paths, all_dev_labels, transform=data_transforms['train'])
            full_dev_sampler = None; class_counts_full_dev = np.bincount(all_dev_labels, minlength=NUM_CLASSES)
            if not np.any(class_counts_full_dev == 0):
                weight_per_class_full_dev = 1. / class_counts_full_dev; samples_weight_full_dev_list = [weight_per_class_full_dev[t] for t in all_dev_labels]
                samples_weight_full_dev = torch.from_numpy(np.array(samples_weight_full_dev_list)).double()
                full_dev_sampler = WeightedRandomSampler(samples_weight_full_dev, len(samples_weight_full_dev), replacement=True)
            full_dev_loader = DataLoader(full_dev_dataset, batch_size=BATCH_SIZE, sampler=full_dev_sampler, worker_init_fn=lambda _: np.random.seed(SEED -1))
            final_model_instance = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs_final = final_model_instance.fc.in_features; final_model_instance.fc = nn.Linear(num_ftrs_final, NUM_CLASSES); final_model_instance = final_model_instance.to(device)
            criterion_final = nn.CrossEntropyLoss(weight=loss_weights); optimizer_final = optim.Adam(final_model_instance.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            print(f"Starting final model training on {len(all_dev_image_paths)} images for {NUM_EPOCHS} epochs.")
            final_model_instance.train()
            for epoch in range(NUM_EPOCHS):
                running_loss = 0.0
                for inputs, labels in full_dev_loader:
                    inputs, labels = inputs.to(device), labels.to(device); optimizer_final.zero_grad()
                    outputs = final_model_instance(inputs); loss = criterion_final(outputs, labels)
                    loss.backward(); optimizer_final.step(); running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(all_dev_image_paths) if len(all_dev_image_paths) > 0 else 0
                print(f"Final Training Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
            torch.save(final_model_instance.state_dict(), MODEL_SAVE_PATH); print(f"Final model saved to {MODEL_SAVE_PATH}")
            final_trained_model = final_model_instance
        else: print("No dev data to train final model.")
    else: # Standard Run
        # ... (Standard run logic from previous script, ensure it sets final_trained_model and optimal_threshold_val_std)
        print("\n--- Starting Standard Single Train/Validation Run ---")
        train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train'])
        val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val'])
        train_targets = np.array(train_dataset.targets); class_counts_train_std = np.bincount(train_targets, minlength=NUM_CLASSES)
        std_sampler = None
        if not np.any(class_counts_train_std == 0):
            weight_per_class_std = 1. / class_counts_train_std; samples_weight_std = np.array([weight_per_class_std[t] for t in train_targets])
            samples_weight_std = torch.from_numpy(samples_weight_std).double(); std_sampler = WeightedRandomSampler(samples_weight_std, len(samples_weight_std), replacement=True)
        train_loader_std = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=std_sampler, worker_init_fn=lambda _: np.random.seed(SEED))
        val_loader_std = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))
        dataset_sizes_std = {'train': len(train_dataset), 'val': len(val_dataset)}
        model_std = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs_std = model_std.fc.in_features; model_std.fc = nn.Linear(num_ftrs_std, NUM_CLASSES); model_std = model_std.to(device)
        criterion_std = nn.CrossEntropyLoss(weight=loss_weights); optimizer_std = optim.Adam(model_std.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler_std = ReduceLROnPlateau(optimizer_std, mode='max', factor=0.1, patience=LR_SCHEDULER_PATIENCE)
        final_trained_model, train_l, val_l, train_met_hist, val_met_hist, _ = train_model_core(model_std, criterion_std, optimizer_std, scheduler_std, train_loader_std, val_loader_std, dataset_sizes_std['train'], dataset_sizes_std['val'], NUM_EPOCHS, BEST_METRIC_FOR_SAVING, POSITIVE_CLASS_LABEL_FOR_PR_AUC, device, MODEL_SAVE_PATH, LEARNING_RATE, False)
        plot_training_history(train_l, val_l, train_met_hist, val_met_hist, metric_name=BEST_METRIC_FOR_SAVING, suffix="_std_run")
        val_eval_results_default = evaluate_model(final_trained_model, val_loader_std, phase_name=f"StdVal DefaultTh")
        val_labels_std = val_eval_results_default.get("labels_np"); val_probs_std = val_eval_results_default.get("probs_class1_np")
        optimal_threshold_val_std, _ = find_optimal_threshold(val_labels_std, val_probs_std, target_metric='f1_minority', minority_class_label=POSITIVE_CLASS_LABEL_FOR_PR_AUC)
        evaluate_model(final_trained_model, val_loader_std, phase_name=f"StdVal OptimalTh", fixed_threshold=optimal_threshold_val_std)


    # --- External Validation ---
    if final_trained_model and external_val_dataloader:
        print(f"\n--- External Validation (Final Model from {MODEL_SAVE_PATH}) ---")
        model_ext = models.resnet18(weights=None); num_ftrs_ext = model_ext.fc.in_features; model_ext.fc = nn.Linear(num_ftrs_ext, NUM_CLASSES)
        if os.path.exists(MODEL_SAVE_PATH):
            model_ext.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device)); model_ext = model_ext.to(device)
            ext_eval_def = evaluate_model(model_ext, external_val_dataloader, phase_name=f"ExtVal DefaultTh")
            ext_lbl = ext_eval_def.get("labels_np"); ext_prb = ext_eval_def.get("probs_class1_np")
            opt_thresh_ext = 0.5
            if USE_CV and 'avg_optimal_threshold_cv' in locals() and not np.isnan(avg_optimal_threshold_cv): opt_thresh_ext = avg_optimal_threshold_cv; print(f"Using avg optimal_thresh from CV: {opt_thresh_ext:.4f}")
            elif not USE_CV and 'optimal_threshold_val_std' in locals(): opt_thresh_ext = optimal_threshold_val_std; print(f"Using optimal_thresh from std val: {opt_thresh_ext:.4f}")
            elif ext_lbl is not None and len(ext_lbl) > 0 : opt_thresh_ext, _ = find_optimal_threshold(ext_lbl, ext_prb, 'f1_minority', POSITIVE_CLASS_LABEL_FOR_PR_AUC); print(f"Using optimal_thresh tuned on ExtVal: {opt_thresh_ext:.4f} (for demo)")
            if opt_thresh_ext != 0.5 or (ext_lbl is not None and len(ext_lbl)>0): evaluate_model(model_ext, external_val_dataloader, phase_name=f"ExtVal OptimalTh", fixed_threshold=opt_thresh_ext)
        else: print(f"Model {MODEL_SAVE_PATH} not found for ext eval.")
    # ... (other final print statements) ...
    print("\nScript finished.")
    if plt.get_fignums(): # Check if any figures were created
        plt.show()