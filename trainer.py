import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_curve, auc # Added auc
)
from sklearn.utils import resample # For bootstrapping
import time
import copy
import random # For seeding

# --- Configuration ---
DATA_DIR = './'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME_TAG = "resnet18_rep" # To distinguish this run's saved files
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, f'best_model_hr_{MODEL_NAME_TAG}.pth')

NUM_CLASSES = 2
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 5e-4
LR_SCHEDULER_PATIENCE = 7
IMAGE_SIZE = (224, 224)
BEST_METRIC_FOR_SAVING = "pr_auc"  # Options: "balanced_accuracy", "pr_auc"
# Ensure class 1 (HR-) is treated as the positive class for PR-AUC if that's intended.
POSITIVE_CLASS_LABEL_FOR_PR_AUC = 1

# --- Reproducibility ---
SEED = 86
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # Potentially make CUDA operations deterministic (can slow down training)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed_value)
            # print(f"MPS seed set to {seed_value}") # Commented out for brevity from user log
        except AttributeError:
            print("torch.mps.manual_seed not available in this PyTorch version for MPS.")
set_seed(SEED)
# print(f"Global random seed set to {SEED}") # Commented out for brevity


# --- Bootstrap Configuration ---
N_BOOTSTRAP_SAMPLES = 1000 
CONFIDENCE_LEVEL = 0.95
ALPHA = (1 - CONFIDENCE_LEVEL) / 2.0


# --- Device Configuration ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    # print("Using MPS (Apple Silicon GPU)") # Commented out for brevity
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

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

# --- Load Datasets ---
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'val', 'external_val']
}

# --- Handle Class Imbalance for Training Set ---
train_targets = np.array(image_datasets['train'].targets)
class_counts_train = np.bincount(train_targets)
# print(f"Training set class counts (0=HR+, 1=HR-): {class_counts_train}") # Commented out for brevity

weight_per_class_sampler = 1. / class_counts_train
samples_weight = np.array([weight_per_class_sampler[t] for t in train_targets])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

loss_weights = torch.tensor([1.0, 2.5], dtype=torch.float32)
loss_weights = loss_weights.to(device)
# print(f"Using manual weights for loss function: {loss_weights}") # Commented out for brevity


dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=sampler, worker_init_fn=lambda _: np.random.seed(SEED)),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED)),
    'external_val': DataLoader(image_datasets['external_val'], batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'external_val']}
class_names = image_datasets['train'].classes 

# print(f"Class names from ImageFolder: {class_names}") # Commented out for brevity
# print(f"Dataset sizes: {dataset_sizes}") # Commented out for brevity
# print(f"Using BATCH_SIZE: {BATCH_SIZE}") # Commented out for brevity
# print(f"Optimizing for: {BEST_METRIC_FOR_SAVING} using class {POSITIVE_CLASS_LABEL_FOR_PR_AUC} as positive for PR-AUC.") # Commented out for brevity


# --- Model Definition ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)
# print("Using ResNet18 model.") # Commented out for brevity

# --- Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=LR_SCHEDULER_PATIENCE)

# --- Training Function (MODIFIED for custom metric) ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, best_metric_name=BEST_METRIC_FOR_SAVING):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric_score = 0.0 
    
    history_train_losses, history_val_losses = [], []
    history_train_metric, history_val_metric = [], []
    
    no_improvement_epochs_lr, min_lr_stop_patience = 0, 5

    metric_name_for_print = ""
    if best_metric_name == "balanced_accuracy":
        metric_name_for_print = "Balanced Acc"
    elif best_metric_name == "pr_auc":
        metric_name_for_print = "PR AUC"
    else:
        raise ValueError(f"Unsupported best_metric_name: {best_metric_name}")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n' + '-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            
            running_loss = 0.0
            all_labels_epoch, all_preds_epoch, all_probs_class_positive_epoch = [], [], []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
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
                if phase == 'val' or (phase == 'train' and best_metric_name == 'pr_auc'): 
                    probabilities = torch.softmax(outputs, dim=1)
                    # MINIMAL CHANGE HERE: Added .detach()
                    all_probs_class_positive_epoch.extend(probabilities[:, POSITIVE_CLASS_LABEL_FOR_PR_AUC].detach().cpu().numpy())
            
            epoch_loss = running_loss / dataset_sizes[phase]
            all_labels_epoch_np = np.array(all_labels_epoch)
            all_preds_epoch_np = np.array(all_preds_epoch)
            
            epoch_bal_acc = balanced_accuracy_score(all_labels_epoch_np, all_preds_epoch_np)
            epoch_pr_auc = 0.0
            if len(np.unique(all_labels_epoch_np)) > 1:
                all_probs_class_positive_epoch_np = np.array(all_probs_class_positive_epoch)
                if len(all_probs_class_positive_epoch_np) == len(all_labels_epoch_np):
                    try:
                        precision_p, recall_p, _ = precision_recall_curve(all_labels_epoch_np, all_probs_class_positive_epoch_np, pos_label=POSITIVE_CLASS_LABEL_FOR_PR_AUC)
                        epoch_pr_auc = auc(recall_p, precision_p)
                    except ValueError as e:
                        print(f"Warning: Could not calculate PR AUC for {phase} phase, epoch {epoch+1}: {e}. Setting to 0.")
                        epoch_pr_auc = 0.0
                else: 
                     print(f"Warning: Mismatch in length of labels and probabilities for PR AUC in {phase} phase, epoch {epoch+1}. PR AUC set to 0.")
                     epoch_pr_auc = 0.0
            else:
                if best_metric_name == 'pr_auc':
                    print(f"Warning: Only one class present in {phase} labels for epoch {epoch+1}. {metric_name_for_print} set to 0.")

            current_epoch_metric_value = 0.0
            if best_metric_name == "balanced_accuracy":
                current_epoch_metric_value = epoch_bal_acc
            elif best_metric_name == "pr_auc":
                current_epoch_metric_value = epoch_pr_auc
            
            if phase == 'train':
                history_train_losses.append(epoch_loss)
                history_train_metric.append(current_epoch_metric_value) 
                print(f'{phase} Loss: {epoch_loss:.4f} {metric_name_for_print}: {current_epoch_metric_value:.4f} (Bal Acc: {epoch_bal_acc:.4f})')
            else: 
                history_val_losses.append(epoch_loss)
                history_val_metric.append(current_epoch_metric_value) 
                
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(current_epoch_metric_value) 
                new_lr = optimizer.param_groups[0]['lr']

                if new_lr < old_lr:
                    print(f"Epoch {epoch+1}: Learning rate reduced from {old_lr} to {new_lr}.")
                    no_improvement_epochs_lr = 0
                else: 
                    no_improvement_epochs_lr += 1
                
                if current_epoch_metric_value > best_metric_score:
                    best_metric_score = current_epoch_metric_value
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"Best model saved to {MODEL_SAVE_PATH} with {metric_name_for_print}: {best_metric_score:.4f}")
                    no_improvement_epochs_lr = 0 
                print(f'{phase} Loss: {epoch_loss:.4f} {metric_name_for_print}: {current_epoch_metric_value:.4f} (Bal Acc: {epoch_bal_acc:.4f})')
        print()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr <= (LEARNING_RATE * 0.01 * 0.5) and no_improvement_epochs_lr >= min_lr_stop_patience:
             print(f"Early stopping: LR is low ({current_lr}), no improvement in validation {metric_name_for_print} for {no_improvement_epochs_lr} epochs.")
             break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val {metric_name_for_print}: {best_metric_score:4f}')
    model.load_state_dict(best_model_wts)
    return model, history_train_losses, history_val_losses, history_train_metric, history_val_metric, metric_name_for_print

# --- Threshold Tuning Function (unchanged) ---
def find_optimal_threshold(labels_np, probs_class1_np, target_metric='f1_minority', minority_class_label=1):
    optimal_threshold, best_metric_value = 0.5, -1
    unique_sorted_probs = np.sort(np.unique(probs_class1_np))
    if len(unique_sorted_probs) == 0: return 0.5
    candidate_thresholds = (unique_sorted_probs[:-1] + unique_sorted_probs[1:]) / 2.0 if len(unique_sorted_probs) > 1 else unique_sorted_probs
    candidate_thresholds = np.clip(np.sort(np.unique(np.append(candidate_thresholds, [0.001, 0.5, 0.999]))), 0.0, 1.0)
    for threshold_val in candidate_thresholds:
        preds_at_threshold = (probs_class1_np >= threshold_val).astype(int)
        if target_metric == 'f1_minority':
            _, _, f1, _ = precision_recall_fscore_support(labels_np, preds_at_threshold, average=None, labels=[0,1], zero_division=0)
            current_metric_value = f1[minority_class_label]
        elif target_metric == 'balanced_accuracy': current_metric_value = balanced_accuracy_score(labels_np, preds_at_threshold)
        else: raise ValueError(f"Unsupported target_metric: {target_metric}")
        if current_metric_value > best_metric_value:
            best_metric_value, optimal_threshold = current_metric_value, threshold_val
        elif current_metric_value == best_metric_value and abs(threshold_val - 0.5) < abs(optimal_threshold - 0.5):
            optimal_threshold = threshold_val
    print(f"Optimal threshold: {optimal_threshold:.4f} for '{target_metric}' (Value: {best_metric_value:.4f})")
    return optimal_threshold

# --- Evaluation Function (MODIFIED for PR-AUC and Confidence Intervals) ---
def calculate_metrics_bootstrap(y_true, y_pred, y_probs_class1):
    metrics = {}
    is_problematic_sample = len(np.unique(y_true)) < 2 
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    try: 
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred) if not is_problematic_sample else 0.0
    except ValueError: metrics['mcc'] = 0.0
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1], zero_division=0)
    metrics['precision_hr+'] = precision[0]; metrics['recall_hr+'] = recall[0]; metrics['f1_hr+'] = f1[0]
    metrics['precision_hr-'] = precision[1]; metrics['recall_hr-'] = recall[1]; metrics['f1_hr-'] = f1[1]
    metrics['specificity'] = recall[0] 
    
    try: 
        metrics['roc_auc'] = roc_auc_score(y_true, y_probs_class1) if not is_problematic_sample else 0.5
    except ValueError: metrics['roc_auc'] = 0.5

    try:
        if not is_problematic_sample:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs_class1, pos_label=POSITIVE_CLASS_LABEL_FOR_PR_AUC)
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        else:
            metrics['pr_auc'] = 0.0 
    except ValueError: 
        metrics['pr_auc'] = 0.0
        
    return metrics

def evaluate_model(model, dataloader, phase_name="Test", fixed_threshold=None):
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
        print("No data to evaluate.")
        empty_metrics_dict = {key: (0, (np.nan, np.nan)) for key in [ 
            "accuracy", "balanced_accuracy", "mcc", "specificity", "pr_auc", 
            "precision_hr+", "recall_hr+", "f1_hr+",
            "precision_hr-", "recall_hr-", "f1_hr-", "roc_auc"
        ]}
        empty_metrics_dict["labels_np"] = labels_np
        empty_metrics_dict["probs_class1_np"] = probs_class1_np
        return empty_metrics_dict

    current_threshold_for_print = f"{fixed_threshold:.4f}" if fixed_threshold is not None else "Default 0.5"
    preds_np = (probs_class1_np >= fixed_threshold).astype(int) if fixed_threshold is not None and POSITIVE_CLASS_LABEL_FOR_PR_AUC == 1 else \
               (probs_class1_np <= fixed_threshold).astype(int) if fixed_threshold is not None and POSITIVE_CLASS_LABEL_FOR_PR_AUC == 0 else \
               np.array(all_preds_default_thresh)


    print(f"\n--- Evaluation Metrics for {phase_name} (Threshold: {current_threshold_for_print}) ---")
    
    point_metrics = calculate_metrics_bootstrap(labels_np, preds_np, probs_class1_np)
    
    bootstrap_metrics_values = {key: [] for key in point_metrics.keys()} 
    n_samples = len(labels_np)
    if n_samples < 10: 
        print("Sample size too small for reliable bootstrapping.")
        metrics_with_ci = {key: (value, (np.nan, np.nan)) for key, value in point_metrics.items()}
    else:
        for i in range(N_BOOTSTRAP_SAMPLES):
            indices = resample(np.arange(n_samples), n_samples=n_samples, random_state=SEED+i) 
            labels_boot = labels_np[indices]
            probs_class1_boot = probs_class1_np[indices]
            
            if len(np.unique(labels_boot)) < 2:
                continue

            preds_boot = (probs_class1_boot >= fixed_threshold).astype(int) if fixed_threshold is not None and POSITIVE_CLASS_LABEL_FOR_PR_AUC == 1 else \
                         (probs_class1_boot <= fixed_threshold).astype(int) if fixed_threshold is not None and POSITIVE_CLASS_LABEL_FOR_PR_AUC == 0 else \
                         (np.array(all_preds_default_thresh)[indices]) 

            current_boot_metrics = calculate_metrics_bootstrap(labels_boot, preds_boot, probs_class1_boot)
            for key in bootstrap_metrics_values.keys():
                bootstrap_metrics_values[key].append(current_boot_metrics.get(key, np.nan)) 
        
        metrics_with_ci = {}
        for key, values in bootstrap_metrics_values.items():
            if not values: 
                 metrics_with_ci[key] = (point_metrics[key], (np.nan, np.nan))
                 continue
            valid_values = [v for v in values if not np.isnan(v)]
            if not valid_values:
                metrics_with_ci[key] = (point_metrics[key], (np.nan, np.nan))
                continue
            lower_bound = np.percentile(valid_values, ALPHA * 100)
            upper_bound = np.percentile(valid_values, (1 - ALPHA) * 100)
            metrics_with_ci[key] = (point_metrics[key], (lower_bound, upper_bound))

    print(f"Overall Accuracy: {metrics_with_ci.get('accuracy', (np.nan,))[0]:.4f} (95% CI: {metrics_with_ci.get('accuracy', (np.nan, (np.nan, np.nan)))[1][0]:.4f}-{metrics_with_ci.get('accuracy', (np.nan, (np.nan, np.nan)))[1][1]:.4f})")
    print(f"Balanced Accuracy: {metrics_with_ci.get('balanced_accuracy', (np.nan,))[0]:.4f} (95% CI: {metrics_with_ci.get('balanced_accuracy', (np.nan, (np.nan, np.nan)))[1][0]:.4f}-{metrics_with_ci.get('balanced_accuracy', (np.nan, (np.nan, np.nan)))[1][1]:.4f})")
    print(f"Specificity (HR+ Recall): {metrics_with_ci.get('specificity', (np.nan,))[0]:.4f} (95% CI: {metrics_with_ci.get('specificity', (np.nan, (np.nan, np.nan)))[1][0]:.4f}-{metrics_with_ci.get('specificity', (np.nan, (np.nan, np.nan)))[1][1]:.4f})")
    print(f"Matthews Correlation Coefficient (MCC): {metrics_with_ci.get('mcc', (np.nan,))[0]:.4f} (95% CI: {metrics_with_ci.get('mcc', (np.nan, (np.nan, np.nan)))[1][0]:.4f}-{metrics_with_ci.get('mcc', (np.nan, (np.nan, np.nan)))[1][1]:.4f})")
    
    print("\nClass-wise metrics (Value (95% CI Lower-Upper)):")
    print(f"  Class {class_names[0]} (HR+):")
    print(f"    Precision: {metrics_with_ci.get('precision_hr+',(np.nan,))[0]:.4f} ({metrics_with_ci.get('precision_hr+',(np.nan,(np.nan,np.nan)))[1][0]:.4f}-{metrics_with_ci.get('precision_hr+',(np.nan,(np.nan,np.nan)))[1][1]:.4f})")
    print(f"    Recall (Specificity): {metrics_with_ci.get('recall_hr+',(np.nan,))[0]:.4f} ({metrics_with_ci.get('recall_hr+',(np.nan,(np.nan,np.nan)))[1][0]:.4f}-{metrics_with_ci.get('recall_hr+',(np.nan,(np.nan,np.nan)))[1][1]:.4f})")
    print(f"    F1-score: {metrics_with_ci.get('f1_hr+',(np.nan,))[0]:.4f} ({metrics_with_ci.get('f1_hr+',(np.nan,(np.nan,np.nan)))[1][0]:.4f}-{metrics_with_ci.get('f1_hr+',(np.nan,(np.nan,np.nan)))[1][1]:.4f})")
    print(f"  Class {class_names[1]} (HR-):")
    print(f"    Precision: {metrics_with_ci.get('precision_hr-',(np.nan,))[0]:.4f} ({metrics_with_ci.get('precision_hr-',(np.nan,(np.nan,np.nan)))[1][0]:.4f}-{metrics_with_ci.get('precision_hr-',(np.nan,(np.nan,np.nan)))[1][1]:.4f})")
    print(f"    Recall (Sensitivity): {metrics_with_ci.get('recall_hr-',(np.nan,))[0]:.4f} ({metrics_with_ci.get('recall_hr-',(np.nan,(np.nan,np.nan)))[1][0]:.4f}-{metrics_with_ci.get('recall_hr-',(np.nan,(np.nan,np.nan)))[1][1]:.4f})")
    print(f"    F1-score: {metrics_with_ci.get('f1_hr-',(np.nan,))[0]:.4f} ({metrics_with_ci.get('f1_hr-',(np.nan,(np.nan,np.nan)))[1][0]:.4f}-{metrics_with_ci.get('f1_hr-',(np.nan,(np.nan,np.nan)))[1][1]:.4f})")

    if metrics_with_ci.get('roc_auc', (None,))[0] is not None:
         print(f"\nROC AUC (for HR- as positive class): {metrics_with_ci['roc_auc'][0]:.4f} (95% CI: {metrics_with_ci['roc_auc'][1][0]:.4f}-{metrics_with_ci['roc_auc'][1][1]:.4f})")
    else: print("\nROC AUC could not be calculated or not available.")
    
    if metrics_with_ci.get('pr_auc', (None,))[0] is not None: 
         print(f"PR AUC (for HR- as positive class): {metrics_with_ci['pr_auc'][0]:.4f} (95% CI: {metrics_with_ci['pr_auc'][1][0]:.4f}-{metrics_with_ci['pr_auc'][1][1]:.4f})")
    else: print("PR AUC could not be calculated or not available.")

    cm = confusion_matrix(labels_np, preds_np, labels=[0,1])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"Pred {class_names[0]} (HR+)", f"Pred {class_names[1]} (HR-)"],
                yticklabels=[f"True {class_names[0]} (HR+)", f"True {class_names[1]} (HR-)"])
    cm_title_thresh = "Def" if fixed_threshold is None else f"{fixed_threshold:.2f}"
    plt.title(f'Confusion Matrix - {phase_name} (Thresh: {cm_title_thresh})')
    plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"cm_{MODEL_NAME_TAG}_{phase_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{cm_title_thresh.replace('.', 'p')}.png"))
    plt.show()
    
    metrics_with_ci["labels_np"] = labels_np
    metrics_with_ci["probs_class1_np"] = probs_class1_np
    return metrics_with_ci


# --- Plotting training history ---
def plot_training_history(train_losses, val_losses, train_metric_scores, val_metric_scores, metric_name="Metric"): 
    epochs_len = len(train_losses)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_len + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.legend(loc='upper right'); plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs_len + 1), train_metric_scores, label=f'Training {metric_name}')
    plt.plot(range(1, len(val_metric_scores) + 1), val_metric_scores, label=f'Validation {metric_name}')
    plt.legend(loc='lower right'); plt.title(f'Training and Validation {metric_name}'); plt.xlabel('Epoch'); plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"training_history_{MODEL_NAME_TAG}_{metric_name.replace(' ', '_').lower()}.png"))
    plt.show()

# --- Main Execution (MODIFIED for new summary print and plot call) ---
if __name__ == '__main__':
    # Simplified print statements from user log
    print(f"MPS seed set to {SEED}")
    print(f"Global random seed set to {SEED}")
    print(f"Using {device}")
    print(f"Training set class counts (0=HR+, 1=HR-): {class_counts_train}")
    print(f"Using manual weights for loss function: {loss_weights.cpu().numpy()}") # .cpu().numpy() for print
    print(f"Class names from ImageFolder: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Using BATCH_SIZE: {BATCH_SIZE}")
    print(f"Optimizing for: {BEST_METRIC_FOR_SAVING} using class {POSITIVE_CLASS_LABEL_FOR_PR_AUC} as positive for PR-AUC.")
    print("Using ResNet18 model.")


    for split in ['train', 'val', 'external_val']:
        if not os.path.exists(os.path.join(DATA_DIR, split)):
            print(f"WARNING: Base directory {os.path.join(DATA_DIR, split)} does not exist.")
            continue
        # Assuming class_names from ImageFolder are actual directory names like '1', '2'
        # If image_datasets[split].classes gives ['HR_pos_folder', 'HR_neg_folder'], use that
        # For now, assuming class_names like ['1', '2'] which are often default
        for label_dir_name in image_datasets[split].classes: 
            path = os.path.join(DATA_DIR, split, label_dir_name)
            if not os.path.exists(path) or (os.path.isdir(path) and not os.listdir(path)):
                print(f"WARNING: Directory {path} is empty or does not exist.")


    print(f"Starting training with {MODEL_NAME_TAG}, optimizing for {BEST_METRIC_FOR_SAVING}...")
    model_ft, train_l, val_l, train_met_hist, val_met_hist, trained_metric_name = train_model(
        model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, best_metric_name=BEST_METRIC_FOR_SAVING
    )
    
    print(f"\nPlotting training history for {MODEL_NAME_TAG} ({trained_metric_name})...")
    plot_training_history(train_l, val_l, train_met_hist, val_met_hist, metric_name=trained_metric_name)

    print(f"\n--- Validation Set Evaluation ({MODEL_NAME_TAG}) ---")
    val_eval_results_default = evaluate_model(model_ft, dataloaders['val'], phase_name=f"Validation Set ({MODEL_NAME_TAG})")
    val_labels = val_eval_results_default.get("labels_np") 
    val_probs_hr_neg = val_eval_results_default.get("probs_class1_np") 
    optimal_threshold_val = 0.5 

    val_eval_results_optimal = None 
    if val_labels is not None and len(val_labels) > 0 and \
       val_probs_hr_neg is not None and len(val_probs_hr_neg) > 0 and \
       len(np.unique(val_labels)) > 1:
        print(f"\nTuning threshold on Validation Set probabilities ({MODEL_NAME_TAG})...")
        optimal_threshold_val = find_optimal_threshold(val_labels, val_probs_hr_neg, target_metric='f1_minority', minority_class_label=POSITIVE_CLASS_LABEL_FOR_PR_AUC)
        print(f"\nRe-evaluating Validation Set with optimal threshold: {optimal_threshold_val:.4f} ({MODEL_NAME_TAG})")
        val_eval_results_optimal = evaluate_model(model_ft, dataloaders['val'], phase_name=f"Validation Set OptimalTh ({MODEL_NAME_TAG})", fixed_threshold=optimal_threshold_val)
    else:
        print("Not enough data or only one class in validation set to tune threshold. Using default 0.5 for optimal, or copying default results.")
        if val_eval_results_default: val_eval_results_optimal = val_eval_results_default


    print(f"\n--- External Validation Set Evaluation ({MODEL_NAME_TAG}) ---")
    best_model_instance = models.resnet18(weights=None) 
    num_ftrs_best = best_model_instance.fc.in_features
    best_model_instance.fc = nn.Linear(num_ftrs_best, NUM_CLASSES)
    
    ext_val_results_default = None
    ext_val_results_optimal = None

    print(f"Loading best {MODEL_NAME_TAG} model from: {MODEL_SAVE_PATH}")
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: Model file not found at {MODEL_SAVE_PATH}.")
    else:
        try:
            best_model_instance.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            best_model_instance = best_model_instance.to(device)
            
            print(f"\nEvaluating External Val Set with default threshold (0.5) ({MODEL_NAME_TAG})...")
            ext_val_results_default = evaluate_model(best_model_instance, dataloaders['external_val'], phase_name=f"External Val Set ({MODEL_NAME_TAG})")
            
            if ext_val_results_default and ext_val_results_default.get("labels_np") is not None and \
               len(ext_val_results_default.get("labels_np")) > 0 : 
                print(f"\nEvaluating External Val Set with optimal threshold from Val set ({optimal_threshold_val:.4f}) ({MODEL_NAME_TAG})...")
                ext_val_results_optimal = evaluate_model(best_model_instance, dataloaders['external_val'], phase_name=f"External Val Set OptimalTh ({MODEL_NAME_TAG})", fixed_threshold=optimal_threshold_val)
            else: 
                print("Default threshold evaluation failed or produced no results for external set, skipping optimal threshold evaluation.")

        except Exception as e:
            print(f"Error loading model or evaluating on external set: {e}")

    print(f"\n\n--- FINAL METRICS SUMMARY ({MODEL_NAME_TAG}) ---")
    
    def print_metrics_summary_ci(phase_results, phase_name, threshold_name, threshold_val_print=""):
        if phase_results and phase_results.get('balanced_accuracy') and not (isinstance(phase_results['balanced_accuracy'][0], float) and np.isnan(phase_results['balanced_accuracy'][0])): 
            print(f"{phase_name} - {threshold_name}{threshold_val_print}:")
            metric_order = [
                "accuracy", "balanced_accuracy", "specificity", "mcc", "roc_auc", "pr_auc",
                "precision_hr+", "recall_hr+", "f1_hr+",
                "precision_hr-", "recall_hr-", "f1_hr-"
            ]
            printed_keys = set()
            for metric_key in metric_order:
                if metric_key in phase_results:
                    metric_data = phase_results[metric_key]
                    if metric_key in ["labels_np", "probs_class1_np"]: continue
                    if isinstance(metric_data, tuple) and len(metric_data) == 2 and \
                       isinstance(metric_data[1], tuple) and len(metric_data[1]) == 2:
                        val, (ci_low, ci_high) = metric_data
                        readable_key = metric_key.replace('_hr+', ' (HR+)').replace('_hr-', ' (HR-)').replace('_', ' ').capitalize()
                        if metric_key == 'recall_hr+': readable_key = 'Specificity (Recall HR+)'
                        elif metric_key == 'recall_hr-': readable_key = 'Sensitivity (Recall HR-)'
                        elif metric_key == 'pr_auc': readable_key = 'PR AUC'
                        elif metric_key == 'roc_auc': readable_key = 'ROC AUC'
                        
                        val_str = f"{val:.4f}" if val is not None and not np.isnan(val) else "N/A"
                        ci_low_str = f"{ci_low:.4f}" if ci_low is not None and not np.isnan(ci_low) else "N/A"
                        ci_high_str = f"{ci_high:.4f}" if ci_high is not None and not np.isnan(ci_high) else "N/A"
                        print(f"  {readable_key:<30}: {val_str} (95% CI: {ci_low_str}-{ci_high_str})")
                        printed_keys.add(metric_key)
            
            for metric_key, metric_data in phase_results.items():
                if metric_key in printed_keys or metric_key in ["labels_np", "probs_class1_np"]:
                    continue
                if isinstance(metric_data, tuple) and len(metric_data) == 2 and \
                   isinstance(metric_data[1], tuple) and len(metric_data[1]) == 2:
                    val, (ci_low, ci_high) = metric_data
                    val_str = f"{val:.4f}" if val is not None and not np.isnan(val) else "N/A"
                    ci_low_str = f"{ci_low:.4f}" if ci_low is not None and not np.isnan(ci_low) else "N/A"
                    ci_high_str = f"{ci_high:.4f}" if ci_high is not None and not np.isnan(ci_high) else "N/A"
                    print(f"  {metric_key.replace('_',' ').capitalize():<30}: {val_str} (95% CI: {ci_low_str}-{ci_high_str})")


            print("-" * 30)
        else:
            print(f"{phase_name} - {threshold_name}{threshold_val_print}: Metrics not available or not computed properly.")

    print(f"\n-- Validation Set ({MODEL_NAME_TAG}) --")
    print_metrics_summary_ci(val_eval_results_default, "Validation", "Default Thresh")
    if val_eval_results_optimal:
        print_metrics_summary_ci(val_eval_results_optimal, "Validation", "Optimal Thresh", f" ({optimal_threshold_val:.2f})")

    print(f"\n-- External Validation Set ({MODEL_NAME_TAG}) --")
    print_metrics_summary_ci(ext_val_results_default, "External Val", "Default Thresh")
    if ext_val_results_optimal:
        print_metrics_summary_ci(ext_val_results_optimal, "External Val", "Optimal Thresh", f" ({optimal_threshold_val:.2f})")

    if not os.path.exists(MODEL_SAVE_PATH) and ext_val_results_default is None: 
        print(f"\nExternal validation could not be performed for {MODEL_NAME_TAG} because the model file was not found or evaluation failed.")
