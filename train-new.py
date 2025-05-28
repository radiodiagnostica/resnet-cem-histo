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
    confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_curve
)
from sklearn.utils import resample # For bootstrapping
import time
import copy
import random # For seeding

# --- Configuration ---
DATA_DIR = './'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME_TAG = "resnet50_rep" # To distinguish this run's saved files
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, f'best_model_hr_{MODEL_NAME_TAG}.pth')

NUM_CLASSES = 2
BATCH_SIZE = 4 # Keep small for ResNet50
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
LR_SCHEDULER_PATIENCE = 7
IMAGE_SIZE = (224, 224)

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
            print(f"MPS seed set to {seed_value}")
        except AttributeError:
            print("torch.mps.manual_seed not available in this PyTorch version for MPS.")
set_seed(SEED)
print(f"Global random seed set to {SEED}")


# --- Bootstrap Configuration ---
N_BOOTSTRAP_SAMPLES = 1000 # Number of bootstrap iterations, 1000 is common. Reduce for speed if needed.
CONFIDENCE_LEVEL = 0.95
ALPHA = (1 - CONFIDENCE_LEVEL) / 2.0


# --- Device Configuration ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
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
print(f"Training set class counts (0=HR+, 1=HR-): {class_counts_train}")

weight_per_class_sampler = 1. / class_counts_train
samples_weight = np.array([weight_per_class_sampler[t] for t in train_targets])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

loss_weights = torch.tensor([1.0, 2.5], dtype=torch.float32)
loss_weights = loss_weights.to(device)
print(f"Using manual weights for loss function: {loss_weights}")


dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=sampler, worker_init_fn=lambda _: np.random.seed(SEED)), # Add worker_init_fn for Dataloader reproducibility
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED)),
    'external_val': DataLoader(image_datasets['external_val'], batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'external_val']}
class_names = image_datasets['train'].classes # Should be ['1', '2'] -> HR+ (mapped to 0), HR- (mapped to 1)

print(f"Class names from ImageFolder: {class_names}")
print(f"Dataset sizes: {dataset_sizes}")
print(f"Using BATCH_SIZE: {BATCH_SIZE}")


# --- Model Definition ---
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)
print("Using ResNet50 model.")

# --- Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=LR_SCHEDULER_PATIENCE)

# --- Training Function (largely unchanged) ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_bal_acc = 0.0
    train_losses, val_losses, train_bal_accs, val_bal_accs = [], [], [], []
    no_improvement_epochs_lr, min_lr_stop_patience = 0, 5

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n' + '-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, all_preds_loop, all_labels_loop = 0.0, [], []
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
                all_preds_loop.extend(preds.cpu().numpy())
                all_labels_loop.extend(labels.cpu().numpy())
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_bal_acc = balanced_accuracy_score(all_labels_loop, all_preds_loop)
            
            if phase == 'train':
                train_losses.append(epoch_loss); train_bal_accs.append(epoch_bal_acc)
            else:
                val_losses.append(epoch_loss); val_bal_accs.append(epoch_bal_acc)
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(epoch_bal_acc)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < old_lr:
                    print(f"Epoch {epoch+1}: Learning rate reduced from {old_lr} to {new_lr}.")
                    no_improvement_epochs_lr = 0
                else: no_improvement_epochs_lr += 1
                if epoch_bal_acc > best_bal_acc:
                    best_bal_acc = epoch_bal_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"Best model saved to {MODEL_SAVE_PATH} with balanced accuracy: {best_bal_acc:.4f}")
                    no_improvement_epochs_lr = 0
            print(f'{phase} Loss: {epoch_loss:.4f} Balanced Acc: {epoch_bal_acc:.4f}')
        print()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr <= (LEARNING_RATE * 0.01 * 0.5) and no_improvement_epochs_lr >= min_lr_stop_patience:
             print(f"Early stopping: LR is low ({current_lr}), no improvement for {no_improvement_epochs_lr} epochs.")
             break
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Balanced Acc: {best_bal_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_bal_accs, val_bal_accs

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

# --- Evaluation Function (MODIFIED for Specificity and Confidence Intervals) ---
def calculate_metrics_bootstrap(y_true, y_pred, y_probs_class1):
    # y_probs_class1 is for ROC AUC, y_pred for others
    metrics = {}
    # Handle cases where a class might be missing in a bootstrap sample
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2: # Needed for some metrics like MCC, ROC AUC
        # Fallback for problematic bootstrap samples (e.g. all one class)
        # This can happen with small datasets. For larger datasets, less likely.
        # For metrics like ROC AUC, it might return error.
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        try: metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        except ValueError: metrics['mcc'] = 0 # Or np.nan
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1], zero_division=0)
        metrics['precision_hr+'] = precision[0]; metrics['recall_hr+'] = recall[0]; metrics['f1_hr+'] = f1[0]
        metrics['precision_hr-'] = precision[1]; metrics['recall_hr-'] = recall[1]; metrics['f1_hr-'] = f1[1]
        metrics['specificity'] = recall[0] # Recall of HR+ (class 0) is specificity for HR-
        
        try: metrics['roc_auc'] = roc_auc_score(y_true, y_probs_class1) if len(np.unique(y_true)) > 1 else 0.5 # Or np.nan
        except ValueError: metrics['roc_auc'] = 0.5 # Fallback if only one class in y_true for probs
        return metrics

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1], zero_division=0)
    metrics['precision_hr+'] = precision[0]
    metrics['recall_hr+'] = recall[0] # This is Specificity for HR-
    metrics['f1_hr+'] = f1[0]
    metrics['precision_hr-'] = precision[1]
    metrics['recall_hr-'] = recall[1] # This is Sensitivity for HR-
    metrics['f1_hr-'] = f1[1]
    metrics['specificity'] = recall[0] # Specificity: TN / (TN+FP) = Recall of Negative Class (HR+)

    metrics['roc_auc'] = roc_auc_score(y_true, y_probs_class1) if len(np.unique(y_true)) > 1 else 0.5 # Ensure more than one class for AUC
    return metrics

def evaluate_model(model, dataloader, phase_name="Test", fixed_threshold=None):
    model.eval()
    all_preds_default_thresh, all_labels, all_probs_class1 = [], [], []
    with torch.no_grad():
        for inputs, labels_batch in dataloader: # Renamed labels to labels_batch
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = model(inputs)
            _, preds_default = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            all_probs_class1.extend(probabilities[:, 1].cpu().numpy())
            all_preds_default_thresh.extend(preds_default.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    labels_np = np.array(all_labels)
    probs_class1_np = np.array(all_probs_class1)

    if len(labels_np) == 0:
        print("No data to evaluate.")
        # Return structure with Nones or defaults
        empty_metrics_dict = {key: (0, (0,0)) for key in [
            "accuracy", "balanced_accuracy", "mcc", "specificity",
            "precision_hr+", "recall_hr+", "f1_hr+",
            "precision_hr-", "recall_hr-", "f1_hr-", "roc_auc"
        ]}
        empty_metrics_dict["labels_np"] = labels_np
        empty_metrics_dict["probs_class1_np"] = probs_class1_np
        return empty_metrics_dict


    current_threshold_for_print = f"{fixed_threshold:.4f}" if fixed_threshold is not None else "Default 0.5"
    preds_np = (probs_class1_np >= fixed_threshold).astype(int) if fixed_threshold is not None else np.array(all_preds_default_thresh)

    print(f"\n--- Evaluation Metrics for {phase_name} (Threshold: {current_threshold_for_print}) ---")
    
    # Calculate point estimates
    point_metrics = calculate_metrics_bootstrap(labels_np, preds_np, probs_class1_np)
    
    # Bootstrap for CIs
    bootstrap_metrics_values = {key: [] for key in point_metrics.keys()}
    n_samples = len(labels_np)
    if n_samples < 10: # Bootstrapping not reliable for very small samples
        print("Sample size too small for reliable bootstrapping.")
        metrics_with_ci = {key: (value, (np.nan, np.nan)) for key, value in point_metrics.items()}
    else:
        for _ in range(N_BOOTSTRAP_SAMPLES):
            indices = resample(np.arange(n_samples), n_samples=n_samples, random_state=SEED+_) # Vary seed for bootstrap
            # Important: resample labels_np and probs_class1_np using the same indices
            # Then, derive preds_np_boot from probs_class1_np_boot and the fixed_threshold
            labels_boot = labels_np[indices]
            probs_class1_boot = probs_class1_np[indices]
            preds_boot = (probs_class1_boot >= fixed_threshold).astype(int) if fixed_threshold is not None else (np.array(all_preds_default_thresh)[indices])
            
            # Skip bootstrap sample if it results in only one class, as some metrics will fail.
            if len(np.unique(labels_boot)) < 2:
                # print(f"Skipping bootstrap sample due to single class in y_true_boot for phase {phase_name}")
                continue # Or handle by assigning NaN or specific value

            current_boot_metrics = calculate_metrics_bootstrap(labels_boot, preds_boot, probs_class1_boot)
            for key in bootstrap_metrics_values.keys():
                bootstrap_metrics_values[key].append(current_boot_metrics[key])
        
        metrics_with_ci = {}
        for key, values in bootstrap_metrics_values.items():
            if not values: # If all bootstrap samples were skipped
                 metrics_with_ci[key] = (point_metrics[key], (np.nan, np.nan))
                 continue
            lower_bound = np.percentile(values, ALPHA * 100)
            upper_bound = np.percentile(values, (1 - ALPHA) * 100)
            metrics_with_ci[key] = (point_metrics[key], (lower_bound, upper_bound))

    # Print metrics with CIs
    print(f"Overall Accuracy: {metrics_with_ci['accuracy'][0]:.4f} (95% CI: {metrics_with_ci['accuracy'][1][0]:.4f}-{metrics_with_ci['accuracy'][1][1]:.4f})")
    print(f"Balanced Accuracy: {metrics_with_ci['balanced_accuracy'][0]:.4f} (95% CI: {metrics_with_ci['balanced_accuracy'][1][0]:.4f}-{metrics_with_ci['balanced_accuracy'][1][1]:.4f})")
    print(f"Specificity (HR+ Recall): {metrics_with_ci['specificity'][0]:.4f} (95% CI: {metrics_with_ci['specificity'][1][0]:.4f}-{metrics_with_ci['specificity'][1][1]:.4f})")
    print(f"Matthews Correlation Coefficient (MCC): {metrics_with_ci['mcc'][0]:.4f} (95% CI: {metrics_with_ci['mcc'][1][0]:.4f}-{metrics_with_ci['mcc'][1][1]:.4f})")
    
    print("\nClass-wise metrics (Value (95% CI Lower-Upper)):")
    # HR+ (Class 0, traditionally negative if HR- is positive)
    print(f"  Class {class_names[0]} (HR+):")
    print(f"    Precision: {metrics_with_ci['precision_hr+'][0]:.4f} ({metrics_with_ci['precision_hr+'][1][0]:.4f}-{metrics_with_ci['precision_hr+'][1][1]:.4f})")
    print(f"    Recall (Specificity): {metrics_with_ci['recall_hr+'][0]:.4f} ({metrics_with_ci['recall_hr+'][1][0]:.4f}-{metrics_with_ci['recall_hr+'][1][1]:.4f})")
    print(f"    F1-score: {metrics_with_ci['f1_hr+'][0]:.4f} ({metrics_with_ci['f1_hr+'][1][0]:.4f}-{metrics_with_ci['f1_hr+'][1][1]:.4f})")
    # HR- (Class 1, traditionally positive)
    print(f"  Class {class_names[1]} (HR-):")
    print(f"    Precision: {metrics_with_ci['precision_hr-'][0]:.4f} ({metrics_with_ci['precision_hr-'][1][0]:.4f}-{metrics_with_ci['precision_hr-'][1][1]:.4f})")
    print(f"    Recall (Sensitivity): {metrics_with_ci['recall_hr-'][0]:.4f} ({metrics_with_ci['recall_hr-'][1][0]:.4f}-{metrics_with_ci['recall_hr-'][1][1]:.4f})")
    print(f"    F1-score: {metrics_with_ci['f1_hr-'][0]:.4f} ({metrics_with_ci['f1_hr-'][1][0]:.4f}-{metrics_with_ci['f1_hr-'][1][1]:.4f})")

    if metrics_with_ci['roc_auc'][0] is not None: # ROC AUC uses raw probabilities
         print(f"\nROC AUC (for HR- as positive class): {metrics_with_ci['roc_auc'][0]:.4f} (95% CI: {metrics_with_ci['roc_auc'][1][0]:.4f}-{metrics_with_ci['roc_auc'][1][1]:.4f})")
    else: print("\nROC AUC could not be calculated.")

    # Confusion Matrix (no CI for this plot)
    cm = confusion_matrix(labels_np, preds_np, labels=[0,1])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"Pred {class_names[0]} (HR+)", f"Pred {class_names[1]} (HR-)"],
                yticklabels=[f"True {class_names[0]} (HR+)", f"True {class_names[1]} (HR-)"])
    cm_title_thresh = "Def" if fixed_threshold is None else f"{fixed_threshold:.2f}"
    plt.title(f'Confusion Matrix - {phase_name} (Thresh: {cm_title_thresh})')
    plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"cm_{MODEL_NAME_TAG}_{phase_name.lower().replace(' ', '_')}_{cm_title_thresh.replace('.', 'p')}.png"))
    plt.show()
    
    # Return metrics_with_ci for summary, and raw probs/labels for potential external use
    metrics_with_ci["labels_np"] = labels_np
    metrics_with_ci["probs_class1_np"] = probs_class1_np
    return metrics_with_ci


# --- Plotting training history (unchanged, but uses MODEL_NAME_TAG for filename) ---
def plot_training_history(train_losses, val_losses, train_accs, val_accs, metric_name="Balanced Accuracy"):
    epochs_len = len(train_losses)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_len + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.legend(loc='upper right'); plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs_len + 1), train_accs, label=f'Training {metric_name}')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label=f'Validation {metric_name}')
    plt.legend(loc='lower right'); plt.title(f'Training and Validation {metric_name}'); plt.xlabel('Epoch'); plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"training_history_{MODEL_NAME_TAG}.png"))
    plt.show()

# --- Main Execution (MODIFIED for new summary print) ---
if __name__ == '__main__':
    for split in ['train', 'val', 'external_val']:
        for label_dir_name in class_names:
            path = os.path.join(DATA_DIR, split, label_dir_name) # Use actual folder names '1', '2'
            if not os.path.exists(path) or (os.path.isdir(path) and not os.listdir(path)):
                print(f"WARNING: Directory {path} is empty or does not exist.")

    print(f"Starting training with {MODEL_NAME_TAG}...")
    model_ft, train_l, val_l, train_ba, val_ba = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
    
    print(f"\nPlotting training history for {MODEL_NAME_TAG}...")
    plot_training_history(train_l, val_l, train_ba, val_ba, metric_name="Balanced Accuracy")

    print(f"\n--- Validation Set Evaluation ({MODEL_NAME_TAG}) ---")
    val_eval_results_default = evaluate_model(model_ft, dataloaders['val'], phase_name=f"Validation Set ({MODEL_NAME_TAG})")
    val_labels = val_eval_results_default["labels_np"]
    val_probs_hr_neg = val_eval_results_default["probs_class1_np"]
    optimal_threshold_val = 0.5

    val_eval_results_optimal = None # Initialize
    if val_labels is not None and len(val_labels) > 0 and val_probs_hr_neg is not None and len(val_probs_hr_neg) > 0 and len(np.unique(val_labels)) > 1:
        print(f"\nTuning threshold on Validation Set probabilities ({MODEL_NAME_TAG})...")
        optimal_threshold_val = find_optimal_threshold(val_labels, val_probs_hr_neg, target_metric='f1_minority', minority_class_label=1)
        print(f"\nRe-evaluating Validation Set with optimal threshold: {optimal_threshold_val:.4f} ({MODEL_NAME_TAG})")
        val_eval_results_optimal = evaluate_model(model_ft, dataloaders['val'], phase_name=f"Validation Set OptimalTh ({MODEL_NAME_TAG})", fixed_threshold=optimal_threshold_val)
    else:
        print("Not enough data or only one class in validation set to tune threshold. Using default 0.5.")
        if val_eval_results_default: val_eval_results_optimal = val_eval_results_default


    print(f"\n--- External Validation Set Evaluation ({MODEL_NAME_TAG}) ---")
    best_model_instance = models.resnet50(weights=None) # Load ResNet50 structure
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
            
            if ext_val_results_default and ext_val_results_default.get("labels_np") is not None and len(ext_val_results_default.get("labels_np")) > 0 : # Check if eval was successful
                print(f"\nEvaluating External Val Set with optimal threshold ({optimal_threshold_val:.4f}) ({MODEL_NAME_TAG})...")
                ext_val_results_optimal = evaluate_model(best_model_instance, dataloaders['external_val'], phase_name=f"External Val Set OptimalTh ({MODEL_NAME_TAG})", fixed_threshold=optimal_threshold_val)
            else: # If default eval failed (e.g. no data), optimal also won't work
                print("Default threshold evaluation failed or produced no results for external set, skipping optimal threshold evaluation.")

        except Exception as e:
            print(f"Error loading model or evaluating on external set: {e}")

    print(f"\n\n--- FINAL METRICS SUMMARY ({MODEL_NAME_TAG}) ---")
    
    def print_metrics_summary_ci(phase_results, phase_name, threshold_name, threshold_val_print=""):
        # Check if phase_results is not None and contains the 'balanced_accuracy' key with valid data
        if phase_results and phase_results.get('balanced_accuracy') and phase_results['balanced_accuracy'][0] != 0:
            print(f"{phase_name} - {threshold_name}{threshold_val_print}:")
            for metric_key, metric_data in phase_results.items(): # Changed iteration variable
                # Skip non-metric keys
                if metric_key in ["labels_np", "probs_class1_np"]:
                    continue

                # Ensure metric_data is in the expected format (value, (ci_low, ci_high))
                if isinstance(metric_data, tuple) and len(metric_data) == 2 and isinstance(metric_data[1], tuple) and len(metric_data[1]) == 2:
                    val, (ci_low, ci_high) = metric_data
                else:
                    # This case shouldn't happen if evaluate_model structures output correctly, but good for safety
                    print(f"  Skipping {metric_key}: unexpected data format.")
                    continue
                
                readable_key = metric_key.replace('_hr+', ' (HR+)').replace('_hr-', ' (HR-)').replace('_', ' ').capitalize()
                if metric_key == 'recall_hr+': readable_key = 'Specificity (Recall HR+)'
                if metric_key == 'recall_hr-': readable_key = 'Sensitivity (Recall HR-)'

                # Handle potential None or NaN for CI values if bootstrap failed for a specific metric
                ci_low_str = f"{ci_low:.4f}" if ci_low is not None and not np.isnan(ci_low) else "N/A"
                ci_high_str = f"{ci_high:.4f}" if ci_high is not None and not np.isnan(ci_high) else "N/A"

                print(f"  {readable_key:<30}: {val:.4f} (95% CI: {ci_low_str}-{ci_high_str})")
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
        print(f"\nExternal validation could not be performed for {MODEL_NAME_TAG} because the model file was not found.")
