#### Training Script

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    images.append((os.path.join(class_dir, img_name), self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image.to(torch.float32), label

# Data Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def calculate_metrics(y_true, y_pred, y_scores, n_bootstrap=1000, confidence_level=0.95):
    # Suppress the UndefinedMetricWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        
        metrics = {
            'accuracy': calculate_ci_and_pvalue(y_true, y_pred, accuracy_score, n_bootstrap, confidence_level),
            'precision': calculate_ci_and_pvalue(y_true, y_pred, lambda yt, yp: precision_score(yt, yp, average='weighted', zero_division=0), n_bootstrap, confidence_level),
            'recall': calculate_ci_and_pvalue(y_true, y_pred, lambda yt, yp: recall_score(yt, yp, average='weighted', zero_division=0), n_bootstrap, confidence_level),
            'f1': calculate_ci_and_pvalue(y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0), n_bootstrap, confidence_level),
            'mcc': calculate_ci_and_pvalue(y_true, y_pred, matthews_corrcoef, n_bootstrap, confidence_level),
            'balanced_acc': calculate_ci_and_pvalue(y_true, y_pred, balanced_accuracy_score, n_bootstrap, confidence_level),
            'auc_roc': calculate_ci_and_pvalue(y_true, y_scores, calculate_auc_roc, n_bootstrap, confidence_level)
        }
    
    return metrics

def calculate_ci_and_pvalue(y_true, y_pred, metric_func, n_bootstrap=1000, confidence_level=0.95):
    observed_metric = metric_func(y_true, y_pred)
    
    # Helper function to identify the metric type
    def identify_metric(func):
        if func == accuracy_score or (hasattr(func, '__name__') and func.__name__ == 'accuracy_score'):
            return 'accuracy'
        elif func in [precision_score, recall_score, f1_score] or \
             (hasattr(func, '__name__') and func.__name__ in ['precision_score', 'recall_score', 'f1_score']) or \
             (callable(func) and func.__name__ == '<lambda>'):  # This line handles lambda functions
            return 'classification'
        elif func == matthews_corrcoef or (hasattr(func, '__name__') and func.__name__ == 'matthews_corrcoef'):
            return 'mcc'
        elif func in [balanced_accuracy_score, calculate_auc_roc] or \
             (hasattr(func, '__name__') and func.__name__ in ['balanced_accuracy_score', 'calculate_auc_roc']):
            return 'balanced'
        else:
            return 'unknown'

    metric_type = identify_metric(metric_func)
    
    if metric_type == 'accuracy':
        # Use Wilson score interval for accuracy
        n = len(y_true)
        z = stats.norm.ppf((1 + confidence_level) / 2)
        p = observed_metric
        ci_lower = (p + z**2/(2*n) - z * np.sqrt((p*(1-p)+z**2/(4*n))/n)) / (1+z**2/n)
        ci_upper = (p + z**2/(2*n) + z * np.sqrt((p*(1-p)+z**2/(4*n))/n)) / (1+z**2/n)
        
        # Binomial test for p-value
        n_correct = int(observed_metric * n)
        p_value = stats.binomtest(n_correct, n, p=0.5, alternative='two-sided').pvalue
    
    elif metric_type == 'classification':
        # Use bootstrap for CI
        bootstrap_metrics = []
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, len(y_true), len(y_true))
            bootstrap_metrics.append(metric_func(y_true[indices], y_pred[indices]))
        
        ci_lower, ci_upper = np.percentile(bootstrap_metrics, [(1-confidence_level)/2*100, (1+confidence_level)/2*100])
        
        # Fisher's exact test for p-value
        cm = confusion_matrix(y_true, y_pred)
        _, p_value = fisher_exact(cm)
        
        # Cohen's h for effect size
        p1 = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0  # True Positive Rate
        p2 = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0  # False Positive Rate
        h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))  # Cohen's h
    
    elif metric_type == 'mcc':
        # Use bootstrap for CI
        bootstrap_metrics = []
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, len(y_true), len(y_true))
            bootstrap_metrics.append(metric_func(y_true[indices], y_pred[indices]))
        
        ci_lower, ci_upper = np.percentile(bootstrap_metrics, [(1-confidence_level)/2*100, (1+confidence_level)/2*100])
        
        # Fisher's z-transformation for p-value
        r = observed_metric
        z = np.arctanh(r)
        se = 1 / np.sqrt(len(y_true) - 3)
        p_value = 2 * (1 - stats.norm.cdf(abs(z) / se))
    
    elif metric_type == 'balanced':
        # Use bootstrap for CI and p-value
        bootstrap_metrics = []
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, len(y_true), len(y_true))
            bootstrap_metrics.append(metric_func(y_true[indices], y_pred[indices]))
        
        ci_lower, ci_upper = np.percentile(bootstrap_metrics, [(1-confidence_level)/2*100, (1+confidence_level)/2*100])
        p_value = np.mean(np.array(bootstrap_metrics) <= 0.5) * 2  # Two-tailed test
    
    else:
        raise ValueError(f"Unsupported metric function: {metric_func}")
    
    return observed_metric, ci_lower, ci_upper, p_value

def calculate_auc_roc(y_true, y_scores):
    # For multi-class, we use one-vs-rest approach
    if y_scores.shape[1] > 2:
        try:
            return roc_auc_score(y_true, y_scores, multi_class='ovr', average='macro')
        except ValueError:
            # If ROC AUC can't be calculated (e.g., a class is never predicted), return NaN
            return float('nan')
    else:
        try:
            return roc_auc_score(y_true, y_scores[:, 1])  # For binary classification
        except ValueError:
            return float('nan')

def create_model(num_classes):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def format_metric(metric_tuple):
    if len(metric_tuple) == 4:
        value, ci_lower, ci_upper, p_value = metric_tuple
    elif len(metric_tuple) == 1:
        value = metric_tuple[0]
        ci_lower, ci_upper, p_value = None, None, None
    else:
        raise ValueError(f"Unexpected metric tuple length: {len(metric_tuple)}")

    def is_nan(x):
        try:
            return np.isnan(x)
        except TypeError:
            return False

    if isinstance(value, torch.Tensor):
        value = value.cpu().item()
    
    value_str = f"{value:.4f}" if not is_nan(value) else "N/A"
    
    if ci_lower is not None and ci_upper is not None:
        ci_str = f"({ci_lower:.4f}, {ci_upper:.4f})" if not (is_nan(ci_lower) or is_nan(ci_upper)) else "N/A"
    else:
        ci_str = "N/A"
    
    if p_value is not None:
        p_str = f"{p_value:.4f}" if not is_nan(p_value) else "N/A"
    else:
        p_str = "N/A"
    
    return f"{value_str} CI: {ci_str} p: {p_str}"

def analyze_dataset(image_datasets):
    for phase in ['train', 'val']:
        print(f"\n{phase.capitalize()} Dataset Distribution:")
        class_counts = {cls: 0 for cls in image_datasets[phase].classes}
        for _, label in image_datasets[phase].images:
            class_counts[image_datasets[phase].classes[label]] += 1
        
        total_images = sum(class_counts.values())
        for cls, count in class_counts.items():
            percentage = (count / total_images) * 100
            print(f"  Class {cls}: {count} images ({percentage:.2f}%)")
        print(f"  Total: {total_images} images")

def print_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("            " + " ".join(f"{name:>10}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>12}" + " ".join(f"{cell:>10}" for cell in row))

def main(target_metric, num_epochs, patience, data_dir, n_bootstrap, confidence_level):
    print(f"Training with target metric: {target_metric}, epochs: {num_epochs}, patience: {patience}")
    print(f"Using dataset directory: {data_dir}")
    print(f"Bootstrap iterations: {n_bootstrap}, Confidence level: {confidence_level}")

    # Load Data
    image_datasets = {x: CustomDataset(os.path.join(data_dir, x), data_transforms['val']) 
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Analyze dataset distribution and print it
    analyze_dataset(image_datasets)

    # Load pre-trained ResNet model
    model = create_model(len(class_names))
    model = model.to(device)
    model = model.to(torch.float32)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training fuction
    def train_model(model, criterion, optimizer, scheduler, num_epochs=50, patience=10):
        writer = SummaryWriter()
        best_model_wts = model.state_dict()
        best_metric_value = float('-inf')
        no_improve_epochs = 0
    
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
        
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
            
                running_loss = 0.0
                running_corrects = 0
                all_labels = []
                all_preds = []
                all_scores = []
            
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                    optimizer.zero_grad()
                
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                    all_labels.extend(labels.cpu().detach().numpy())
                    all_preds.extend(preds.cpu().detach().numpy())
                    all_scores.extend(outputs.cpu().detach().numpy())
            
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.float() / dataset_sizes[phase]
            
                # Calculate additional metrics
                metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_scores), n_bootstrap=n_bootstrap, confidence_level=confidence_level)

                print(f'{phase} Loss: {format_metric((epoch_loss,))} ' + ' '.join(f'{k}: {format_metric(v)}' for k, v in metrics.items()))
                
                # Update the TensorBoard logging:
                writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
                for k, v in metrics.items():
                    writer.add_scalar(f'{phase} {k}', v[0], epoch)
                    writer.add_scalar(f'{phase} {k}_ci_lower', v[1], epoch)
                    writer.add_scalar(f'{phase} {k}_ci_upper', v[2], epoch)
                    writer.add_scalar(f'{phase} {k}_p_value', v[3], epoch)
            
                if phase == 'val':
                    scheduler.step(epoch_loss)
                    if metrics[target_metric][0] > best_metric_value:
                        best_metric_value = metrics[target_metric][0]
                        best_model_wts = model.state_dict()
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
            
                    if no_improve_epochs >= patience:
                        # TODO: Previously this would not cause stop
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        model.load_state_dict(best_model_wts)
                        return model
        
            print()
    
        print(f'Best val {target_metric}: {best_metric_value:4f}')
        model.load_state_dict(best_model_wts)
        writer.close()
        return model

    # Train the model
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs, patience=patience)

    # Evaluate the model on the validation set
    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())

    # Calculate final metrics
    final_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_scores))

    print("\nFinal Metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {format_metric(v)}")

    # Print confusion matrix
    print_confusion_matrix(all_labels, all_preds, class_names)

    # Save the trained model
    torch.save(model.state_dict(), 'resnet_model.pth')

    print("Training complete. Model saved.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='Train a model with specified parameters')
    parser.add_argument('--metric', type=str, default='accuracy', 
                        choices=['accuracy', 'precision', 'recall', 'f1', 'mcc', 'balanced_acc', 'auc_roc'],
                        help='Metric to optimize during training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations for confidence intervals')
    parser.add_argument('--confidence_level', type=float, default=0.95,
                        help='Confidence level for the confidence intervals')
    args = parser.parse_args()
    
    main(args.metric, args.epochs, args.patience, args.data_dir, args.n_bootstrap, args.confidence_level)