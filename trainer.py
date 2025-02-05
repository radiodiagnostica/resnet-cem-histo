#### Training Script

import os
import argparse
import cv2
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

from tabulate import tabulate

def format_metrics(metrics):
    table_rows = []
    for metric_name, metric_value in metrics.items():
        if len(metric_value) == 4:
            value, ci_lower, ci_upper, p_value = metric_value
        elif len(metric_value) == 1:
            value = metric_value[0]
            ci_lower, ci_upper, p_value = None, None, None
        else:
            raise ValueError(f"Unexpected metric tuple length: {len(metric_value)}")

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
            p_str = f"{p_value:.8f}" if not is_nan(p_value) else "N/A"
        else:
            p_str = "N/A"
        
        value = f"{value_str} {ci_str}; {p_str}"
        table_rows.append([metric_name, value]) 

    # Define the headers
    headers = ["Metric", "Value (CI95; P)"]

    # Print the table
    print("Metrics:")
    print(tabulate(table_rows, headers=headers, tablefmt="pipe"))

def analyze_dataset(image_datasets):
    phases = ['train', 'val']
    
    if 'external_val' in image_datasets:
        phases.append('external_val')
    
    for phase in phases:
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

def generate_activation_maps(model, image_path, class_idx, transform):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    import numpy as np

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Initialize GradCAM
    target_layers = [model.layer4[-1]]  # Last layer of ResNet

    # Ensure model is in eval mode but with gradients enabled
    model.eval()
    
    # Create a wrapper class to handle MPS/CUDA/CPU devices
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            
        def forward(self, x):
            return self.model(x)

        def __call__(self, x):
            return self.forward(x)

    wrapped_model = ModelWrapper(model)
    
    # Initialize GradCAM with the wrapped model
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    
    # Define target category
    targets = [ClassifierOutputTarget(class_idx)]
    
    # Generate activation map
    try:
        # Ensure input tensor requires grad
        input_tensor.requires_grad = True
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Convert PIL Image to numpy array
        rgb_img = np.array(img)
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = rgb_img / 255.0
        
        # Ensure rgb_img is in the correct format
        if rgb_img.max() > 1:
            rgb_img = rgb_img / 255.0
        
        # Overlay activation map on original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        return visualization
    
    except Exception as e:
        print(f"Error generating activation map: {str(e)}")
        # Return the original image if CAM generation fails
        rgb_img = np.array(img)
        rgb_img = cv2.resize(rgb_img, (224, 224))
        return rgb_img

def main(target_metric, num_epochs, patience, data_dir, n_bootstrap, confidence_level):
    print(f"Training with target metric: {target_metric}, epochs: {num_epochs}, patience: {patience}")
    print(f"Using dataset directory: {data_dir}")
    print(f"Bootstrap iterations: {n_bootstrap}, Confidence level: {confidence_level}")

    # Load Data
    image_datasets = {x: CustomDataset(os.path.join(data_dir, x), data_transforms['val']) 
                      for x in ['train', 'val']}
    
    # Check for external validation set
    external_val_dir = os.path.join(data_dir, 'external_val')
    has_external_val = os.path.exists(external_val_dir)
    if has_external_val:
        image_datasets['external_val'] = CustomDataset(external_val_dir, data_transforms['val'])
        print("External validation set found!")

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
                   for x in image_datasets.keys()}
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets.keys()}
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

                print(f'\n{phase} Loss: {epoch_loss:.4f}')

                format_metrics(metrics)
                
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

    # Evaluate on validation set
    def evaluate_model(model, dataloader, phase):
        model.eval()
        all_labels = []
        all_preds = []
        all_scores = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(outputs.cpu().numpy())

        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_scores))
        print(f"\n{phase.capitalize()} Metrics:")
        format_metrics(metrics)
        print_confusion_matrix(all_labels, all_preds, class_names)
        return metrics, all_labels, all_preds

    # Evaluate on validation set
    val_metrics, val_labels, val_preds = evaluate_model(model, dataloaders['val'], 'validation')

    # Evaluate on external validation set if available
    if has_external_val:
        print("\nEvaluating on external validation set:")
        ext_metrics, ext_labels, ext_preds = evaluate_model(model, dataloaders['external_val'], 'external validation')

    # Generate activation maps
    print("\nGenerating activation maps...")
    os.makedirs('activation_maps', exist_ok=True)

    def get_sample_images(dataset, num_samples_per_class=2):
        class_examples = {cls: [] for cls in range(len(class_names))}
        for i in range(len(dataset)):
            img_path, label = dataset.images[i]
            if len(class_examples[label]) < num_samples_per_class:
                class_examples[label].append(img_path)
            if all(len(examples) >= num_samples_per_class for examples in class_examples.values()):
                break
        return class_examples

    # Get sample images from validation set
    val_examples = get_sample_images(image_datasets['val'])

    # Generate and save activation maps
    model.eval()
    for class_idx, img_paths in val_examples.items():
        for i, img_path in enumerate(img_paths):
            visualization = generate_activation_maps(model, img_path, class_idx, data_transforms['val'])
            
            # Save the visualization
            class_name = class_names[class_idx]
            output_path = f'activation_maps/class_{class_name}_sample_{i+1}_activation.png'
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"Saved activation map for class {class_name} (sample {i+1})")

    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'external_val_metrics': ext_metrics if has_external_val else None,
        'class_names': class_names
    }, 'resnet_model.pth')

    print("\nTraining complete. Model and results saved.")

def evaluate_existing_model(model_path, data_dir, n_bootstrap, confidence_level):
    print(f"Evaluating model: {model_path}")
    print(f"Using dataset directory: {data_dir}")

    # Load Data
    image_datasets = {x: CustomDataset(os.path.join(data_dir, x), data_transforms['val']) 
                    for x in ['val']}
    
    # Check for external validation set
    external_val_dir = os.path.join(data_dir, 'external_val')
    has_external_val = os.path.exists(external_val_dir)
    if has_external_val:
        image_datasets['external_val'] = CustomDataset(external_val_dir, data_transforms['val'])
        print("External validation set found!")

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=False, num_workers=0)
                  for x in image_datasets.keys()}
    class_names = image_datasets['val'].classes

    # Load the model
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(len(class_names))
    
    # Handle different saving formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with metadata
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Simple format (just the state dict)
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()

    def evaluate_dataset(model, dataloader, phase):
        all_labels = []
        all_preds = []
        all_scores = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(outputs.cpu().numpy())

        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), 
                                 np.array(all_scores), n_bootstrap, confidence_level)
        print(f"\n{phase.capitalize()} Metrics:")
        format_metrics(metrics)
        print_confusion_matrix(all_labels, all_preds, class_names)
        return metrics

    # Evaluate on validation set
    print("\nEvaluating on validation set:")
    val_metrics = evaluate_dataset(model, dataloaders['val'], 'validation')

    # Evaluate on external validation set if available
    if has_external_val:
        print("\nEvaluating on external validation set:")
        ext_metrics = evaluate_dataset(model, dataloaders['external_val'], 'external validation')

    # Generate activation maps
    print("\nGenerating activation maps...")
    os.makedirs('activation_maps', exist_ok=True)

    def get_sample_images(dataset, num_samples_per_class=2):
        class_examples = {cls: [] for cls in range(len(class_names))}
        for i in range(len(dataset)):
            img_path, label = dataset.images[i]
            if len(class_examples[label]) < num_samples_per_class:
                class_examples[label].append(img_path)
            if all(len(examples) >= num_samples_per_class for examples in class_examples.values()):
                break
        return class_examples

    # Get sample images and generate activation maps
    val_examples = get_sample_images(image_datasets['val'])
    for class_idx, img_paths in val_examples.items():
        for i, img_path in enumerate(img_paths):
            visualization = generate_activation_maps(model, img_path, class_idx, data_transforms['val'])
            class_name = class_names[class_idx]
            output_path = f'activation_maps/class_{class_name}_sample_{i+1}_activation.png'
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"Saved activation map for class {class_name} (sample {i+1})")

    print("\nEvaluation complete.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='Train or evaluate a model with specified parameters')
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
    parser.add_argument('--evaluate', type=str,
                        help='Path to existing model for evaluation')
    args = parser.parse_args()

    if args.evaluate:
        # Run evaluation mode
        evaluate_existing_model(args.evaluate, args.data_dir, args.n_bootstrap, args.confidence_level)
    else:
        # Run training mode
        main(args.metric, args.epochs, args.patience, args.data_dir, args.n_bootstrap, args.confidence_level)

