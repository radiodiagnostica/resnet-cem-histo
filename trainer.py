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
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, balanced_accuracy_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np

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

def calculate_metrics(y_true, y_pred, y_scores):
    # Suppress the UndefinedMetricWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    auc_roc = calculate_auc_roc(y_true, y_scores)
    accuracy = (y_true == y_pred).mean()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'balanced_acc': balanced_acc,
        'auc_roc': auc_roc
    }

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

def format_metric(value):
    if isinstance(value, torch.Tensor):
        value = value.cpu().item()
    return f"{value:.4f}" if not np.isnan(value) else "N/A"

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

def main(target_metric, num_epochs, patience, data_dir):
    print(f"Training with target metric: {target_metric}, epochs: {num_epochs}, patience: {patience}")
    print(f"Using dataset directory: {data_dir}")

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
                metrics = calculate_metrics(
                    np.array(all_labels), np.array(all_preds), np.array(all_scores)
                )

                print(f'{phase} Loss: {format_metric(epoch_loss)} ' + 
                      ' '.join(f'{k}: {format_metric(v)}' for k, v in metrics.items()))
                
                writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
                for k, v in metrics.items():
                    writer.add_scalar(f'{phase} {k}', v, epoch)
            
                if phase == 'val':
                    scheduler.step(epoch_loss)
                    if metrics[target_metric] > best_metric_value:
                        best_metric_value = metrics[target_metric]
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
    args = parser.parse_args()
    main(args.metric, args.epochs, args.patience, args.data_dir)
