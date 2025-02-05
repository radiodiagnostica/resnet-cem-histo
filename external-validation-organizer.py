import os
import shutil
import random

def create_external_val_set(base_path, external_val_split=0.2, random_seed=42):
    """
    Reorganize the dataset to create an external validation set from both train and val sets.
    
    Args:
        base_path (str): Path to the main dataset directory
        external_val_split (float): Proportion of data to move to external_val
        random_seed (int): Random seed for reproducibility
    """
    random.seed(random_seed)
    
    # Create external_val directory if it doesn't exist
    external_val_path = os.path.join(base_path, 'external_val')
    os.makedirs(external_val_path, exist_ok=True)
    
    # Create subdirectories for each class in external_val
    for class_name in ['1', '2']:
        os.makedirs(os.path.join(external_val_path, class_name), exist_ok=True)
    
    # Process each class
    for class_name in ['1', '2']:
        # Process train folder
        train_path = os.path.join(base_path, 'train', class_name)
        train_files = os.listdir(train_path)
        
        # Process val folder
        val_path = os.path.join(base_path, 'val', class_name)
        val_files = os.listdir(val_path)
        
        # Calculate number of files to move from each folder
        num_train_files_to_move = int(len(train_files) * external_val_split)
        num_val_files_to_move = int(len(val_files) * external_val_split)
        
        # Randomly select files to move from train
        train_files_to_move = random.sample(train_files, num_train_files_to_move)
        
        # Randomly select files to move from val
        val_files_to_move = random.sample(val_files, num_val_files_to_move)
        
        # Move selected files from train to external_val
        for file_name in train_files_to_move:
            source_path = os.path.join(train_path, file_name)
            dest_path = os.path.join(external_val_path, class_name, f"train_{file_name}")
            shutil.move(source_path, dest_path)
            
        # Move selected files from val to external_val
        for file_name in val_files_to_move:
            source_path = os.path.join(val_path, file_name)
            dest_path = os.path.join(external_val_path, class_name, f"val_{file_name}")
            shutil.move(source_path, dest_path)
            
        print(f"Class {class_name}:")
        print(f"  Moved {num_train_files_to_move} files from train to external validation")
        print(f"  Moved {num_val_files_to_move} files from validation to external validation")

def print_dataset_stats(base_path):
    """Print statistics about the dataset distribution"""
    print("\nDataset statistics:")
    for split in ['train', 'val', 'external_val']:
        print(f"\n{split} set:")
        for class_name in ['1', '2']:
            path = os.path.join(base_path, split, class_name)
            if os.path.exists(path):
                num_files = len(os.listdir(path))
                print(f"  Class {class_name}: {num_files} files")

def backup_dataset(base_path):
    """Create a backup of the dataset"""
    backup_path = base_path + '_backup'
    if not os.path.exists(backup_path):
        shutil.copytree(base_path, backup_path)
        print("Created backup of dataset")
    else:
        print("Backup already exists")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create external validation set')
    parser.add_argument('--dataset_path', required=True, help='Path to dataset')
    parser.add_argument('--split_ratio', type=float, default=0.2, help='Ratio of data to move to external_val')
    args = parser.parse_args()
    
    try:
        # Create backup
        # backup_dataset(dataset_path)
        
        # Print initial statistics
        print("\nInitial dataset distribution:")
        print_dataset_stats(args.dataset_path)
        
        # Create external validation set
        create_external_val_set(args.dataset_path, args.split_ratio)  # You can adjust the split ratio
        
        # Print final statistics
        print("\nFinal dataset distribution:")
        print_dataset_stats(args.dataset_path)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")