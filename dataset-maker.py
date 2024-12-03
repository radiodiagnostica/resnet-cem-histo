import argparse
import os
import csv
import shutil
import random
from pathlib import Path

def get_csv_columns(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
    return headers

def get_patient_label_map(csv_file):
    headers = get_csv_columns(csv_file)
    
    patient_column = next((col for col in headers if 'PZ' in col.upper() or 'PATIENT' in col.upper()), None)
    label_column = next((col for col in headers if 'IMMUNOFENOTIPO' in col.upper() or 'LABEL' in col.upper()), None)
    
    if not patient_column or not label_column:
        print(f"Error: Could not identify patient and label columns. Available columns are: {headers}")
        exit(1)
    
    patient_label_map = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_name = row[patient_column].strip()
            label = row[label_column].strip()
            patient_label_map[patient_name] = label
    return patient_label_map

def find_matching_folders(root_dir, patient_name):
    surname = patient_name.split()[0].lower()
    initials = ''.join([name[0].lower() for name in patient_name.split()[1:]])
    pattern = f"{surname}{initials}"
    
    matching_folders = []
    for folder in os.listdir(root_dir):
        if folder.lower().startswith(pattern):
            matching_folders.append(os.path.join(root_dir, folder))
    return matching_folders

def create_dataset_structure(output_dir):
    for split in ['train', 'val']:
        for label in range(1, 6):
            os.makedirs(os.path.join(output_dir, split, str(label)), exist_ok=True)

def copy_and_rename_images(src_folders, dst_folder, patient_name, label):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_count = 0
    
    for src_folder in src_folders:
        for root, _, files in os.walk(src_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    src_path = os.path.join(root, file)
                    patient_name_no_spaces = patient_name.replace(' ', '-')
                    new_filename = f"{patient_name_no_spaces}_{image_count}{Path(file).suffix}"
                    dst_path = os.path.join(dst_folder, new_filename)
                    try:
                        shutil.copy2(src_path, dst_path)
                        image_count += 1
                    except OSError as e:
                        print(f"Error copying file {src_path} to {dst_path}: {e}")
                        print(f"Please check permissions for the destination folder: {dst_folder}")

def create_resnet_dataset(csv_file, root_dir, output_dir, train_ratio=0.8):
    patient_label_map = get_patient_label_map(csv_file)
    try:
        create_dataset_structure(output_dir)
    except OSError as e:
        print(f"Error creating dataset structure in {output_dir}: {e}")
        print("Please check if you have write permissions for this directory.")
        return

    for patient_name, label in patient_label_map.items():
        matching_folders = find_matching_folders(root_dir, patient_name)
        if matching_folders:
            split = 'train' if random.random() < train_ratio else 'val'
            dst_folder = os.path.join(output_dir, split, label)
            copy_and_rename_images(matching_folders, dst_folder, patient_name, label)
        else:
            print(f"No matching folder found for patient: {patient_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for ResNet training')
    parser.add_argument('--csv_file', type=str, default='dataset-map.csv',
                        help='Path to the CSV file containing patient labels')
    parser.add_argument('--root_dir', type=str, default='.',
                        help='Root directory containing patient folders')
    parser.add_argument('--output_dir', type=str, default='dataset',
                        help='Output directory for the prepared dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data to use for training (default: 0.8)')
    args = parser.parse_args()
    
    create_resnet_dataset(args.csv_file, args.root_dir, args.output_dir, args.train_ratio)
    print("Dataset creation completed.")
