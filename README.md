# ResNet for Breast Cancer Hormone Receptors Prediction from CEM

This project contains two main scripts:
1. `dataset-maker.py`: Prepares the dataset for training
2. `train_model.py`: Trains a ResNet model on the prepared dataset

## Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- Pillow
- matplotlib
- tensorboard
- scikit-learn

You can install the required packages using:

```
pip install -r requirements.txt
```

## Preparing the Dataset

The `dataset-maker.py` script organizes your image data into a structure suitable for training.

### Usage

```
python dataset-maker.py [--csv_file CSV_FILE] [--root_dir ROOT_DIR] [--output_dir OUTPUT_DIR] [--train_ratio TRAIN_RATIO]
```

### Arguments

- `--csv_file`: Path to the CSV file containing patient labels (default: 'dataset-map.csv')
- `--root_dir`: Root directory containing patient folders (default: current directory)
- `--output_dir`: Output directory for the prepared dataset (default: 'dataset')
- `--train_ratio`: Ratio of data to use for training (default: 0.8)

### Example

```
python dataset-maker.py --csv_file path/to/dataset-map.csv --root_dir path/to/patient/folders --output_dir path/to/prepared/dataset --train_ratio 0.8
```

## Training the Model

The `train_model.py` script trains a ResNet model on the prepared dataset.

### Usage

```
python train_model.py [--metric METRIC] [--epochs EPOCHS] [--patience PATIENCE] [--data_dir DATA_DIR]
```

### Arguments

- `--metric`: Metric to optimize during training (choices: accuracy, precision, recall, f1, mcc, balanced_acc, auc_roc; default: accuracy)
- `--epochs`: Number of training epochs (default: 50)
- `--patience`: Number of epochs with no improvement after which training will be stopped (default: 10)
- `--data_dir`: Path to the prepared dataset directory (default: 'dataset')

### Example

```
python train_model.py --metric accuracy --epochs 100 --patience 15 --data_dir path/to/prepared/dataset
```

## Workflow

1. Prepare your dataset using `dataset-maker.py`
2. Train the model using `train_model.py`
3. The trained model will be saved as 'resnet_model.pth' in the current directory

## Notes

- Ensure that you have sufficient disk space for the prepared dataset and model checkpoints.
- Training may take a considerable amount of time depending on your hardware and the size of your dataset.
- You can monitor the training progress using TensorBoard. Launch TensorBoard by running `tensorboard --logdir=runs` in your terminal.
