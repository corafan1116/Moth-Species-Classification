# Moth Species Classification

This repository contains the code and supporting materials for the MSc Extended Research Project, *Automated Classification of Moth Species for Biodiversity Monitoring Using Deep Learning*.

The project evaluates five models:

- Baseline CNN
- CNN with Data Augmentation
- ResNet50
- EfficientNet-B0
- Vision Transformer (ViT)

The project also evaluates model robustness under low-light conditions, Gaussian blur and image rotation, together with cross-model error analysis and Grad-CAM visualisation.

## Repository Structure

```text
Moth-Species-Classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── CNN_Baseline.ipynb
│   ├── CNN_Augmentation.ipynb
│   ├── ResNet50.ipynb
│   ├── EfficientNetB0.ipynb
│   ├── ViT.ipynb
│   ├── robustness_analysis.ipynb
│   ├── CrossModel_ErrorAnalysis.ipynb
│   └── LowLight_GradCAM.ipynb
│
└── results/
    ├── overall_classification_results.csv
    ├── model_complexity_results.csv
    ├── robustness_accuracy_results.csv
    │
    └── error_analysis/
        ├── CNN_predictions.csv
        ├── CNNAug_predictions.csv
        ├── ResNet50_predictions.csv
        ├── EfficientNetB0_predictions.csv
        ├── ViT_predictions.csv
        ├── CrossModel_ConfusionPairs.csv
        └── Species_Error_Frequency.csv
```

## Environment and Installation

The experiments were conducted using Python 3.10 and TensorFlow 2.16.2. Keras Hub is required for the Vision Transformer implementation. Model training was conducted using Google Colab with an NVIDIA T4 GPU.

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses a subset of 20 moth species from the publicly available *Butterfly & Moths Image Classification 100 species* dataset on Kaggle.

Only the moth categories were retained for this study; butterfly categories were excluded. The original dataset is not included in this repository.

The dataset can be obtained from the original [Kaggle source](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species).

### Dataset Configuration

- Selected classes: 20 moth species
- Image format: RGB
- Image resolution: 224 × 224 pixels
- Data split: Training / Validation / Test

The following 20 moth species were selected:

| Class Index | Species |
|---:|---|
| 0 | ARCIGERA FLOWER MOTH |
| 1 | ATLAS MOTH |
| 2 | BANDED TIGER MOTH |
| 3 | BIRD CHERRY ERMINE MOTH |
| 4 | CINNABAR MOTH |
| 5 | CLEARWING MOTH |
| 6 | COMET MOTH |
| 7 | EMPEROR GUM MOTH |
| 8 | GARDEN TIGER MOTH |
| 9 | GIANT LEOPARD MOTH |
| 10 | HERCULES MOTH |
| 11 | HUMMING BIRD HAWK MOTH |
| 12 | IO MOTH |
| 13 | LUNA MOTH |
| 14 | MADAGASCAN SUNSET MOTH |
| 15 | OLEANDER HAWK MOTH |
| 16 | POLYPHEMUS MOTH |
| 17 | ROSY MAPLE MOTH |
| 18 | SIXSPOT BURNET MOTH |
| 19 | WHITE LINED SPHINX MOTH |

### Dataset Split

The selected dataset contains 2,767 images across the three predefined splits:

| Split | Number of Images | Number of Classes |
|---|---:|---:|
| Training | 2,567 | 20 |
| Validation | 100 | 20 |
| Test | 100 | 20 |
| **Total** | **2,767** | **20** |

The predefined training, validation and test split was maintained throughout the main experiments.

### Dataset Directory Structure

After downloading the dataset, the selected moth images should be organised as follows:

```text
dataset/
├── train/
│   ├── ARCIGERA FLOWER MOTH/
│   ├── ATLAS MOTH/
│   ├── BANDED TIGER MOTH/
│   ├── ...
│   └── WHITE LINED SPHINX MOTH/
│
├── valid/
│   ├── ARCIGERA FLOWER MOTH/
│   ├── ...
│   └── WHITE LINED SPHINX MOTH/
│
└── test/
    ├── ARCIGERA FLOWER MOTH/
    ├── ...
    └── WHITE LINED SPHINX MOTH/
```

Only the 20 selected moth classes are required for the experiments.

## Data Preprocessing

Images were resized to 224 × 224 pixels and loaded using TensorFlow's `image_dataset_from_directory` function. Model-specific preprocessing was then applied within each model pipeline.

Data augmentation was applied only to the training data using TensorFlow augmentation layers:

- Horizontal flip
- Rotation
- Gaussian noise

Validation and test datasets were not augmented.

## Models

### Baseline CNN

The baseline CNN was trained from scratch. The architecture consists of:

- Convolutional layers with 32, 64 and 128 filters
- ReLU activation
- Max pooling after each convolutional layer
- Fully connected layer with 128 neurons
- Dropout of 0.5
- 20-neuron Softmax output layer

### CNN with Data Augmentation

The second CNN experiment used the same general baseline CNN architecture while applying online data augmentation to the training data.

### Transfer Learning Models

Three transfer learning models were evaluated:

- ResNet50
- EfficientNet-B0
- Vision Transformer (ViT)

The transfer learning models were initialised using ImageNet-pretrained weights. The pretrained feature extraction layers were kept frozen, while task-specific classification layers were trained for the 20 moth classes.

## Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Initial Learning Rate | 0.001 |
| Batch Size | 32 |
| Maximum Epochs | 20 |
| Early Stopping | Validation loss (patience = 5) |
| Input Image Size | 224 × 224 |

Early stopping was applied based on validation loss, and the best validation weights were restored when training stopped.

## Model Evaluation

Models were evaluated using accuracy, precision, recall, F1-score and confusion matrices, with accuracy used as the primary metric.

## Robustness Evaluation

Robustness was evaluated on three transformed versions of the test set:

- Low-light conditions
- Gaussian blur
- Image rotation

Each model was evaluated on the original and transformed test sets, and performance was compared using classification accuracy.

## Error Analysis

The `results/error_analysis/` directory contains prediction-level results, cross-model confusion pairs and species-level error frequencies used for the cross-model error analysis.

## Grad-CAM Analysis

`LowLight_GradCAM.ipynb` contains the Grad-CAM-based visualisation analysis used to investigate model behaviour under low-light conditions.

The Grad-CAM notebook requires a trained `EfficientNetB0.keras` model. This model file is not included in the repository and should be generated by running `EfficientNetB0.ipynb` before executing the Grad-CAM analysis.

## How to Reproduce

1. Download the dataset from the original Kaggle source and prepare the 20 selected classes using the specified training, validation and test structure.
2. Install the required packages using `pip install -r requirements.txt`.
3. Update the dataset path in the relevant notebook where required.
4. Run the notebook cells sequentially to reproduce the corresponding experiment or analysis.
5. For Grad-CAM, run `EfficientNetB0.ipynb` first to generate the required `EfficientNetB0.keras` model.
