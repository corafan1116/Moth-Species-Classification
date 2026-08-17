# Moth Species Classification

This repository contains the code and supporting materials for an MSc Extended Research Project investigating deep learning approaches for automated moth species classification.

The project evaluates five models within a consistent experimental framework:

- Baseline CNN
- CNN with Data Augmentation
- ResNet50
- EfficientNet-B0
- Vision Transformer (ViT)

In addition to classification performance, the project evaluates model robustness under different image conditions, including low-light conditions, Gaussian blur and image rotation. Cross-model error analysis and Grad-CAM-based visualisation are also included.

## Project Overview

The study focuses on the automated classification of 20 moth species selected from the publicly available *Butterflies and Moths Species Classification Dataset* on Kaggle.

The experiments compare a CNN trained from scratch with transfer learning approaches using ResNet50, EfficientNet-B0 and Vision Transformer (ViT). The same dataset split and preprocessing procedure are used to provide a consistent basis for model comparison.

The project also investigates how model performance changes under controlled image perturbations and analyses common misclassification patterns across models.

## Repository Structure

```text
Moth-Species-Classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── CNN_Baseline.ipynb
│   ├── CNN_Augmentation.ipynb
│   ├── ResNet50.ipynb
│   ├── EfficientNetB0.ipynb
│   ├── ViT.ipynb
│   ├── robustness_analysis.ipynb
│   ├── CrossModel_ErrorAnalysis.ipynb
│   └── LowLight_GradCAM.ipynb
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

### Notebook Descriptions

| Notebook | Purpose |
|---|---|
| `data_exploration.ipynb` | Dataset exploration and inspection |
| `CNN_Baseline.ipynb` | Development and evaluation of the baseline CNN |
| `CNN_Augmentation.ipynb` | Training and evaluation of the CNN with data augmentation |
| `ResNet50.ipynb` | Training and evaluation of ResNet50 |
| `EfficientNetB0.ipynb` | Training and evaluation of EfficientNet-B0 |
| `ViT.ipynb` | Training and evaluation of Vision Transformer (ViT) |
| `robustness_analysis.ipynb` | Robustness evaluation under image perturbations |
| `CrossModel_ErrorAnalysis.ipynb` | Cross-model misclassification and error analysis |
| `LowLight_GradCAM.ipynb` | Grad-CAM visualisation for model feature analysis under low-light conditions |

## Environment

The experiments were developed using the following software environment:

| Component | Version / Specification | Purpose |
|---|---|---|
| Programming Language | Python 3.10 | Model development |
| Deep Learning Framework | TensorFlow 2.16.2 | Model training and evaluation |
| High-level API | Keras | Model implementation |
| Image Processing Library | OpenCV | Image preprocessing |
| Numerical Computing | NumPy | Data manipulation |
| Development Environment | Visual Studio Code | Local development and debugging |
| Training Platform | Google Colab | GPU-accelerated training |

Model training was conducted using Google Colab with an NVIDIA T4 GPU, while Visual Studio Code was used for local development and debugging.

## Installation

Install the required Python packages using:

`pip install -r requirements.txt`

## Dataset

The project uses a subset of 20 moth species from the publicly available *Butterfly & Moths Image Classification 100 species* on Kaggle.

Only the moth categories were retained for this study; butterfly categories were excluded.

The original dataset is **not included in this repository**.

The dataset can be obtained from its original [Kaggle source](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species).

### Dataset Configuration

- Selected classes: 20 moth species
- Image format: RGB
- Image resolution: 224 × 224 pixels
- Data split: Training / Validation / Test
- Original dataset split: Maintained throughout the experiments

The original training, validation and testing split was maintained throughout the experiments to ensure consistency when comparing different models.

## Data Preprocessing

All images were resized to 224 × 224 pixels and pixel values were normalised to the range `[0, 1]`.

Data augmentation was applied only to the training dataset using TensorFlow's built-in image augmentation layers.

The augmentation operations were:

- Horizontal flip
- Rotation
- Zoom

The validation and testing datasets were not augmented.

The datasets were loaded using the TensorFlow `tf.data` pipeline for batch processing during training.

## Models

### Baseline CNN

The baseline CNN was trained from scratch using the moth dataset.

The architecture consists of:

- Convolutional layer: 32 filters
- Convolutional layer: 64 filters
- Convolutional layer: 128 filters
- ReLU activation
- Max pooling after each convolutional layer
- Fully connected layer: 128 neurons
- Dropout: 0.5
- Output layer: 20 neurons with Softmax activation

### CNN with Data Augmentation

A second CNN experiment used the same general baseline CNN approach while applying online data augmentation to the training data.

### Transfer Learning Models

Three transfer learning models were evaluated:

- ResNet50
- EfficientNet-B0
- Vision Transformer (ViT)

The transfer learning models were initialised using ImageNet-pretrained weights. Their original classification layers were replaced with task-specific output layers containing 20 neurons with Softmax activation.

For the transfer learning experiments, the pretrained feature extraction layers were kept frozen and the newly added classification layers were trained using the moth dataset.

## Training Configuration

The models were trained using the following configuration:

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Initial Learning Rate | 0.001 |
| Batch Size | 32 |
| Maximum Epochs | 20 |
| Early Stopping | Validation loss |
| Model Checkpoint | Best validation model |
| Input Image Size | 224 × 224 |
| Data Augmentation | Horizontal Flip, Rotation, Zoom |

The same training, validation and testing datasets were used throughout the main model comparison. Early stopping was applied based on validation loss, and the best-performing model was saved using the ModelCheckpoint callback.

## Model Evaluation

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Accuracy was used as the primary overall performance metric, while precision, recall and F1-score provided additional class-level performance information. Confusion matrices were used to examine class-specific prediction behaviour and common misclassification patterns.

## Robustness Evaluation

Model robustness was evaluated using three additional test datasets generated from the original test dataset using controlled image transformations:

- Reduced brightness (low-light conditions)
- Gaussian blur
- Image rotation

The original class labels were retained for all transformed images.

Each trained model was evaluated on:

1. The original test dataset
2. The low-light dataset
3. The Gaussian-blur dataset
4. The rotated dataset

Robustness was assessed by comparing accuracy on the original and transformed datasets.

The accuracy drop was calculated as:

`Accuracy Drop = Accuracy_Original - Accuracy_Perturbed`

A smaller accuracy drop indicates lower sensitivity to the corresponding image perturbation.

## Error Analysis

The `error_analysis/` directory contains prediction-level results and derived error-analysis data used in the project.

These files include:

- Per-model prediction results
- Cross-model confusion pairs
- Species-level error frequencies

The error-analysis notebooks use these results to investigate common misclassification patterns and compare errors across the evaluated models.

## Grad-CAM Analysis

`LowLight_GradCAM.ipynb` contains the Grad-CAM-based visualisation analysis conducted as part of the investigation into model behaviour under low-light conditions.

The analysis is intended to provide visual insight into the image regions contributing to model predictions.

## How to Use This Repository

1. Obtain the publicly available moth dataset from its original source.
2. Prepare the dataset using the original training, validation and testing split.
3. Open the relevant notebook in `notebooks/`.
4. Ensure the software environment and package versions are compatible with the environment described above.
5. Update the dataset path in the notebook where required.
6. Run the notebook cells sequentially to reproduce the corresponding experiment or analysis.

The notebooks are organised according to the main stages of the research, from dataset exploration and model training to robustness evaluation and error analysis.

## Reproducibility Notes

To support reproducibility, the repository provides:

- The notebooks used for model development and analysis
- The main training configuration
- The software environment used for the experiments
- Dataset and preprocessing information
- Model architecture information
- Robustness evaluation procedures
- Prediction-level error-analysis files

The original dataset and trained model files are not included in this repository.
