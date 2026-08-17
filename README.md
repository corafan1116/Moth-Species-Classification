# Moth Species Classification

Deep learning models for automated moth species classification, robustness evaluation and error analysis.

## Project Overview

This project investigates deep learning approaches for automated classification of 20 moth species.

Five models are evaluated within a unified experimental framework:

- Baseline CNN
- CNN with Data Augmentation
- ResNet50
- EfficientNet-B0
- Vision Transformer (ViT)

In addition to classification performance, the project evaluates model robustness under different image conditions, including low-light conditions, Gaussian blur and image rotation. Cross-model error analysis and Grad-CAM-based visualisation are also included.

## Repository Structure

```text
Moth-Species-Classification/
│
├── README.md
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
| `LowLight_GradCAM.ipynb` | Grad-CAM-based visualisation for model feature analysis under low-light conditions |

## Environment

The experiments were developed using the following software environment:

| Component | Version / Specification |
|---|---|
| Programming Language | Python 3.10 |
| Deep Learning Framework | TensorFlow 2.16.2 |
| High-level API | Keras |
| Image Processing | OpenCV |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Visualisation | Matplotlib |
| Development Environment | Visual Studio Code |
| Training Platform | Google Colab |
| GPU | NVIDIA T4 |

Model training was conducted using Google Colab with an NVIDIA T4 GPU, while Visual Studio Code was used for local development and debugging.

## Dataset

The project uses a subset of 20 moth species from the publicly available *Butterflies and Moths Species Classification Dataset* on Kaggle.

Only the moth categories were retained for this study; butterfly categories were excluded.

The original dataset is not included in this repository.

### Dataset Configuration

- Number of selected species: 20
- Image format: RGB JPG
- Image resolution: 224 × 224 pixels
- Dataset split: Training / Validation / Test
- Dataset split: Original split provided by the dataset

The original training, validation and testing split was maintained throughout the experiments to ensure consistency when comparing different models.

## Data Preprocessing

All images were resized to 224 × 224 pixels and pixel values were normalised to the range `[0, 1]`.

Data augmentation was applied only to the training dataset using TensorFlow augmentation layers.

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

A second CNN experiment used the baseline CNN architecture while applying online data augmentation to the training data.

### Transfer Learning Models

Three transfer learning models were evaluated:

- ResNet50
- EfficientNet-B0
- Vision Transformer (ViT)

The transfer learning models were initialised using ImageNet-pretrained weights. Their original classification layers were replaced with task-specific output layers containing 20 neurons with Softmax activation.

The pretrained feature extraction layers were kept frozen and the newly added classification layers were trained using the moth dataset.

## Training Configuration

The models were trained using the following configuration:

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Initial Learning Rate | 0.001 |
| Batch Size | 32 |
| Maximum Epochs | 20 |
| Early Stopping | Enabled |
| Model Checkpointing | Enabled |
| Input Image Size | 224 × 224 |
| Number of Classes | 20 |

Early stopping was applied based on validation loss, and ModelCheckpoint was used to save the best-performing model.

## Model Evaluation

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Accuracy was used as the primary overall performance metric, while precision, recall and F1-score provided additional performance information. Confusion matrices were used to examine class-specific prediction behaviour and common misclassification patterns.

## Robustness Evaluation

Model robustness was evaluated using three transformed versions of the original test dataset:

- Low-light conditions
- Gaussian blur
- Image rotation

Each trained model was evaluated on:

1. The original test dataset
2. The low-light dataset
3. The Gaussian-blur dataset
4. The rotated dataset

Robustness was assessed by comparing classification accuracy across the original and transformed datasets.

The accuracy drop was calculated as:

```text
Accuracy Drop = Accuracy_Original - Accuracy_Perturbed
