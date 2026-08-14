# Moth Species Classification

Deep learning models for automated moth species classification, robustness evaluation and Grad-CAM analysis.

## Project Overview

This project investigates the use of deep learning for automated moth species classification. Five models are evaluated within a unified experimental framework: a baseline Convolutional Neural Network (CNN), a CNN with data augmentation, ResNet50, EfficientNet-B0 and Vision Transformer (ViT).

In addition to evaluating overall classification performance, the project investigates model robustness under different image conditions, including low-light, Gaussian blur and rotation. Error analysis and Grad-CAM visualisation are also used to examine recurrent misclassification patterns and the visual features considered by the best-performing model.

The project aims to provide a more comprehensive evaluation of deep learning approaches for fine-grained moth species classification and their potential application to practical biodiversity monitoring.

## Research Objectives

The main objectives of this project are to:

- Develop a baseline CNN for moth species classification.
- Investigate the effect of data augmentation on CNN performance.
- Compare CNN-based and transfer learning models using standard classification metrics.
- Evaluate model robustness under low-light, Gaussian blur and rotation conditions.
- Analyse recurrent misclassification patterns across different models.
- Use Grad-CAM to investigate the visual regions contributing to model predictions.
- Assess the practical implications of model performance and computational complexity for biodiversity monitoring.

## Models

The following five models are evaluated:

| Model | Description |
|---|---|
| Baseline CNN | CNN trained from scratch |
| CNN + Data Augmentation | CNN trained with online data augmentation |
| ResNet50 | ImageNet-pretrained transfer learning model |
| EfficientNet-B0 | ImageNet-pretrained transfer learning model |
| ViT | ImageNet-pretrained Vision Transformer |

## Dataset

The project uses a publicly available moth image dataset. A subset of 20 moth species was selected for the experiments.

All images were resized to 224 × 224 pixels and normalised before model training. Data augmentation was applied online to the training data for the augmented CNN experiment.

The original dataset and transformed test images are not included in this repository. The dataset source and detailed preprocessing procedure are described in the dissertation.

The robustness transformation procedures used to generate the low-light, Gaussian blur and rotation test conditions are provided in `robustness_analysis.ipynb`.

## Experimental Setup

The experiments were implemented using Python and TensorFlow/Keras.

The main experimental settings include:

- Image size: 224 × 224 pixels
- Optimiser: Adam
- Batch size: 32
- Maximum epochs: 20
- Early stopping and model checkpointing
- Training environment: Google Colab with NVIDIA T4 GPU

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Robustness Evaluation

Model robustness was evaluated using three image perturbation conditions:

- Low-light
- Gaussian blur
- Rotation

Classification accuracy was compared between the original test set and each perturbed test set to investigate how different models responded to changes in image quality and orientation.

## Error Analysis

Cross-model error analysis was performed to identify species and confusion pairs that consistently caused classification errors across different architectures.

The analysis includes:

- Species-level misclassification frequency
- Cross-model confusion pairs
- Model-specific prediction results
- EfficientNet-B0 confusion matrix

EfficientNet-B0 was selected for detailed analysis because it achieved the highest overall classification accuracy.

## Grad-CAM Analysis

Grad-CAM was used to investigate the image regions contributing to EfficientNet-B0 predictions.

Two types of analysis were performed:

1. Grad-CAM analysis of misclassified samples.
2. Grad-CAM comparison between original and low-light images.

These analyses were used to investigate whether the model relied primarily on morphological features such as wings and body structures, and whether contextual information influenced some predictions.

## Main Results

EfficientNet-B0 achieved the highest overall classification accuracy among the evaluated models.

| Model | Accuracy |
|---|---:|
| Baseline CNN | 77% |
| CNN + Data Augmentation | 88% |
| ResNet50 | 96% |
| EfficientNet-B0 | 97% |
| ViT | 94% |

EfficientNet-B0 also had substantially fewer parameters than the other transfer learning models, with approximately 4.08 million parameters.

Under robustness evaluation, EfficientNet-B0 maintained strong performance across the tested image conditions. It achieved 98% accuracy under low-light conditions, 92% under Gaussian blur and 97% under rotation.

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
├── error_analysis/
│   ├── CNN_predictions.csv
│   ├── CNNAug_predictions.csv
│   ├── ResNet50_predictions.csv
│   ├── EfficientNetB0_predictions.csv
│   ├── ViT_predictions.csv
│   ├── CrossModel_ConfusionPairs.csv
│   └── Species_Error_Frequency.csv
│
└── videos/
