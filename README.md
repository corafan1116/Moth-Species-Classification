# Moth Species Classification

Deep learning models for automated moth species classification, robustness evaluation and error analysis.

## Project Overview

This project investigates deep learning approaches for automated classification of 20 moth species.

Five models are evaluated within a unified experimental framework:

- Baseline CNN
- CNN with data augmentation
- ResNet50
- EfficientNet-B0
- Vision Transformer (ViT)

The project also evaluates model robustness under different image conditions, including low-light, Gaussian blur and rotation, together with cross-model error analysis and Grad-CAM-based visualisation.

## Repository Structure

- `notebooks/` — Training, evaluation and analysis notebooks
- `error_analysis/` — Prediction results and error analysis data
- `video/` — Project demonstration video
- `.gitignore` — Git ignore configuration
- `README.md` — Project documentation

## Environment

- Python 3.10
- TensorFlow 2.16.2
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

The models were trained using Google Colab with an NVIDIA T4 GPU.

## Dataset

The project uses a subset of 20 moth species from the publicly available Butterflies and Moths Species Classification Dataset on Kaggle.

The original dataset is not included in this repository.
