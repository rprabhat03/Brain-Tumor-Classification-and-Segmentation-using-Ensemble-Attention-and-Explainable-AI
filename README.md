# 🧠 Brain Tumor Classification & Segmentation using Explainable AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.13-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Keras-Integrated-red?style=for-the-badge&logo=keras" />
  <img src="https://img.shields.io/badge/Flask-Web%20App-green?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

> A comprehensive deep learning pipeline for automatic classification and pixel-level segmentation of brain tumors from MRI scans, augmented with GradCAM-based explainability and deployed as a Flask web application.

**B.Tech Final Year Project — Electronics and Computer Science Engineering**
**KIIT (Deemed to be University), Bhubaneswar | March 2026**

**Team:** Pratyush Kumar · Harsh Raj · Prabhat Ranjan · Simran Jena
**Supervisor:** Prof. Tejaswani Kar, School of Electronics Engineering

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Explainability — GradCAM](#explainability--gradcam)
- [Segmentation — Attention U-Net](#segmentation--attention-u-net)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Technology Stack](#technology-stack)
- [Challenges & Solutions](#challenges--solutions)
- [Future Scope](#future-scope)
- [References](#references)
- [Disclaimer](#disclaimer)

---

## Overview

Brain tumors are among the most life-threatening neurological conditions globally. Accurate and early diagnosis is critical — yet traditional radiological interpretation is time-consuming, expert-dependent, and prone to inter-observer variability. The radiologist-to-population ratio in countries like India stands at approximately 1:100,000, far below WHO recommendations.

This project presents an end-to-end AI-powered pipeline that:

1. **Classifies** brain MRI scans into four tumor categories using nine transfer learning architectures.
2. **Segments** tumor regions at the pixel level using an Attention U-Net with EfficientNetB3 encoder.
3. **Explains** model predictions with GradCAM heatmaps, highlighting anatomically relevant attention regions.
4. **Deploys** everything in an interactive Flask web application with a premium dark-themed UI.

---

## Key Results

| Task | Best Model | Metric | Score |
|---|---|---|---|
| **Classification** | Xception | Test Accuracy | **96.97%** |
| **Classification** | Xception | AUC | **0.9917** |
| **Classification (Efficient)** | MobileNetV2 | Accuracy / Params | **96.07% / 3.0M** |
| **Segmentation** | Attention U-Net + EfficientNetB3 | Mean Dice Score | **0.8282** |
| **Segmentation** | Attention U-Net + EfficientNetB3 | IoU Score | **~0.75** |

**Per-class AUC (Xception):** Glioma 0.997 · Meningioma 0.994 · No Tumor 0.999 · Pituitary 0.998

---

## Dataset

### Sources
A large-scale, deduplicated dataset of **~15,000 MRI images** was compiled from three publicly available Kaggle repositories:

| Dataset | Kaggle Source | Images | Notes |
|---|---|---|---|
| **Dataset A** | `sartajbhuvaji/brain-tumor-classification-mri` | ~3,264 | Standardised 4-class; uniform contrast. Used as deduplication baseline. |
| **Dataset B** | `masoudnickparvar/brain-tumor-mri-dataset` | ~7,153 | Axial, coronal & sagittal views; high variance in contrast. |
| **Dataset C** | `ahmedhamada0/brain-tumor-detection` (Br35H) | ~3,060 | Binary origin; augments the No Tumor class. |
| **Merged (Post-Dedup)** | — | **~15,000** | Unique, non-overlapping scans across 4 classes. |

**Classes:** Glioma · Meningioma · No Tumor · Pituitary

### Deduplication Pipeline
Naive concatenation causes severe data leakage (identical images in train and test sets). To prevent this:
- **Perceptual Hashing (pHash):** 16-bit pHash via the `imagehash` library detects near-duplicates even after minor compression or resizing — something MD5 misses.
- Dataset A was the baseline hash set. Images from B and C were purged if their Hamming Distance was zero against any existing hash.

### Preprocessing
| Step | Details |
|---|---|
| Spatial Normalisation | Resize to 224×224 (CUBIC); 299×299 for InceptionV3/Xception |
| Channel Expansion | Grayscale → 3-channel RGB for ImageNet-pretrained models |
| Brain-Region Cropping | OpenCV contour detection to remove skull base / background |
| Pixel Normalisation | Rescale to [0, 1]; model-specific preprocessing applied |
| Augmentation (train only) | Horizontal flip, rotation ±20°, zoom ±10%, brightness ±20%, shear |
| Dataset Split | Stratified 70% train / 15% validation / 15% test |

### Segmentation Dataset
A separate dataset of **3,064 MRI–mask pairs** (`nikhilroxtomar/brain-tumor-segmentation`) was used exclusively for the U-Net segmentation task. Images normalised to [0, 1]; masks binarized at threshold 0.5.

---

## Architecture

### Classification — Transfer Learning (9 Models)
All nine models use **ImageNet pre-trained weights** with a custom classification head:

```
Backbone (frozen) → Global Average Pooling → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)
```

**Two-phase training:**
- **Phase 1:** Frozen backbone, Adam (lr=1e-3), convergence of classification head.
- **Phase 2:** Partial backbone unfreezing, Adam (lr=1e-5), fine-tuning.

**Training config:** Batch size 32 · Max 30 epochs · Early stopping (patience=5) · ReduceLROnPlateau

### Segmentation — Attention U-Net + EfficientNetB3
- **Encoder:** EfficientNetB3 (ImageNet pre-trained) replacing the standard U-Net encoder for richer hierarchical feature extraction.
- **Skip Connections:** Feature maps from EfficientNetB3 stages (strides 2, 4, 8, 16, 32) fed into decoder levels.
- **Attention Gates:** Squeeze-and-Channel Excitation (scSE) gates recalibrate channel-wise and spatial feature responses, suppressing background and enhancing tumor sensitivity.
- **Bottleneck:** 1,024 filters.
- **Decoder:** Transposed convolutions + attention-gated skip connections.
- **Loss Function:** Combined Binary Cross-Entropy + Dice Loss (handles class imbalance).
- **Training:** 80 epochs on 3,064 image-mask pairs.

---

## Model Performance

Full comparison on the held-out test set (ranked by accuracy):

| Rank | Model | Params | Accuracy | Precision | Recall | F1 | AUC | Train Time | Epochs |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **Xception** | 22.0M | **96.97%** | 97.10% | 96.90% | 96.96% | **0.9917** | 3,735s | 24 |
| 🥈 2 | **MobileNetV2** | 3.0M | 96.07% | 96.20% | 95.93% | 96.07% | 0.9911 | 2,286s | 30 |
| 🥉 3 | DenseNet121 | 7.7M | 94.41% | 94.61% | 94.34% | 94.38% | 0.9898 | 2,234s | 29 |
| 4 | EfficientNetB0 | 4.8M | 86.76% | — | — | — | 0.985–0.986 | 2,134s | 27 |
| 5 | ResNet50 | 24.1M | 89.31% | — | — | — | 0.961–0.996 | 1,662s | 20 |
| 6 | ResNet101 | 43.2M | 88.34% | — | — | — | 0.950–0.995 | 1,568s | 20 |
| 7 | InceptionV3 | 22.3M | 88.34% | — | — | — | 0.959–0.996 | 2,872s | 20 |
| 8 | VGG16 | 138M | 84.62% | — | — | — | — | — | — |
| 9 | VGG19 | 143M | 81.52% | — | — | — | — | — | — |

**Key takeaway:** MobileNetV2 achieves 96.07% accuracy with only 3.0M parameters — a **7.3× reduction** in model size vs. Xception for a marginal 0.9% accuracy drop, making it ideal for deployment.

---

## Explainability — GradCAM

GradCAM (Gradient-weighted Class Activation Mapping) was applied to the best-performing **Xception** model. For each input:
1. Gradient of the class score w.r.t. the last convolutional layer's feature maps is computed.
2. Global average pooling produces per-channel importance weights.
3. A weighted activation map is generated, ReLU-applied, resized, and overlaid on the original MRI using a jet colormap.

**GradCAM results confirm anatomically correct attention across all four tumor classes:**

| Class | Prediction Confidence |
|---|---|
| Glioma | 99.85% |
| Meningioma | 100.00% |
| No Tumor | 99.97% |
| Pituitary | 99.74% |

This bridges the "black-box" gap of CNNs and is critical for clinical trust and adoption.

---

## Segmentation — Attention U-Net

**Training results over 80 epochs:**
- Dice Coefficient: ~0.10 (epoch 1) → **~0.75** (epoch 80)
- IoU Score: converges to **~0.75** on validation
- BCE + Dice Loss: decreases from 0.58 → below 0.20

**Per-sample Dice scores on test set:**
- Range: 0.456 (large, irregular tumors) → 0.968 (compact, well-defined tumors)
- **Mean Dice: 0.8282**

Predictions accurately delineate tumor boundaries across axial, sagittal, and coronal MRI orientations.

---

## Web Application

A Flask-based web app integrates all three components (classification, GradCAM, segmentation) into a single user-facing interface:

- **UI:** Premium dark-themed glassmorphism-style responsive design
- **Features:**
  - Drag-and-drop MRI upload
  - Real-time classification with confidence scores for all 4 classes
  - GradCAM heatmap generation and overlay
  - U-Net tumor segmentation overlay
- **Backend:** Server-side inference in Python (TensorFlow/Keras), results returned as JSON and rendered via JavaScript

---

## Project Structure

```
Brain-Tumor-Classification-and-Segmentation/
│
├── docs/                        # Project report and documentation
│
├── notebooks/                   # Jupyter notebooks for training & experiments
│   ├── classification/          # One notebook per model (Xception, MobileNetV2, etc.)
│   └── segmentation/            # Attention U-Net training notebook
│
├── utils/                       # Helper modules
│   ├── preprocessing.py         # Brain-region cropping, resize, augmentation
│   ├── gradcam.py               # GradCAM implementation for Xception
│   └── phash_dedup.py           # Perceptual hash deduplication pipeline
│
├── webapp/                      # Flask web application
│   ├── app.py                   # Main Flask application
│   ├── static/                  # CSS, JS, assets
│   └── templates/               # HTML templates
│
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- pip
- (Recommended) GPU with CUDA support for training

### 1. Clone the Repository
```bash
git clone https://github.com/rprabhat03/Brain-Tumor-Classification-and-Segmentation-using-Ensemble-Attention-and-Explainable-AI.git
cd Brain-Tumor-Classification-and-Segmentation-using-Ensemble-Attention-and-Explainable-AI
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Model Weights
> Place the trained `.h5` or `.keras` model files in `webapp/models/` before running the app.
> Model weights are not included in the repo due to file size. Train using the notebooks or contact the authors.

### 5. Run the Web Application
```bash
cd webapp
python app.py
```

Open your browser at **http://127.0.0.1:5000/**

### 6. Training from Scratch (Optional)
Run notebooks in `notebooks/classification/` (one per architecture) and `notebooks/segmentation/` on Google Colab / Kaggle with GPU enabled.

---

## Technology Stack

| Category | Tools |
|---|---|
| **Deep Learning** | TensorFlow 2.13, Keras (Functional & Sequential API) |
| **Computer Vision** | OpenCV 4.8, Pillow (PIL) |
| **Data / EDA** | NumPy 1.24, Pandas 2.0, Matplotlib 3.7, Seaborn |
| **ML Utilities** | Scikit-learn 1.3 |
| **Deduplication** | `imagehash` (perceptual hashing / pHash) |
| **Web Backend** | Flask, Flask-CORS |
| **Training Infra** | Google Colab Pro, Kaggle Notebooks (NVIDIA Tesla T4 GPU) |

---

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| **Dataset heterogeneity & duplicates** across 3 Kaggle sources | 16-bit pHash deduplication pipeline; Hamming Distance = 0 threshold purges near-identical images that MD5 would miss |
| **Class imbalance** (Meningioma is minority class) | Stratified splits + class-weighted augmentation + computed class weights passed to training |
| **Overfitting** in large models (Xception: 13.35M trainable params) | Dropout(0.5), L2 regularisation, early stopping, ReduceLROnPlateau |
| **Clinical trust / black-box CNNs** | GradCAM visualisations confirm anatomically correct attention regions for all 4 tumor classes |

---

## Future Scope

- **Ensemble Attention Models:** Implement Ensemble Co-Attention (MobileNetV3 + EfficientNetB7) as in Celik et al. [1] — projected to push accuracy past 98%.
- **3D MRI Analysis:** Extend the pipeline to volumetric (3D) MRI for richer spatial context.
- **Advanced Explainability:** Integrate SHAP, LIME, or Transformer attention maps.
- **Cloud Deployment:** Deploy on AWS/GCP/Azure with auto-scaling for real-world usage.
- **EfficientNetV2 Backbone:** Replicate Hassan & Ghadiri's 99.16% benchmark with EfficientNetV2b0.
- **DICOM Integration:** Accept raw DICOM files directly, bypassing manual conversion.

---

## References

[1] Celik, G., Celik, F., & Celik, M. — *Ensemble Co-Attention Network for Brain Tumor Classification* (MobileNetV3 + EfficientNetB7; 98.94% on Figshare).

[2] Ilani, R., Shi, J., & Banad, F.M. — *Comparative Study: U-Net, CNN, and Transfer Learning for Brain Tumor Classification* (U-Net: 98.56%; 96.01% cross-dataset).

[3] Hassan, H., & Ghadiri, N. — *EfficientNetV2b0 for Brain Tumor Classification* (99.16% accuracy, 368,804 parameters).

[4] Wang, X., Li, L., & Cheng, X. — *EE-UNet: EfficientNetB4 Encoder + CRF-RNN for Segmentation* (Dice: 0.9624 disc, 0.9228 cup).

[5] Selvaraju, R.R. et al. — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.

---

## Disclaimer

> ⚠️ **This project is for educational and research demonstration purposes only.**
> It is **not a validated medical diagnostic tool** and must not be used for clinical decision-making.
> Any clinical deployment would require formal regulatory approval (e.g., FDA, CE marking), HIPAA/GDPR compliance, and prospective clinical validation.

---

<p align="center">
  Made at KIIT University, Bhubaneswar | March 2026
</p>
