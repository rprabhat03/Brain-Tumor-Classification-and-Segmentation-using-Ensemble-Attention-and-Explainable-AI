# 🧠 NeuroVision AI  
## Brain Tumor Classification and Segmentation using Ensemble Attention and Explainable AI

---

## 📌 Project Overview
NeuroVision AI is an end-to-end deep learning–based system designed for **brain tumor classification and segmentation from MRI images**.  
The project integrates **ensemble CNN models**, **attention mechanisms**, and **Explainable AI (Grad-CAM)** to provide accurate, interpretable, and efficient medical image analysis.

This work is developed as a **Minor Project** under faculty supervision and is intended for **academic and research purposes**.

---

## 🎯 Project Objectives
- Classify brain tumors into multiple categories using MRI scans
- Segment tumor regions from MRI images using a lightweight model
- Improve classification robustness using ensemble learning
- Provide explainability through Grad-CAM visualizations
- Develop a web-based interface for easy interaction
- Explore feasibility of **edge-AI deployment** for offline medical inference

---

## 📂 Datasets Used

### 1️⃣ Brain Tumor Classification Dataset
- **Path (used in Colab):**  
  `/content/drive/MyDrive/BrainTumor`
- **Classes:**
  - glioma_tumor  
  - meningioma_tumor  
  - no_tumor  
  - pituitary_tumor
- **Data Split:**
  - Training images: 2,875  
  - Testing images: 394
- **Data Type:** 2D MRI images

---

### 2️⃣ Brain Tumor Segmentation Dataset
- **Path (used in Colab):**  
  `/content/drive/MyDrive/BrainTumor/BrainTumorSegmentation`
- **Contents:**
  - `images/` – MRI images  
  - `masks/` – Ground truth tumor masks
- **Total Samples:** ~3,064 image–mask pairs
- **Task:** Binary tumor segmentation

⚠️ *Datasets are not included in this repository due to size constraints.*

---

## 🧠 Model Architectures

### 🔹 Classification (Ensemble Model)
- **Base Models:**
  - EfficientNetV2B0 (ImageNet pretrained)
  - MobileNetV2 (ImageNet pretrained)
- **Enhancements:**
  - Attention Block
  - Global Average Pooling
  - Softmax output layer (4 classes)
- **Ensemble Strategy:**
  - Simple averaging of predicted probabilities

**Training Details:**
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Batch Size: 16  
- Epochs: 10  

---

### 🔹 Segmentation Model
- **Architecture:** Mini U-Net (lightweight)
- **Input:** 128 × 128 × 1 grayscale MRI image
- **Output:** Binary tumor mask
- **Loss Function:** Binary Crossentropy
- **Epochs:** 5

---

### 🔹 Explainable AI
- **Technique Used:** Grad-CAM
- **Model Applied:** EfficientNetV2B0
- **Purpose:**  
  Highlight regions in MRI images that influence classification decisions, improving transparency and trust.

---

## 🌐 Web Application
The project includes a **web-based interface** that allows users to:
- Upload MRI images
- View tumor classification results with confidence scores
- Visualize Grad-CAM heatmaps
- View tumor segmentation masks and overlays

---

## 🧪 Results
- **Classification Accuracy:** 78%
- **Segmentation Pixel-wise Accuracy:** ~98%
- Grad-CAM visualizations confirm focus on tumor-relevant regions
- Some misclassification observed between glioma and meningioma due to visual similarity

---

## 🧩 Hardware Deployment (Conceptual)
- Designed for deployment on **edge devices** such as:
  - Raspberry Pi
  - NVIDIA Jetson Nano
- Enables:
  - Offline inference
  - Low-latency medical analysis
  - Privacy-preserving computation
- Training is performed on cloud/high-performance systems, while inference is intended for embedded hardware

---

## 👥 Team Members
- **Pratyush Kumar** (2330100)  
- **Harsh Raj** (2330162)  
- **Prabhat Ranjan** (2330384)  
- **Simran Jena** (2330407)

---

## 👩‍🏫 Supervisor
**Prof. Tejaswini Kar**  
School of Electronics Engineering  
KIIT Deemed to be University, Bhubaneswar

---

## ⚠️ Disclaimer
This project is intended **only for academic and research purposes** and is **not a clinical diagnostic tool**.

---

## 📜 License
This repository is shared for educational use under an academic open-source license.
