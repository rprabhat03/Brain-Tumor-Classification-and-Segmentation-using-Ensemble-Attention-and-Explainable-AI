# 📁 01 - Data Processing & Preparation

This section explains how raw MRI data is transformed into a format that the AI models can "understand" and learn from.

## 1. Dataset Overview
The project uses a combined and cleaned dataset of brain MRI scans. 
- **Classification Dataset**: ~3,200 images across 4 classes (Glioma, Meningioma, Pituitary, No Tumor).
- **Segmentation Dataset**: 3,064 MRI scans with paired "binary masks" (ground truth).

## 2. Preprocessing Pipeline
To ensure high accuracy, every image goes through a strict cleaning process:

### A. Resizing & Normalization
- All images are resized to **224x224** or **299x299** pixels (depending on the model's requirements).
- We use **Min-Max Scaling** (dividing by 255.0) to keep pixel values between 0 and 1. This helps the model's "neurons" learn much faster and prevents mathematical overflow.

### B. Image Augmentation
Since medical datasets are small, we "artificially" create more data through **Augmentation**. This prevents the model from "memorizing" specific images (overfitting):
- **Rotation**: Randomly rotating the MRI (e.g., within 20 degrees).
- **Flipping**: Horizontal and Vertical reflections of the brain.
- **Brightness/Contrast**: Adjusting lighting to simulate different MRI machine settings.
- **Shearing & Zooming**: Changing the perspective slightly.

## 3. Data Split
We split the data into 3 distinct groups:
1. **Training (70-80%)**: The "study material" the AI uses to learn.
2. **Validation (10-15%)**: A "practice test" the AI uses during training to check its progress.
3. **Testing (10-15%)**: The "final exam"—images the AI has *never* seen before, used to calculate the final accuracy scores.

## 4. Class Imbalance Handling
In medical data, some tumor types are rarer than others. We use **Batch Shuffling** and **Data Augmentation** to ensure the model doesn't become biased toward the most common class.
