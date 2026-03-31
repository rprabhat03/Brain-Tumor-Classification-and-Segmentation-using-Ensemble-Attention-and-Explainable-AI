# 🔬 03 - Segmentation & AI Explainability

How the system highlights the tumor and "explains" its own reasoning using **GradCAM** and **Attention U-Net**.

## 1. What is AI Explainability (XAI)?
AI is often a "black box" where we don't know why it made a choice. We use **XAI** so the doctor can trust the AI by seeing *what it's looking at*.

## 2. GradCAM Heatmap (The "Why")
**GradCAM** (Gradient-weighted Class Activation Mapping) is an algorithm that visualizes the "attention" of the **Xception** classifier. 
- **Heatmap**: Red areas are "high attention," Blue areas are ignored.
- **Goal**: If the classifier says "Glioma," the red spot should be right on top of the actual tumor.
- **How it works**: It looks at the very last convolutional layer's gradients to find out which neurons were most active for a specific class.

## 3. Attention U-Net (The "Where")
While GradCAM is a "blurry" heatmap, the **Attention U-Net** is a razor-sharp "precision tool."
- **Standard U-Net**: Takes the original image, "shrinks" it to learn features, then "inflates" it back to create a mask.
- **Attention Gates (scSE)**: We added **Spatial and Channel Squeeze-and-Excitation** (scSE) blocks. 
    1. **Spatial Attention**: Learns *where* the tumor is located spatially.
    2. **Channel Attention**: Learns *which* feature maps (like edge-filters or texture-filters) are most important.
- **The Result**: A clean, binary mask that perfectly outlines the tumor.

## 4. Dice & IoU Metrics
Standard accuracy is bad for segmentation because 99% of a brain in an MRI is healthy tissue. We use:
- **Dice Coefficient**: Measures the overlap between the AI's mask and the Ground Truth mask.
- **IoU (Intersection-over-Union)**: Calculates the area of overlap divided by the total area covered by both masks.
