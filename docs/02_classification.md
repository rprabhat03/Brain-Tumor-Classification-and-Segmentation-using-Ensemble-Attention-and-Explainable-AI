# 🧠 02 - Brain Tumor Classification

This section describes how the system identifies the *type* of brain tumor from an MRI.

## 1. Transfer Learning Model Suite
We don't build a single model—we use a "Model Zoo" of 10+ state-of-the-art architectures to find the best performer:
- **Xception (The Best)**: Excellent balance between speed and accuracy. 
- **ResNet (50 & 101)**: Uses "Skip Connections" to train deeper without losing data.
- **InceptionV3**: Uses "Inception Blocks" to scan at different zoom levels.
- **VGG (16 & 19)**: Simple and robust, great for standard MRI features.
- **DenseNet121**: Every layer is connected to every other layer!
- **MobileNetV2**: Super-fast, designed for mobile/web devices.

## 2. Training Technique (Fine-Tuning)
Instead of starting from zero, we use **Pretrained ImageNet Weights**. 
1. **The Base**: The models were already trained on millions of general images.
2. **The Head**: We replaced the original "output layer" with our custom **4-Class Dense Head** (Glioma, Meningioma, No Tumor, Pituitary).
3. **The Freeze**: For the first few hours, we "freeze" the base so only our new head learns.
4. **The Melt**: Finally, we "unfreeze" the whole model to perfectly adapt to subtle MRI textures.

## 3. Advanced Learning Features
- **Categorical Cross-Entropy**: Used to measure how "wrong" the model's guess is.
- **Adam Optimizer**: A "smart" math engine that adjusts the model's learning speed automatically.
- **Learning Rate Decay**: Slows down the learning speed as it gets closer to the solution—like a car slowing down for a parking spot.

## 4. Evaluation Metrics
We don't just use **Accuracy**. We also measure:
- **Precision**: Does it give false alarms?
- **Recall**: Does it miss any tumors?
- **AUC (Area Under Curve)**: How well it can separate one tumor type from another.
