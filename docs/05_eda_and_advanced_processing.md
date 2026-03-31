# 🔍 05 - In-Depth EDA & Advanced Processing

This guide provides a deep technical analysis of the Exploratory Data Analysis (EDA) and the rigorous data cleaning process used to create the high-quality dataset for NeuroScan AI.

## 1. The "Big Data" Challenge
Our project merged three separate Kaggle datasets to create a robust, generalized model. However, merging datasets from different sources introduces **Noise**, **Duplicates**, and **Inconsistencies**.

### The Sources:
1. **Sartaj Dataset**: Good variety but smaller resolution.
2. **Masoud Dataset**: High quality, but contains some overlapping images.
3. **Br35H Dataset**: Specialized in detection (Yes/No), utilized for our "No Tumor" class.

## 2. Advanced Data Cleaning: Perceptual Hashing
When merging datasets, the same MRI image might appear in two different sources with slightly different brightness or filenames. Standard "File Hash" checks fail here.
- **Solution**: We employed **Perceptual Hashing (pHash)**. 
- **How it works**: It creates a "fingerprint" of the image based on its structure rather than its raw pixels. 
- **Result**: We identified and removed **~1,200 near-duplicate images**, ensuring our testing set is truly unique.

## 3. Pixel Intensity "Signatures"
Through our EDA (Notebook `01_eda.ipynb`), we discovered that each tumor type has a unique pixel intensity histogram:
- **Gliomas**: Often show a "bi-modal" distribution with spikes in middle-gray values (representing the complex, mixed-density tissue).
- **Pituitary Tumors**: Frequently show highndensity "clumps" because these tumors are often very compact and bright on T1-weighted MRIs.
- **No Tumor**: The histogram is shifted toward the left (darker), representing the healthy, consistent brain matter without "hotspots."

## 4. Dimensional Analysis & Standarization
Our raw images varied from **128x128** to **1024x1024**. 
- **Insight**: Small images can't be "upscaled" without losing sharpness, and huge images are too slow to train.
- **Decision**: We standardized everything to **224x224 (for the CNN heads)** and **299x299 (for Xception)** using **Anti-Aliasing Interpolation** to preserve the thin edges of the brain structure.

## 5. Class Balancing & Stratification
A major "Gotcha" in medical AI is **Class Imbalance**. If you have 10,000 "No Tumor" images and only 100 "Glioma" images, the AI will just cheat and guess "No Tumor" every time.
- **Our Strategy**: 
    1. **Under-sampling**: We capped the largest class to prevent dominance.
    2. **Stratified Splitting**: We used `StratifiedShuffleSplit` to ensure that if 25% of your total data is "Meningioma," then **exactly 25%** of your training, validation, and testing sets are also "Meningioma."

## 6. Augmentation: Simulating Medical Variety
Mris look different depending on the hospital's machine (Siemens vs. GE vs. Philips). We used **Advanced Data Augmentation** to simulate this:
- **Brightness Range [0.8, 1.2]**: Simulates different exposure levels.
- **Shear & Zoom**: Simulates slight patient movement or different head tilt angles in the scanner.
- **Horizontal Flipping**: Reflects the symmetrical nature of the brain.
