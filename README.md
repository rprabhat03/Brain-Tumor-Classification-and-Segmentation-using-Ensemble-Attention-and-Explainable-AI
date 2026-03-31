🧠 Brain Tumor Classification & Segmentation using Explainable AI

A deep learning project that detects, classifies, and localizes brain tumors from MRI scans — with visual explanations to make the model more trustworthy.

🚀 What this project does

This project is built to answer three important questions:

Is there a tumor?
What type of tumor is it?
Where exactly is it located?

Instead of just giving predictions, it also explains why the model made a decision using heatmaps.

📁 Project Structure
├── docs/          # Project report, documentation
├── notebooks/     # Training & experimentation notebooks
├── utils/         # Helper functions (preprocessing, GradCAM, etc.)
├── webapp/        # Flask web application
├── requirements.txt
├── .gitignore
└── README.md
🧩 Key Features
🧠 Multi-class Classification
Glioma
Meningioma
Pituitary
No Tumor
🔥 Explainable AI (Grad-CAM)
Highlights important regions in MRI
🎯 Tumor Segmentation
Pixel-level detection using U-Net
⚡ Multiple Models Compared
Xception (best accuracy)
MobileNetV2 (best lightweight model)
ResNet, DenseNet, VGG, EfficientNet
🌐 Web App Interface
Upload MRI → Get prediction + heatmap + segmentation
📊 Results (Quick Summary)
🥇 Best Model: Xception
🎯 Accuracy: ~96.97%
📈 AUC: ~0.99
⚡ Best for Deployment: MobileNetV2 (fast + efficient)
🗂️ Dataset
~15,000 MRI images (combined from multiple sources)
Cleaned using duplicate removal (pHash)
Balanced across 4 classes
⚙️ Tech Stack
Deep Learning: TensorFlow, Keras
Image Processing: OpenCV
Backend: Flask
Data Tools: NumPy, Pandas
Visualization: Matplotlib
▶️ How to Run the Project
1. Clone the repo
git clone https://github.com/rprabhat03/Brain-Tumor-Classification-and-Segmentation-using-Ensemble-Attention-and-Explainable-AI.git
cd Brain-Tumor-Classification-and-Segmentation-using-Ensemble-Attention-and-Explainable-AI
2. Install dependencies
pip install -r requirements.txt
3. Run the web app
cd webapp
python app.py
4. Open in browser
http://127.0.0.1:5000/
🧠 How it Works (Simple Flow)
Upload MRI image
Preprocessing (resize, normalize)
Model predicts tumor class
Grad-CAM generates heatmap
U-Net segments tumor
Results displayed on UI
⚠️ Disclaimer

This project is for educational and research purposes only.
It is not a medical diagnostic tool.

👨‍💻 Team
Pratyush Kumar
Harsh Raj
Prabhat Ranjan
Simran Jena
🔮 Future Scope
Ensemble attention models (higher accuracy)
3D MRI analysis
Better explainability methods
Cloud deployment
