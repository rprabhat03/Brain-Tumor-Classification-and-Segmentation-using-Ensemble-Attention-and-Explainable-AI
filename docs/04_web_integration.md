# 🌐 04 - Web Integration (Backend & Frontend)

How the **Flask Web App** connects all our AI models into a single, user-friendly interface.

## 1. The Technology Stack
- **Backend**: **Flask** (Python) — Fast, light, and easy to link with TensorFlow.
- **Frontend**: **HTML5**, **Vanilla CSS**, **JavaScript (ES6)** — A premium, "glassmorphic" Dark Mode design.
- **AI Framework**: **TensorFlow 2.x** — The core of all calculations.

## 2. The Backend (Flask)
The `app.py` is the "brain" of the website:
- **`model_loader.py`**: A special tool that "detects" which models are in your `models/` folder and loads them into memory.
- **The `/predict` Route**:
    1. Receives an uploaded image from the browser.
    2. Runs it through **Xception** (Classification).
    3. Runs it through **Attention U-Net** (Segmentation).
    4. Generates a **GradCAM Heatmap** explaining the result.
    5. Packages all the data (Base64 images and JSON numbers) and sends it back to the frontend.

## 3. The Frontend (JavaScript)
The `main.js` handles the "User Experience" (UX):
- **Drag-and-Drop**: Users can quickly drag their MRI scans onto the "Drop Zone."
- **Real-Time Display**: It uses **AJAX (Fetch API)** to get the AI results *without* refreshing the whole page.
- **Dynamic Layout**: The website only "unlocks" and shows the Segmentation and GradCAM sections once the classification is ready.
- **Glassmorphic Design**: A premium, modern look that emphasizes a medical/high-tech atmosphere.

## 4. API Response (The "Contract")
The Backend and Frontend talk using a **JSON Bridge**:
- **`original`**: The image the user uploaded.
- **`classification`**: Class name (Glioma/Meningioma/etc.) and Confidence (e.g., 98.4%).
- **`segmentation`**: The binary mask and the "Tumor Overlay."
- **`gradcam`**: The "Atmospheric" heatmap overlay.
- **`tumor_area_percent`**: A calculation of how much space the tumor occupies in the MRI slice.
