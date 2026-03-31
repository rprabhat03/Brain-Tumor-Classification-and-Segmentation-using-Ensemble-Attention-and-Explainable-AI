"""
Flask Web Application for Brain Tumor Classification & Segmentation.
Upload an MRI image → Get classification, segmentation, and GradCAM results.
"""

import os
# Silence TensorFlow logs for a cleaner terminal presentation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

import sys
import numpy as np
from PIL import Image
import io
import base64
import json

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add parent directory to path for utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_DESCRIPTIONS = {
    'Glioma': 'A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glial cells that surround and support nerve cells.',
    'Meningioma': 'A tumor that arises from the meninges — the membranes that surround the brain and spinal cord. Most meningiomas are benign.',
    'No Tumor': 'No tumor detected in the MRI scan. The brain appears normal.',
    'Pituitary': 'A tumor that forms in the pituitary gland at the base of the brain. Pituitary tumors are generally benign.'
}
CLASS_SEVERITY = {
    'Glioma': 'High',
    'Meningioma': 'Medium',
    'No Tumor': 'None',
    'Pituitary': 'Low-Medium'
}

# Global model variables
classification_model = None
classification_model_name = None
segmentation_model = None


def load_models():
    """Load classification and segmentation models."""
    global classification_model, classification_model_name, segmentation_model

    # Try to load classification model (try multiple models)
    model_priority = [
        'xception_best.h5', 'efficientnetb0_best.h5', 'resnet50_best.h5',
        'densenet121_best.h5', 'vgg16_best.h5', 'mobilenetv2_best.h5', 'custom_cnn_best.h5'
    ]

    for model_file in model_priority:
        model_path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(model_path):
            try:
                classification_model = tf.keras.models.load_model(model_path)
                classification_model_name = model_file
                print(f"[OK] Classification model loaded: {model_file}")
                break
            except Exception as e:
                print(f"[WARN] Failed to load {model_file}: {e}")

    if classification_model is None:
        print("[WARN] No classification model found. Run training notebooks first.")

    # Try to load segmentation model (Attention U-Net preferred, fallback to U-Net)
    from utils.model_loader import load_trained_model
    seg_candidates = ['attention_unet_best.h5', 'unet_best.h5']
    for seg_file in seg_candidates:
        seg_path = os.path.join(MODEL_DIR, seg_file)
        if os.path.exists(seg_path):
            try:
                segmentation_model = load_trained_model(seg_path)
                print(f"[OK] Segmentation model loaded: {seg_file}")
                break
            except Exception as e:
                print(f"[WARN] Failed to load {seg_file}: {e}")
    else:
        print("[WARN] No segmentation model found. Run segmentation notebook first.")


def check_is_valid_mri(img_array):
    """
    Heuristic check to determine if the uploaded image is likely a brain MRI scan.
    - MRI scans are generally grayscale (RGB channels are very similar).
    - MRI scans usually have a large dark background.
    """
    # 1. Check if image is mostly grayscale
    channel_std = np.std(img_array, axis=2)
    mean_color_deviation = np.mean(channel_std)
    is_grayscale = mean_color_deviation < 15.0
    
    # 2. Check for dark background
    gray_img = np.mean(img_array, axis=2)
    dark_pixels_ratio = np.sum(gray_img < 25) / gray_img.size
    has_background = dark_pixels_ratio > 0.05
    
    if not is_grayscale:
        return False, "Image appears to be in color. Please upload a strictly grayscale MRI scan."
    
    if not has_background:
        return False, "Image lacks the typical dark background of an MRI scan."
        
    return True, "Valid"


def image_to_base64(img_array):
    """Convert numpy array to base64 encoded image."""
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Process uploaded image and return predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess image
        img = Image.open(file.stream).convert('RGB')
        original_img = np.array(img)

        # Validate if image is an MRI scan
        is_valid, reason = check_is_valid_mri(original_img)
        if not is_valid:
            return jsonify({'error': f'Invalid Image: {reason}'}), 400

        result = {
            'original': image_to_base64(original_img),
            'classification': None,
            'segmentation': None,
            'gradcam': None,
        }

        # Classification
        if classification_model is not None:
            cls_size = classification_model.input_shape[1]
            img_cls = img.resize((cls_size, cls_size))
            img_cls_array = np.array(img_cls).astype(np.float32)
            img_cls_batch = np.expand_dims(img_cls_array, axis=0)

            # Apply correct preprocessing based on the loaded model
            if 'xception' in classification_model_name:
                img_cls_batch = tf.keras.applications.xception.preprocess_input(img_cls_batch)
            elif 'resnet50' in classification_model_name:
                img_cls_batch = tf.keras.applications.resnet50.preprocess_input(img_cls_batch)
            elif 'densenet' in classification_model_name:
                img_cls_batch = tf.keras.applications.densenet.preprocess_input(img_cls_batch)
            elif 'vgg16' in classification_model_name:
                img_cls_batch = tf.keras.applications.vgg16.preprocess_input(img_cls_batch)
            elif 'mobilenet' in classification_model_name:
                img_cls_batch = tf.keras.applications.mobilenet_v2.preprocess_input(img_cls_batch)
            else:
                img_cls_batch /= 255.0  # Fallback for custom CNN or EfficientNet (which expects 0-255 or handles it internally)

            # We also need the display image (0-1 range) for overlaying GradCAM
            img_cls_display = np.array(img_cls) / 255.0

            predictions = classification_model.predict(img_cls_batch, verbose=0)
            pred_class = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][pred_class])

            result['classification'] = {
                'class': CLASS_NAMES[pred_class],
                'confidence': float(round(confidence * 100, 1)),
                'probabilities': {
                    CLASS_NAMES[i]: float(round(predictions[0][i] * 100, 1))
                    for i in range(len(CLASS_NAMES))
                },
                'description': CLASS_DESCRIPTIONS[CLASS_NAMES[pred_class]],
                'severity': CLASS_SEVERITY[CLASS_NAMES[pred_class]],
            }

            # Only show GradCAM and Segmentation if a tumor is detected
            if CLASS_NAMES[pred_class] != 'No Tumor':
                # GradCAM
                try:
                    from utils.gradcam import make_gradcam_heatmap, overlay_heatmap
                    heatmap = make_gradcam_heatmap(img_cls_batch, classification_model)
                    overlay = overlay_heatmap(img_cls_array, heatmap, alpha=0.55) # increased alpha for stronger popup
                    result['gradcam'] = {
                        'heatmap': image_to_base64(
                            (plt_colormap(heatmap, img_cls_array, size=img_cls_array.shape[1]) * 255).astype(np.uint8)
                        ),
                        'overlay': image_to_base64(overlay),
                    }
                except Exception as e:
                    import traceback
                    print(f"GradCAM error: {e}")
                    traceback.print_exc()

            # Segmentation (only if tumor detected)
            if segmentation_model is not None and CLASS_NAMES[pred_class] != 'No Tumor':
                seg_size = segmentation_model.input_shape[1]
                img_seg = img.resize((seg_size, seg_size))
                img_seg_array = np.array(img_seg) / 255.0
                img_seg_batch = np.expand_dims(img_seg_array, axis=0)

                seg_pred = segmentation_model.predict(img_seg_batch, verbose=0)
                seg_mask = (seg_pred[0].squeeze() > 0.5).astype(np.float32)

                # Create colored mask overlay
                overlay_seg = img_seg_array.copy()
                mask_colored = np.zeros_like(overlay_seg)
                mask_colored[:, :, 0] = seg_mask  # Red channel
                overlay_seg = overlay_seg * 0.7 + mask_colored * 0.3

                # Calculate tumor area percentage
                tumor_area = (seg_mask.sum() / seg_mask.size) * 100

                result['segmentation'] = {
                    'mask': image_to_base64(
                        (seg_mask * 255).astype(np.uint8)
                    ),
                    'overlay': image_to_base64(overlay_seg),
                    'tumor_area_percent': float(round(tumor_area, 1)),
                }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def plt_colormap(heatmap, img_array, size=224):
    """Apply matplotlib colormap to heatmap smoothly, masking background space."""
    import matplotlib.cm as cm
    import cv2
    
    # Normalize img_array to uint8 for mask creation
    if img_array.max() <= 1.0:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)
    
    # Resize heatmap to image dimensions first for correct masking
    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    
    # Medical mask to kill corners
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY) if img_uint8.ndim == 3 else img_uint8
    _, brain_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    brain_mask = cv2.erode(brain_mask, np.ones((11, 11), np.uint8), iterations=1)
    brain_mask_float = (brain_mask / 255.0).astype(np.float32)
    
    # Zero out heatmap outside brain
    heatmap_masked = heatmap_resized * brain_mask_float
    
    # Resize to output size
    heatmap_final = cv2.resize(heatmap_masked, (size, size), interpolation=cv2.INTER_CUBIC)
    heatmap_final = np.clip(heatmap_final, 0.0, 1.0)
    
    colored = cm.jet(heatmap_final)[:, :, :3]  # Remove alpha channel
    return colored


if __name__ == '__main__':
    print("=" * 60)
    print("Brain Tumor Analysis Web Application")
    print("=" * 60)
    load_models()
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop.\n")

    # Import matplotlib for colormap (needed for GradCAM)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    app.run(debug=True, host='0.0.0.0', port=5000)
