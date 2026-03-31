"""
GradCAM (Gradient-weighted Class Activation Mapping) utilities.
Provides explainability visualizations for brain tumor classification models.
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm


def get_last_conv_layer_name(model):
    """
    Automatically find the last convolutional layer in a model.
    Works with most Keras models including transfer learning architectures.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, 'layers'):
            # For nested models (e.g., base model inside Sequential)
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate a GradCAM heatmap for a given image and model.
    Supports both flat models and nested transfer-learning models.
    """
    # Ensure input is a tensor to prevent Functional model struct mismatches
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # 1. Detect if model contains a nested base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        # --- STANDARD FLAT MODEL EXECUTION ---
        if last_conv_layer_name is None:
            last_conv_layer_name = get_last_conv_layer_name(model)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    else:
        # --- NESTED MODEL EXECUTION (TRANSFER LEARNING) ---
        if last_conv_layer_name is None:
            last_conv_layer_name = get_last_conv_layer_name(base_model)

        # Collect any preprocessing layers before the base model
        pre_layers = []
        for layer in model.layers:
            if layer == base_model:
                break
            if not isinstance(layer, tf.keras.layers.InputLayer):
                pre_layers.append(layer)

        # Inner model for gradients
        grad_inner_model = tf.keras.models.Model(
            inputs=base_model.input, 
            outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
        )

        # Classifier model (layers after base model)
        classifier_input = tf.keras.Input(shape=base_model.outputs[0].shape[1:])
        x = classifier_input
        start_classifier = False
        for layer in model.layers:
            if start_classifier:
                x = layer(x)
            if layer == base_model:
                start_classifier = True
        classifier_model = tf.keras.models.Model(inputs=classifier_input, outputs=x)

        # Compute gradients
        with tf.GradientTape() as tape:
            x = img_tensor
            for p in pre_layers:
                x = p(x)

            conv_outputs, base_outputs = grad_inner_model(x)
            tape.watch(conv_outputs)

            preds = classifier_model(base_outputs)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap_np = heatmap.numpy()
        
        # Gentle smoothing: upscale then apply Gaussian blur for organic-looking heatmaps
        h, w = heatmap_np.shape
        heatmap_up = cv2.resize(heatmap_np, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        heatmap_up = cv2.GaussianBlur(heatmap_up, (0, 0), sigmaX=3)
        # Resize back to original grid size (overlay_heatmap will upscale to image size later)
        heatmap_np = cv2.resize(heatmap_up, (w, h), interpolation=cv2.INTER_CUBIC)
        # Re-normalize after blur
        heatmap_np = np.clip(heatmap_np, 0, None)
        heatmap_np = heatmap_np / (heatmap_np.max() + 1e-8)
        
        return heatmap_np


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay a GradCAM heatmap on the original image.

    Args:
        img: original image as numpy array (H, W, 3), values in [0, 255] or [0, 1]
        heatmap: GradCAM heatmap (H_map, W_map), values in [0, 1]
        alpha: transparency factor for the heatmap overlay
        colormap: OpenCV colormap for the heatmap

    Returns:
        superimposed_img: numpy array (H, W, 3), uint8
    """
    # Ensure image is uint8 in 0-255 range
    if img.max() <= 1.0:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # Create brain mask from the original image to suppress background
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY) if img_uint8.ndim == 3 else img_uint8
    _, brain_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    brain_mask = cv2.erode(brain_mask, np.ones((11, 11), np.uint8), iterations=1)
    brain_mask_float = (brain_mask / 255.0).astype(np.float32)
    
    # Zero out heatmap outside the brain
    heatmap_resized = heatmap_resized * brain_mask_float

    # Convert heatmap to colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Use per-pixel alpha: blend strongly where heatmap is hot, show original where cold
    # This prevents cold blue JET tint from covering the entire brain
    blend_mask = heatmap_resized[..., np.newaxis]  # (H, W, 1)
    blend_strength = np.clip(blend_mask * (alpha * 2.5), 0, alpha)  # Scale up so hot spots get full alpha
    
    superimposed = (img_uint8.astype(np.float32) * (1 - blend_strength) + 
                    heatmap_colored.astype(np.float32) * blend_strength)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return superimposed


def generate_gradcam_visualization(
    model, img_path, img_size=224, class_names=None,
    last_conv_layer_name=None, save_path=None
):
    """
    Complete GradCAM pipeline: load image → predict → generate heatmap → overlay → display/save.

    Args:
        model: trained Keras model
        img_path: path to input image
        img_size: expected input size
        class_names: list of class names
        last_conv_layer_name: (optional) last conv layer name
        save_path: (optional) path to save the visualization

    Returns:
        dict with keys: prediction, confidence, heatmap, overlay
    """
    from PIL import Image

    # Load and preprocess
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((img_size, img_size))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_batch, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = float(predictions[0][pred_class])

    if class_names is None:
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Generate heatmap
    heatmap = make_gradcam_heatmap(
        img_batch, model, last_conv_layer_name, pred_class
    )

    # Overlay
    overlay = overlay_heatmap(img_array, heatmap, alpha=0.4)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('GradCAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(
        f'Prediction: {class_names[pred_class]} ({confidence:.1%})',
        fontsize=14, fontweight='bold'
    )
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved GradCAM visualization to {save_path}")

    plt.show()

    return {
        'prediction': class_names[pred_class],
        'confidence': confidence,
        'heatmap': heatmap,
        'overlay': overlay,
        'probabilities': predictions[0]
    }


def gradcam_comparison(models_dict, img_path, img_size=224, class_names=None, save_path=None):
    """
    Compare GradCAM heatmaps across multiple models for the same image.

    Args:
        models_dict: dict of {model_name: model}
        img_path: path to image
        img_size: input size
        class_names: list of class names
        save_path: optional save path

    Returns:
        dict of results per model
    """
    from PIL import Image

    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((img_size, img_size))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    if class_names is None:
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    n_models = len(models_dict)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(4 * (n_models + 1), 8))

    # Original image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    results = {}
    for i, (name, model) in enumerate(models_dict.items(), 1):
        # Handle different input sizes
        if 'inception' in name.lower() or 'xception' in name.lower():
            current_size = 299
        else:
            current_size = img_size

        img_r = img.resize((current_size, current_size))
        img_a = np.array(img_r) / 255.0
        img_b = np.expand_dims(img_a, axis=0)

        preds = model.predict(img_b, verbose=0)
        pred_class = np.argmax(preds[0])
        confidence = float(preds[0][pred_class])

        heatmap = make_gradcam_heatmap(img_b, model)
        overlay = overlay_heatmap(img_a, heatmap)

        axes[0, i].imshow(heatmap, cmap='jet')
        axes[0, i].set_title(f'{name}\nHeatmap', fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(
            f'{class_names[pred_class]} ({confidence:.1%})',
            fontsize=10
        )
        axes[1, i].axis('off')

        results[name] = {
            'prediction': class_names[pred_class],
            'confidence': confidence,
            'heatmap': heatmap,
            'overlay': overlay
        }

    plt.suptitle('GradCAM Comparison Across Models', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return results
