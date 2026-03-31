"""
Model loading and building utilities for Brain Tumor Classification.
Provides consistent model architectures for all 10 classification models.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, ResNet101V2,
    InceptionV3, DenseNet121, MobileNetV2,
    EfficientNetB0, Xception
)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
NUM_CLASSES = 4
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Build a custom 4-block CNN from scratch.
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classification Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='custom_cnn')

    return model


def build_transfer_model(
    base_model_class,
    model_name,
    input_shape=(224, 224, 3),
    num_classes=NUM_CLASSES,
    freeze_base=True,
    fine_tune_from=None
):
    """
    Build a transfer learning model with a custom classification head.

    Args:
        base_model_class: Keras application class (e.g., VGG16, ResNet50)
        model_name: name for the model
        input_shape: input image shape
        num_classes: number of output classes
        freeze_base: whether to freeze the base model
        fine_tune_from: layer index from which to unfreeze (for fine-tuning)

    Returns:
        Compiled Keras model
    """
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    if freeze_base:
        base_model.trainable = False

    if fine_tune_from is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False

    # Custom classification head
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name=model_name)

    return model


def build_vgg16(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, **kwargs):
    return build_transfer_model(VGG16, 'vgg16', input_shape, num_classes, **kwargs)


def build_vgg19(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, **kwargs):
    return build_transfer_model(VGG19, 'vgg19', input_shape, num_classes, **kwargs)


def build_resnet50(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, **kwargs):
    return build_transfer_model(ResNet50, 'resnet50', input_shape, num_classes, **kwargs)


def build_resnet101(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, **kwargs):
    return build_transfer_model(ResNet101V2, 'resnet101', input_shape, num_classes, **kwargs)


def build_inceptionv3(input_shape=(299, 299, 3), num_classes=NUM_CLASSES, **kwargs):
    return build_transfer_model(InceptionV3, 'inceptionv3', input_shape, num_classes, **kwargs)


def build_densenet121(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, **kwargs):
    return build_transfer_model(DenseNet121, 'densenet121', input_shape, num_classes, **kwargs)


def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, **kwargs):
    return build_transfer_model(MobileNetV2, 'mobilenetv2', input_shape, num_classes, **kwargs)


def build_efficientnetb0(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, **kwargs):
    return build_transfer_model(EfficientNetB0, 'efficientnetb0', input_shape, num_classes, **kwargs)


def build_xception(input_shape=(299, 299, 3), num_classes=NUM_CLASSES, **kwargs):
    return build_transfer_model(Xception, 'xception', input_shape, num_classes, **kwargs)


# ─────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────
MODEL_REGISTRY = {
    'custom_cnn': {'builder': build_custom_cnn, 'input_size': 224},
    'vgg16': {'builder': build_vgg16, 'input_size': 224},
    'vgg19': {'builder': build_vgg19, 'input_size': 224},
    'resnet50': {'builder': build_resnet50, 'input_size': 224},
    'resnet101': {'builder': build_resnet101, 'input_size': 224},
    'inceptionv3': {'builder': build_inceptionv3, 'input_size': 299},
    'densenet121': {'builder': build_densenet121, 'input_size': 224},
    'mobilenetv2': {'builder': build_mobilenetv2, 'input_size': 224},
    'efficientnetb0': {'builder': build_efficientnetb0, 'input_size': 224},
    'xception': {'builder': build_xception, 'input_size': 299},
}


def build_model(model_name, **kwargs):
    """Build a model by name from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    builder = MODEL_REGISTRY[model_name]['builder']
    return builder(**kwargs)


def compile_model(model, learning_rate=1e-4):
    """Compile a model with standard settings for brain tumor classification."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model


def get_callbacks(model_name, models_dir='../models', patience=5):
    """Get standard training callbacks."""
    os.makedirs(models_dir, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(models_dir, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


# ─────────────────────────────────────────────
# Custom Layers for Segmentation Support
# ─────────────────────────────────────────────

class scSEAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, kernel_size=7, **kwargs):
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        super(scSEAttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        reduced = max(1, channels // self.reduction_ratio)

        self.se_dense_1 = tf.keras.layers.Dense(reduced, activation='relu')
        self.se_dense_2 = tf.keras.layers.Dense(channels, activation='sigmoid')

        # FIX: saved model's spatial conv had 2 input channels
        # meaning it was applied to a 2-channel tensor (avg + max pool concat)
        self.s_se_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(self.kernel_size, self.kernel_size),
            activation='sigmoid',
            padding='same',
            use_bias=False
        )
        super(scSEAttentionBlock, self).build(input_shape)

    def call(self, x):
        # Channel SE path
        avg_pool = tf.reduce_mean(x, axis=[1, 2])
        c = self.se_dense_1(avg_pool)
        c = self.se_dense_2(c)
        c = tf.reshape(c, [-1, 1, 1, tf.shape(c)[-1]])
        x_c = x * c

        # Spatial SE path — FIX: use avg+max concat (2 channels) before conv
        avg_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)  # (B,H,W,1)
        max_spatial = tf.reduce_max(x, axis=-1, keepdims=True)   # (B,H,W,1)
        spatial_concat = tf.concat([avg_spatial, max_spatial], axis=-1)  # (B,H,W,2)
        s = self.s_se_conv(spatial_concat)  # now kernel is (7,7,2,1) ✓
        x_s = x * s

        return x_c + x_s

    def get_config(self):
        config = super(scSEAttentionBlock, self).get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size,
        })
        return config


def load_trained_model(model_path, custom_objects=None):
    """
    Load a trained model from .h5 file with flexible custom object handling.
    """
    if custom_objects is None:
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'iou_metric': iou_metric,
            'iou_score': iou_metric,
            'f-score': dice_coefficient,
            'dice_loss_plus_binary_focal_loss': dice_loss,
            'scSEAttentionBlock': scSEAttentionBlock # Critical for Attention U-Net
        }

    return tf.keras.models.load_model(
        model_path, 
        custom_objects=custom_objects,
        compile=False # Inference mode
    )


# ─────────────────────────────────────────────
# Attention Gate (Oktay et al., 2018)
# ─────────────────────────────────────────────
def attention_gate(x, gating, inter_channels):
    """
    Additive Attention Gate for skip connections.

    Learns to highlight salient features from the encoder (x)
    using the gating signal from the decoder (g). Suppresses
    irrelevant background regions.

    Formula: α = σ(ψ(ReLU(W_x·x + W_g·g + b)))
             output = x * α

    Args:
        x: encoder feature map (skip connection)
        gating: decoder feature map (gating signal)
        inter_channels: number of intermediate channels

    Returns:
        Attention-weighted encoder features
    """
    # Transform encoder features
    theta_x = layers.Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(x)
    theta_x = layers.BatchNormalization()(theta_x)

    # Transform gating signal
    phi_g = layers.Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(gating)
    phi_g = layers.BatchNormalization()(phi_g)

    # Additive attention
    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add)

    # Attention coefficients (sigmoid → 0 to 1)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act)
    psi = layers.BatchNormalization()(psi)
    psi = layers.Activation('sigmoid')(psi)

    # Multiply encoder features by attention map
    return layers.Multiply()([x, psi])


# ─────────────────────────────────────────────
# Attention U-Net for Segmentation
# ─────────────────────────────────────────────
def build_attention_unet(input_shape=(256, 256, 3), num_classes=1):
    """
    Build an Attention U-Net model for binary segmentation.

    Adds attention gates at each skip connection in the decoder path.
    The attention gates learn to suppress irrelevant encoder features
    and amplify features relevant to the tumor region.

    Reference: Oktay et al., "Attention U-Net: Learning Where to Look
    for the Pancreas", 2018 (arXiv:1804.03999)

    Args:
        input_shape: input image shape
        num_classes: 1 for binary segmentation

    Returns:
        Attention U-Net Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # ── Encoder (Contracting Path) ──
    # Block 1
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.1)(p1)

    # Block 2
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.1)(p2)

    # Block 3
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.2)(p3)

    # Block 4
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.2)(p4)

    # ── Bottleneck ──
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.3)(c5)

    # ── Decoder (Expanding Path) with Attention Gates ──
    # Block 6: Attention Gate on c4, gated by upsampled c5
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    a4 = attention_gate(c4, u6, inter_channels=256)
    u6 = layers.concatenate([u6, a4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Dropout(0.2)(c6)

    # Block 7: Attention Gate on c3, gated by upsampled c6
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    a3 = attention_gate(c3, u7, inter_channels=128)
    u7 = layers.concatenate([u7, a3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Dropout(0.2)(c7)

    # Block 8: Attention Gate on c2, gated by upsampled c7
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    a2 = attention_gate(c2, u8, inter_channels=64)
    u8 = layers.concatenate([u8, a2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Dropout(0.1)(c8)

    # Block 9: Attention Gate on c1, gated by upsampled c8
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    a1 = attention_gate(c1, u9, inter_channels=32)
    u9 = layers.concatenate([u9, a1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)

    # ── Output ──
    if num_classes == 1:
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    else:
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = models.Model(inputs=inputs, outputs=outputs, name='attention_unet')

    return model


# Backward-compatible alias
def build_unet(input_shape=(256, 256, 3), num_classes=1):
    """Legacy alias — now builds Attention U-Net."""
    return build_attention_unet(input_shape, num_classes)


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Dice coefficient metric for segmentation."""
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) + smooth
    )


def dice_loss(y_true, y_pred):
    """Dice loss for segmentation training."""
    return 1 - dice_coefficient(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1.0):
    """Intersection over Union metric."""
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    union = tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) - intersection
    return (intersection + smooth) / (union + smooth)
