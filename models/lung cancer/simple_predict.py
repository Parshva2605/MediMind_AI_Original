import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Simple Lung Cancer Predictor - Malignant or Non-malignant
----------------------------------------------------------
Load the best model and predict: Malignant or Non-malignant

Usage:
    python simple_predict.py path/to/image.jpg
    
Or edit IMAGE_PATH below and run without arguments.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB3

# --------------------------------------------------
# EDIT THESE PATHS
# --------------------------------------------------
MODEL_PATH = r"D:\Lungs_Cancer\checkpoints\stage2_best.h5"
IMAGE_PATH = r"D:\Lungs_Cancer\Datatset\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset\Malignant cases\Malignant case (2).jpg"


def build_model():
    """Rebuild model architecture."""
    try:
        base = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
    except Exception:
        base = EfficientNetB3(include_top=False, weights=None, input_shape=(512, 512, 3))
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= 150)
    inputs = keras.Input(shape=(512, 512, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def preprocess_image(img_path):
    """Apply training-like preprocessing: RGB->Gray, resize, CLAHE, normalize, stack to 3 channels."""
    img = keras.utils.load_img(str(img_path))
    arr = keras.utils.img_to_array(img)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    # RGB -> Grayscale
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Resize to 512x512
    resized = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    # Normalize
    normalized = enhanced.astype(np.float32) / 255.0
    # Stack to 3 channels
    rgb3 = np.stack([normalized, normalized, normalized], axis=-1)
    return rgb3


def predict(model, img_path):
    """Predict and return binary label: Malignant or Non-malignant."""
    arr = preprocess_image(img_path)
    x = np.expand_dims(arr, axis=0)
    probs = model.predict(x, verbose=0)[0]
    
    # probs = [p_benign, p_malignant, p_normal]
    p_malignant = float(probs[1])
    p_non_malignant = float(probs[0] + probs[2])
    
    if p_malignant > 0.5:
        return "Malignant"
    else:
        return "Non-malignant"


def main():
    # Determine image path
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        img_path = Path(IMAGE_PATH)
    
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        print("Usage: python simple_predict.py path/to/image.jpg")
        print("Or edit IMAGE_PATH in this script.")
        return
    
    print(f"Loading model: {MODEL_PATH}")
    model = build_model()
    model.load_weights(MODEL_PATH)
    
    print(f"Predicting: {img_path}")
    result = predict(model, img_path)
    
    print(f"\nResult: {result}")


if __name__ == '__main__':
    # Optional GPU memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
    main()
