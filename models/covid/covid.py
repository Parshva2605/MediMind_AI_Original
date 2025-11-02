"""
covid.py - Simple COVID-19 Detection from Chest X-ray Images
=============================================================

MODEL INFORMATION:
------------------
Model Architecture:  EfficientNetB3 (Transfer Learning)
Model Weight File:   checkpoints/model_epoch_28_acc_0.8987.h5
Model Size:          ~45 MB (47,401,336 bytes)
Total Parameters:    11,738,803 parameters
Input Shape:         (300, 300, 3) - RGB images

PERFORMANCE METRICS:
--------------------
Overall Accuracy:    91.45% (on 4,235 test images)
Top-2 Accuracy:      99.06%
COVID Detection:     92.70% (on 3,616 COVID images)
                     - Correctly detected: 3,352 cases
                     - Missed: 264 cases

Per-Class Accuracy:
  • COVID:           91.57% (663/724 test images)
  • Normal:          94.61% (1,929/2,039 test images)
  • Lung_Opacity:    87.28% (1,050/1,203 test images)
  • Viral Pneumonia: 85.87% (231/269 test images)

MODEL ARCHITECTURE DETAILS:
---------------------------
Base Model:
  - EfficientNetB3 pre-trained on ImageNet
  - Partially fine-tuned (last 100 layers trainable)
  - Input: 300×300×3 RGB images
  
Custom Classifier Head:
  1. GlobalAveragePooling2D
  2. Dense(512, relu) + L2 regularization (0.001)
  3. BatchNormalization + Dropout(0.5)
  4. Dense(256, relu) + L2 regularization (0.001)
  5. BatchNormalization + Dropout(0.4)
  6. Dense(128, relu) + L2 regularization (0.001)
  7. BatchNormalization + Dropout(0.3)
  8. Dense(4, softmax) - Output layer

TRAINING DETAILS:
-----------------
Training Data:       ~18,000 chest X-ray images
Validation Split:    15% from training set
Preprocessing:       EfficientNet preprocess_input
Augmentation:        Rotation, shift, zoom, brightness
Loss Function:       Categorical Crossentropy (label smoothing 0.05)
Optimizer:           Adam
Learning Rates:
  - Phase 1: 3e-4 (frozen base)
  - Phase 2: 1e-5 (partial fine-tune)
Class Weighting:     Balanced using sklearn
Training Epochs:     28 epochs (best checkpoint)

DEPENDENCIES:
-------------
- TensorFlow >= 2.10
- NumPy >= 1.21
- Python >= 3.8

USAGE:
------
1. Set IMAGE_PATH in Configuration section (line ~78)
2. Run: python covid.py
3. Or: python covid.py "path/to/xray.jpg"

OUTPUT CLASSES:
---------------
1. COVID           - COVID-19 infection detected
2. Lung_Opacity    - Cloudy/hazy lung areas (pneumonia, fluid, inflammation)
3. Normal          - Healthy, clear lungs
4. Viral Pneumonia - Viral lung infection (non-COVID)

CITATION/CREDITS:
-----------------
Model trained on publicly available COVID-19 chest X-ray dataset
Training Date: October 2025
Framework: TensorFlow/Keras with EfficientNetB3

DISCLAIMER:
-----------
This is an AI-based screening tool for educational/research purposes.
NOT a replacement for professional medical diagnosis.
Always consult qualified healthcare professionals for medical decisions.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ======================
# CONFIGURATION
# ======================
MODEL_PATH = r"checkpoints\model_epoch_28_acc_0.8987.h5"

# ⚙️ PUT YOUR IMAGE PATH HERE
IMAGE_PATH = r"C:\Users\91851\Desktop\datasets\covid -19\covid\COVID\images\COVID-1.png"

# Examples:
# IMAGE_PATH = r"C:\Users\YourName\Desktop\my_xray.jpg"
# IMAGE_PATH = r"covid_split_dataset\test\Normal\Normal-1.png"

IMG_SIZE = (300, 300)
CLASS_LABELS = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# ======================
# BUILD MODEL
# ======================
def build_model():
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(300, 300, 3)
    )
    
    base_model.trainable = True
    for layer in base_model.layers[:-100]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])
    
    return model

# ======================
# LOAD MODEL
# ======================
def load_model_weights():
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(MODEL_PATH)
    return model

# ======================
# PREDICT
# ======================
def predict(img_path):
    # Load image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    probs = predictions[0]
    
    # Get result
    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_LABELS[predicted_idx]
    confidence = float(probs[predicted_idx]) * 100
    
    return predicted_class, confidence

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    # Enable GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    
    # Load model
    print("Loading model...")
    model = load_model_weights()
    
    # Get image path
    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip().strip('"').strip("'")
    else:
        img_path = IMAGE_PATH
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"ERROR: Image not found - {img_path}")
        sys.exit(1)
    
    # Predict
    print(f"Analyzing: {img_path}")
    result, conf = predict(img_path)
    
    # Print result
    print(f"\nResult: {result} ({conf:.1f}%)")
    
    # Explain what it means
    print("\nWhat this means:")
    if result == "COVID":
        print("  → COVID-19 infection detected")
        print("  → Virus that causes respiratory illness")
        print("  → Requires immediate medical attention and isolation")
    elif result == "Lung_Opacity":
        print("  → Cloudy/hazy areas in lungs detected")
        print("  → Could be pneumonia, fluid, or inflammation")
        print("  → Medical evaluation recommended")
    elif result == "Normal":
        print("  → No abnormalities detected")
        print("  → Lungs appear healthy and clear")
        print("  → Continue regular health monitoring")
    elif result == "Viral Pneumonia":
        print("  → Viral lung infection detected (non-COVID)")
        print("  → Caused by viruses other than COVID-19")
        print("  → Medical treatment may be needed")
