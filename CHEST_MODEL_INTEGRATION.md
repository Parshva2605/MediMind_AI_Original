# Chest X-Ray Model Integration - Complete Guide

## Overview
This document explains the integration of the `best_chest_model.h5` - an optimized chest X-ray disease detection model with **94.90% accuracy**.

## Model Details

### Location
```
models/chest/best_chest_model.h5
```

### Specifications
- **Input**: 224x224 RGB images
- **Output**: 14 disease probabilities (sigmoid activation)
- **Threshold**: 0.35 (optimized for this model)
- **Accuracy**: 94.90%
- **Architecture**: CNN-based with transfer learning

### 14 Disease Labels (in exact order)
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

## Image Preprocessing Pipeline

The model requires specific preprocessing steps (CRITICAL for accuracy):

```python
import cv2
import numpy as np

# Step 1: Load image
img = cv2.imread(image_path)

# Step 2: Convert BGR to RGB (OpenCV loads as BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 3: Resize to 224x224
img = cv2.resize(img, (224, 224))

# Step 4: Normalize to [0, 1]
img = img / 255.0

# Step 5: Add batch dimension
img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)
```

## Model Loading

```python
import tensorflow as tf

# Simple loading - no custom objects needed
model = tf.keras.models.load_model('models/chest/best_chest_model.h5', compile=False)
```

## Prediction Process

```python
# Make prediction
predictions = model.predict(img, verbose=0)[0]  # Shape: (14,)

# Apply threshold
THRESHOLD = 0.35
detected_diseases = []

DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

for i, disease in enumerate(DISEASES):
    probability = float(predictions[i])
    if probability >= THRESHOLD:
        detected_diseases.append((disease, probability * 100))
```

## Integration Points in Application

### 1. Model Loading (`app.py` line ~164)
```python
def load_chest_disease_model():
    model_path = os.path.join('models', 'chest', 'best_chest_model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    return model
```

### 2. Image Processing (`app.py` line ~900+)
```python
def process_chest_xray(image_path):
    # Load model
    if chest_disease_model is None:
        return {'error': 'Model not loaded'}
    
    # Preprocess
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = chest_disease_model.predict(img, verbose=0)[0]
    
    # Process results with THRESHOLD = 0.35
    ...
```

### 3. Result Display (`templates/test_result.html`)
- Shows top 3 conditions
- Displays all 14 diseases in table
- Highlights detected conditions (above threshold)
- Shows detection summary

### 4. PDF Report Generation (`app.py` - `generate_report()`)
- Includes model info and threshold
- Lists all detected conditions
- Shows confidence percentages
- Adds medical recommendations

### 5. AI Summary (`ai_helper.py`)
- Generates detailed medical summary
- Lists detected conditions
- Provides recommendations based on findings
- Includes disclaimers

## Testing

Run the test script to verify integration:

```bash
python test_chest_model.py
```

Expected output:
```
✅ Model file found
✅ TensorFlow loaded
✅ Model loaded successfully
✅ 14 disease labels configured
✅ Preprocessing successful
✅ Prediction successful
✅ Threshold detection working
🎉 ALL TESTS PASSED!
```

## Threshold Explanation

**Why 0.35 instead of 0.50?**

The 0.35 threshold was optimized for this specific model based on:
- **Sensitivity**: Better detection of early-stage conditions
- **Specificity**: Balanced to reduce false negatives
- **Clinical relevance**: Early detection is prioritized for medical screening

This is different from the default 0.5 threshold commonly used in binary classification.

## Performance Metrics

- **Accuracy**: 94.90%
- **Threshold**: 0.35 (optimized)
- **Input Size**: 224x224 RGB
- **Processing Time**: ~100-300ms per image (GPU)
- **Model Size**: ~90-150 MB

## Common Issues & Solutions

### Issue 1: Model not loading
**Solution**: Ensure file exists at `models/chest/best_chest_model.h5`

### Issue 2: Wrong predictions
**Solution**: Verify preprocessing steps (BGR→RGB conversion is critical)

### Issue 3: All predictions are low
**Solution**: Check image normalization (should be 0-1 range)

### Issue 4: Import errors
**Solution**: Install dependencies:
```bash
pip install tensorflow opencv-python numpy
```

## Files Modified

1. **app.py**
   - `load_chest_disease_model()` - Updated model path and loading
   - `process_chest_xray()` - Complete rewrite with proper preprocessing
   - `generate_report()` - Enhanced for new result format

2. **ai_helper.py**
   - `generate_fallback_summary()` - Improved chest X-ray summaries

3. **templates/test_result.html**
   - Enhanced disease table with visual indicators
   - Added threshold information display
   - Better status badges

4. **test_chest_model.py** (NEW)
   - Comprehensive testing script

5. **CHEST_MODEL_INTEGRATION.md** (THIS FILE)
   - Complete documentation

## Reference Implementation

Original working code: `models/chest/1.py`

This implementation follows the exact preprocessing and prediction pipeline proven to work with this model.

## Support

For issues or questions:
1. Check this documentation
2. Run `test_chest_model.py` to diagnose
3. Verify preprocessing pipeline matches reference
4. Check TensorFlow version compatibility

---

**Last Updated**: October 31, 2025
**Model Version**: best_chest_model.h5 (94.90% accuracy)
**Integration Status**: ✅ Complete and Tested
