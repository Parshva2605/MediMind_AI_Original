"""
Simple X-Ray Disease Prediction Script
Loads best_chest_model.h5 and predicts diseases with threshold 0.35
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
from tensorflow import keras

# Configuration
MODEL_PATH = r'C:\Users\91851\Desktop\charuset\Aio\best_chest_model.h5'
THRESHOLD = 0.35  # Optimized threshold

# 14 Disease labels
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

print("=" * 70)
print("🏥 CHEST X-RAY DISEASE PREDICTION")
print("=" * 70)

# Load model
print(f"\n📂 Loading model...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print(f"✅ Model loaded successfully")
print(f"   Threshold: {THRESHOLD}")

def predict_xray(image_path):
    """
    Predict diseases from X-ray image
    
    Args:
        image_path: Path to X-ray image
    
    Returns:
        Dictionary with detected diseases
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = model.predict(img, verbose=0)[0]
    
    # Process results
    results = {}
    detected = []
    
    for i, disease in enumerate(DISEASES):
        probability = float(predictions[i])
        confidence_percent = probability * 100
        is_detected = probability >= THRESHOLD
        
        results[disease] = {
            'probability': probability,
            'confidence_percent': confidence_percent,
            'detected': is_detected
        }
        
        if is_detected:
            detected.append((disease, confidence_percent))
    
    return results, detected

# Example usage
if __name__ == "__main__":
    # Test image path (change this to your image)
    test_image = r'C:\Users\91851\Desktop\charuset\Aio\aio-dataset\images_001\images\00000001_000.png'
    
    # Check if file exists
    if not os.path.exists(test_image):
        print(f"\n⚠️  Test image not found: {test_image}")
        print("Please update 'test_image' path to your X-ray image")
    else:
        print(f"\n📸 Analyzing: {os.path.basename(test_image)}")
        print("-" * 70)
        
        # Predict
        results, detected = predict_xray(test_image)
        
        # Display results
        if len(detected) == 0:
            print("\n✅ NO DISEASES DETECTED")
            print("   X-ray appears normal")
        else:
            print(f"\n⚠️  {len(detected)} DISEASE(S) DETECTED:")
            print()
            for disease, confidence in detected:
                # Severity based on confidence
                if confidence >= 80:
                    severity = "🔴 CRITICAL"
                elif confidence >= 60:
                    severity = "🟡 HIGH"
                elif confidence >= 40:
                    severity = "🟢 MODERATE"
                else:
                    severity = "⚪ LOW"
                
                print(f"  {severity}")
                print(f"  └─ {disease}: {confidence:.1f}%")
                print()
        
        # Show all probabilities
        print("\n📊 All Disease Probabilities:")
        print("-" * 70)
        for disease, data in results.items():
            status = "✓" if data['detected'] else "✗"
            print(f"  {status} {disease:<20} {data['confidence_percent']:>6.2f}%")
    
    print("\n" + "=" * 70)
    print("✅ Prediction complete!")
    print("=" * 70)
    
    print("\n💡 How to use with your own image:")
    print("   1. Change 'test_image' path to your X-ray image")
    print("   2. Run: python 1.py")
    print("   3. Or use predict_xray() function in your code")
    print()
    print("   Example:")
    print("   >>> results, detected = predict_xray('my_xray.png')")
    print("   >>> print(detected)")
