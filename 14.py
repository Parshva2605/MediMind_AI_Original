import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ✅ Disease labels (must match training order)
disease_labels = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia'
]

# ✅ Load trained model
model = load_model('final_chest_disease_model.h5')
print("✅ Model loaded successfully.")

# ✅ Function to load and preprocess image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize like training
    return np.expand_dims(img_array, axis=0), img  # shape: (1, 224, 224, 3)

# ✅ Predict diseases
def predict_diseases(img_path, threshold=0.5):
    img_batch, original_img = preprocess_image(img_path)
    preds = model.predict(img_batch)[0]  # shape: (14,)
    
    results = {}
    for i, prob in enumerate(preds):
        label = disease_labels[i]
        results[label] = float(prob)

    # Show predictions above threshold
    print("\n🔎 Predictions above threshold:")
    for disease, prob in results.items():
        if prob >= threshold:
            print(f"✅ {disease}: {prob:.2f}")
    print("\n📋 All Predictions:")
    for disease, prob in results.items():
        print(f"{disease}: {prob:.2f}")
    
    # Optional: show image
    plt.imshow(original_img)
    plt.title("X-ray Image")
    plt.axis('off')
    plt.show()

# ✅ Run prediction
image_path = "check.png"  # replace with actual image path
predict_diseases(image_path)
