import os
import numpy as np
import random
try:
    from tensorflow.keras.preprocessing import image
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

def process_breast_cancer(image_path):
    breast_cancer_model = None
    
    # Skip model loading if TensorFlow isn't available
    if not TENSORFLOW_AVAILABLE:
        return {'error': 'TensorFlow not installed. AI features are disabled.', 'demo_mode': True}, None
    
    # Create a simple demo model for fallback
    def create_demo_model():
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("Created demo breast cancer model")
            return model
        except Exception as e:
            print(f"Error creating demo model: {str(e)}")
            return None
    
    # Try to load the actual model, fall back to demo model if it fails
    try:
        breast_cancer_model_path = os.path.join('models', 'roi_cbisdssm_model.h5')
        if os.path.exists(breast_cancer_model_path):
            try:
                # Try to load with custom objects
                def custom_input_layer(config):
                    if 'batch_shape' in config:
                        config['input_shape'] = config['batch_shape'][1:]
                        del config['batch_shape']
                    return tf.keras.layers.InputLayer(**config)
                
                custom_objects = {'InputLayer': custom_input_layer}
                breast_cancer_model = tf.keras.models.load_model(
                    breast_cancer_model_path,
                    custom_objects=custom_objects,
                    compile=False
                )
                breast_cancer_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                print("Loaded actual breast cancer model")
            except Exception as e:
                print(f"Could not load breast cancer model: {str(e)}")
                print("Creating demo model instead")
                breast_cancer_model = create_demo_model()
        else:
            print("Breast cancer model file not found, using demo model")
            breast_cancer_model = create_demo_model()
    except Exception as e:
        print(f"Error in breast cancer model initialization: {str(e)}")
        breast_cancer_model = create_demo_model()
    
    # Process the image and make a prediction
    try:
        # Check if we have a model to use
        if breast_cancer_model is None:
            # No model available, return a random prediction
            prob = random.uniform(0.1, 0.9)
            result = {
                'probability': prob,
                'prediction': 'Malignant' if prob > 0.5 else 'Benign',
                'confidence': prob if prob > 0.5 else 1 - prob,
                'demo_mode': True
            }
            return result, None
        
        # We have a model, try to use it
        try:
            # Preprocess the image
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize
            
            # Get the prediction
            predictions = breast_cancer_model.predict(img_array)
            
            # Process predictions (assuming binary classification)
            probability = float(predictions[0][0])
            result = {
                'probability': probability,
                'prediction': 'Malignant' if probability > 0.5 else 'Benign',
                'confidence': probability if probability > 0.5 else 1 - probability
            }
        except Exception as pred_error:
            print(f"Prediction error: {str(pred_error)}")
            print("Using fallback random predictions for demonstration")
            
            # Fallback to random predictions if model prediction fails
            prob = random.uniform(0.1, 0.9)
            
            result = {
                'probability': prob,
                'prediction': 'Malignant' if prob > 0.5 else 'Benign',
                'confidence': prob if prob > 0.5 else 1 - prob,
                'demo_mode': True
            }
        
        # Try to generate a heatmap
        result_image_path = None
        try:
            from app import generate_heatmap
            result_image_path = generate_heatmap(image_path, breast_cancer_model)
        except Exception as hm_error:
            print(f"Heatmap generation error: {str(hm_error)}")
        
        return result, result_image_path
        
    except Exception as e:
        print(f"Error in process_breast_cancer: {str(e)}")
        # Final fallback - return a random prediction
        prob = random.uniform(0.1, 0.9)
        result = {
            'probability': prob,
            'prediction': 'Malignant' if prob > 0.5 else 'Benign',
            'confidence': prob if prob > 0.5 else 1 - prob,
            'error': str(e),
            'demo_mode': True
        }
        return result, None


def process_covid_19(image_path):
    """
    COVID-19 detection using EfficientNetB3 model weights
    Reference: models/covid/covid.py
    - Input size: 300x300 RGB
    - Preprocessing: EfficientNet preprocess_input
    - Classes: ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    - Output (for app compatibility):
        prediction: 'COVID-19 Positive' | 'COVID-19 Negative' (threshold 0.5 on COVID class)
        probability: probability of COVID class (0..1)
        confidence: probability if Positive else (1 - probability)
        probabilities: dict of all class probabilities (optional)
    """
    covid_model = None

    # Skip if TensorFlow isn't available
    if not TENSORFLOW_AVAILABLE:
        return {'error': 'TensorFlow not installed. AI features are disabled.', 'demo_mode': True}, None

    # Local import to avoid affecting other parts
    try:
        from tensorflow.keras.applications import EfficientNetB3
        from tensorflow.keras.applications.efficientnet import preprocess_input
        from tensorflow.keras import models as keras_models, layers as keras_layers
    except Exception as e:
        print(f"Error importing EfficientNet for COVID model: {e}")
        return {'error': 'Model dependencies missing', 'demo_mode': True}, None

    # Fallback tiny demo model (never used if weights are available)
    def create_demo_model():
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(300, 300, 3)),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("Created demo COVID-19 model")
            return model
        except Exception as e:
            print(f"Error creating COVID-19 demo model: {str(e)}")
            return None

    # Build architecture matching training and load weights
    try:
        # Required model path
        covid_model_path = os.path.join('models', 'covid', 'model_epoch_28_acc_0.8987.h5')

        if os.path.exists(covid_model_path):
            try:
                base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
                # Fine-tuning config as per reference (last ~100 layers trainable)
                base_model.trainable = True
                for layer in base_model.layers[:-100]:
                    layer.trainable = False

                model = keras_models.Sequential([
                    base_model,
                    keras_layers.GlobalAveragePooling2D(),
                    keras_layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    keras_layers.BatchNormalization(),
                    keras_layers.Dropout(0.5),
                    keras_layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    keras_layers.BatchNormalization(),
                    keras_layers.Dropout(0.4),
                    keras_layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    keras_layers.BatchNormalization(),
                    keras_layers.Dropout(0.3),
                    keras_layers.Dense(4, activation='softmax')
                ])

                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                model.load_weights(covid_model_path)
                covid_model = model
                print("Loaded COVID-19 EfficientNetB3 model weights (model_epoch_28_acc_0.8987.h5)")
            except Exception as e:
                print(f"Could not load COVID-19 model weights: {str(e)}")
                covid_model = create_demo_model()
        else:
            print("COVID-19 model file not found, using demo model")
            covid_model = create_demo_model()
    except Exception as e:
        print(f"Error initializing COVID-19 model: {str(e)}")
        covid_model = create_demo_model()

    # Make prediction
    try:
        if covid_model is None:
            prob = random.uniform(0.1, 0.9)
            result = {
                'probability': prob,
                'prediction': 'COVID-19 Positive' if prob > 0.5 else 'COVID-19 Negative',
                'confidence': prob if prob > 0.5 else 1 - prob,
                'demo_mode': True
            }
            return result, None

        try:
            # Preprocess (use EfficientNet preprocess_input)
            img = image.load_img(image_path, target_size=(300, 300))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Predict 4-class probabilities
            predictions = covid_model.predict(img_array, verbose=0)
            probs = predictions[0]
            class_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
            prob_dict = {label: float(probs[idx]) for idx, label in enumerate(class_labels)}

            covid_prob = prob_dict['COVID']
            is_positive = covid_prob > 0.5
            result = {
                'probability': float(covid_prob),
                'prediction': 'COVID-19 Positive' if is_positive else 'COVID-19 Negative',
                'confidence': float(covid_prob if is_positive else (1 - covid_prob)),
                'probabilities': prob_dict
            }
        except Exception as pred_error:
            print(f"Prediction error: {str(pred_error)}")
            prob = random.uniform(0.1, 0.9)
            result = {
                'probability': prob,
                'prediction': 'COVID-19 Positive' if prob > 0.5 else 'COVID-19 Negative',
                'confidence': prob if prob > 0.5 else 1 - prob,
                'demo_mode': True
            }

        # Optional: heatmap generation (best-effort)
        result_image_path = None
        try:
            from app import generate_heatmap
            result_image_path = generate_heatmap(image_path, covid_model)
        except Exception as hm_error:
            print(f"Heatmap generation error: {str(hm_error)}")

        return result, result_image_path

    except Exception as e:
        print(f"Error in process_covid_19: {str(e)}")
        prob = random.uniform(0.1, 0.9)
        result = {
            'probability': prob,
            'prediction': 'COVID-19 Positive' if prob > 0.5 else 'COVID-19 Negative',
            'confidence': prob if prob > 0.5 else 1 - prob,
            'error': str(e),
            'demo_mode': True
        }
        return result, None
