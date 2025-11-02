import os
import json
import uuid
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from supabase import create_client
from dotenv import load_dotenv

# Import our AI helper functions
from ai_helper import generate_ai_summary
from new_functions import process_covid_19

# Try to import TensorFlow, but provide graceful fallback if it's not installed
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not found. AI features will be disabled.")
    print("To enable AI features, install TensorFlow: pip install tensorflow")
from functools import wraps
import io
import base64

# Try importing optional dependencies with graceful fallback
try:
    import numpy as np
except ImportError:
    print("NumPy not found. Install with: pip install numpy")
    np = None

try:
    import cv2
except ImportError:
    print("OpenCV not found. Install with: pip install opencv-python")
    cv2 = None
    
# Import for Ollama LLM integration
try:
    import requests
    import json
    OLLAMA_AVAILABLE = True
except ImportError:
    print("Requests library not found. Install with: pip install requests")
    OLLAMA_AVAILABLE = False

try:
    from fpdf import FPDF
except ImportError:
    print("FPDF not found. Report generation will be disabled. Install with: pip install fpdf")
    FPDF = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
except ImportError:
    print("Matplotlib not found. Some visualization features will be disabled. Install with: pip install matplotlib")
    plt = None
    FigureCanvas = None

try:
    from PIL import Image
except ImportError:
    print("Pillow not found. Install with: pip install Pillow")
    Image = None

# Import password hashing library
import hashlib
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORTS_FOLDER'] = 'reports'
app.config['PROFILE_IMAGES'] = 'static/profile_images'

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase = create_client(supabase_url, supabase_key)

# Ollama LLM API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:7b"

# Ensure upload and reports directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

# Supabase table names
DOCTORS_TABLE = 'doctors'
PATIENTS_TABLE = 'patients'
TESTS_TABLE = 'tests'

# Helper functions for database operations
def get_doctor_by_id(doctor_id):
    response = supabase.table(DOCTORS_TABLE).select('*').eq('id', doctor_id).execute()
    if response.data:
        return response.data[0]
    return None

def get_doctor_by_email(email):
    response = supabase.table(DOCTORS_TABLE).select('*').eq('email', email).execute()
    if response.data:
        return response.data[0]
    return None

def get_patients_by_doctor(doctor_id):
    response = supabase.table(PATIENTS_TABLE).select('*').eq('doctor_id', doctor_id).execute()
    return response.data

def get_patient_by_id(patient_id):
    response = supabase.table(PATIENTS_TABLE).select('*').eq('id', patient_id).execute()
    if response.data:
        return response.data[0]
    return None

def get_test_by_id(test_id):
    response = supabase.table(TESTS_TABLE).select('*').eq('id', test_id).execute()
    if response.data:
        return response.data[0]
    return None

def get_tests_by_patient(patient_id):
    try:
        response = supabase.table(TESTS_TABLE).select('*').eq('patient_id', patient_id).execute()
        tests = response.data if response.data else []
        
        # Debug information about tests
        print(f"Retrieved {len(tests)} tests for patient {patient_id}")
        for test in tests:
            if 'report_path' in test and test['report_path']:
                print(f"Test {test['id']} has report: {test['report_path']}")
                # Check if file exists
                if os.path.exists(test['report_path']):
                    print(f"Report file exists at {test['report_path']}")
                else:
                    print(f"WARNING: Report file does not exist at {test['report_path']}")
        
        return tests
    except Exception as e:
        print(f"Error getting tests for patient {patient_id}: {str(e)}")
        return []

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'doctor_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Load the chest disease model
def load_chest_disease_model():
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, skipping model loading")
        return None
        
    try:
        # Use the best_chest_model.h5 - optimized model with 94.90% accuracy
        model_path = os.path.join('models', 'chest', 'best_chest_model.h5')
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return None
            
        try:
            import tensorflow as tf
            
            # Simple loading approach as per reference code (1.py)
            # This model was trained properly and doesn't need custom objects
            print(f"📂 Loading chest disease model from {model_path}...")
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✅ Chest disease model loaded successfully")
            print(f"   Model expects 224x224 RGB images")
            print(f"   Outputs: 14 disease probabilities")
            print(f"   Using optimized threshold: 0.35")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
            
    except Exception as outer_e:
        print(f"Outer error loading model: {str(outer_e)}")
        return None

# Load the lung cancer model
def load_lung_cancer_model():
    """
    Load the lung cancer CT scan model (stage2_best.h5)
    Binary classification: Malignant vs Non-malignant
    Accuracy: 96.8%
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, skipping model loading")
        return None
        
    try:
        model_path = os.path.join('models', 'lung cancer', 'stage2_best.h5')
        if not os.path.exists(model_path):
            print(f"Lung cancer model file not found at {model_path}")
            return None
            
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, regularizers
            from tensorflow.keras.applications import EfficientNetB3
            
            # Build model architecture (must match training)
            print(f"📂 Building lung cancer model architecture...")
            
            try:
                base = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
            except Exception:
                print("   Using EfficientNetB3 without ImageNet weights")
                base = EfficientNetB3(include_top=False, weights=None, input_shape=(512, 512, 3))
            
            # Set trainability (as per training)
            for i, layer in enumerate(base.layers):
                layer.trainable = (i >= 150)
            
            # Build full model
            inputs = tf.keras.Input(shape=(512, 512, 3))
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
            outputs = layers.Dense(3, activation='softmax')(x)  # 3 classes: Benign, Malignant, Normal
            
            model = tf.keras.Model(inputs, outputs)
            
            # Load weights
            print(f"📂 Loading weights from {model_path}...")
            model.load_weights(model_path)
            
            print("✅ Lung cancer model loaded successfully")
            print("   Input: 512x512 CT scan images")
            print("   Output: Malignant vs Non-malignant (96.8% accuracy)")
            
            return model
            
        except Exception as e:
            print(f"Error loading lung cancer model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as outer_e:
        print(f"Outer error loading lung cancer model: {str(outer_e)}")
        return None

# Load the COVID-19 model
def load_covid_model():
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, skipping model loading")
        return None
        
    try:
        # Use the EfficientNetB3 architecture and load weights, matching models/covid/covid.py
        model_path = os.path.join('models', 'covid', 'model_epoch_28_acc_0.8987.h5')
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return None

        try:
            import tensorflow as tf
            from tensorflow.keras.applications import EfficientNetB3
            from tensorflow.keras import models as keras_models, layers as keras_layers

            # Build architecture
            base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
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

            # Compile then load weights
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print(f"📂 Loading COVID-19 weights from {model_path}...")
            model.load_weights(model_path)
            print("✅ COVID-19 model loaded successfully (EfficientNetB3, 4-class)")

        except Exception as e:
            print(f"Error loading COVID-19 model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

        return model
        
    except Exception as outer_e:
        print(f"Outer error loading model: {str(outer_e)}")
        return None
            
        print("Pneumonia model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading pneumonia model: {str(e)}")
        return None

# Load models globally
chest_disease_model = None
lung_cancer_model = None
try:
    if TENSORFLOW_AVAILABLE:
        chest_disease_model = load_chest_disease_model()
        lung_cancer_model = load_lung_cancer_model()
except Exception as e:
    print(f"Error loading models: {str(e)}")

# Password hashing function
def hash_password(password, salt=None):
    if not salt:
        salt = os.urandom(32)
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    # Convert to base64 for storage in Supabase
    return base64.b64encode(salt + hash_obj).decode('utf-8')

# Password verification function
def verify_password(stored_password, provided_password):
    # Decode from base64
    binary_data = base64.b64decode(stored_password.encode('utf-8'))
    salt = binary_data[:32]  # The salt is the first 32 bytes
    stored_hash = binary_data[32:]
    hash_obj = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    return hash_obj == stored_hash

# Create profile image directory if it doesn't exist
os.makedirs(app.config['PROFILE_IMAGES'], exist_ok=True)

# Routes
@app.route('/')
def index():
    if 'doctor_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        specialty = request.form.get('specialty', '')
        phone = request.form.get('phone', '')
        
        # Check if email already exists
        existing_doctor = get_doctor_by_email(email)
        if existing_doctor:
            flash('Email already registered. Please login instead.', 'danger')
            return redirect(url_for('login'))
        
        # Hash the password
        hashed_password = hash_password(password)
        
        # Handle profile image upload if provided
        profile_image_path = None
        if 'profile_image' in request.files:
            profile_image = request.files['profile_image']
            if profile_image.filename != '':
                filename = secure_filename(f"{uuid.uuid4()}_{profile_image.filename}")
                image_path = os.path.join(app.config['PROFILE_IMAGES'], filename)
                profile_image.save(image_path)
                profile_image_path = f"profile_images/{filename}"
        
        # Create new doctor
        new_doctor = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'specialty': specialty,
            'phone': phone,
            'profile_image': profile_image_path,
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Insert into Supabase
        response = supabase.table(DOCTORS_TABLE).insert(new_doctor).execute()
        
        if not response.data:
            flash('Error creating account. Please try again.', 'danger')
            return redirect(url_for('signup'))
            
        # Get the newly created doctor
        new_doctor_id = response.data[0]['id']
        
        # Log in the new user
        session['doctor_id'] = new_doctor_id
        session['doctor_name'] = name
        session['doctor_email'] = email
        if profile_image_path:
            session['profile_image'] = profile_image_path
        
        flash('Account created successfully!', 'success')
        return redirect(url_for('dashboard'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Find doctor by email
        doctor = get_doctor_by_email(email)
        
        if doctor and verify_password(doctor['password'], password):
            # Login successful
            session['doctor_id'] = doctor['id']
            session['doctor_name'] = doctor['name']
            session['doctor_email'] = doctor['email']
            if doctor['profile_image']:
                session['profile_image'] = doctor['profile_image']
            
            return redirect(url_for('dashboard'))
        else:
            # Login failed
            flash('Invalid email or password. Please try again.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    doctor_id = session.get('doctor_id')
    patients = get_patients_by_doctor(doctor_id)
    
    # Format dates for patients from Supabase and fetch tests for each patient
    for patient in patients:
        if 'last_visit' in patient and patient['last_visit']:
            # Try to parse the ISO format date string to a datetime object
            try:
                if isinstance(patient['last_visit'], str):
                    dt = datetime.fromisoformat(patient['last_visit'].replace('Z', '+00:00'))
                    patient['last_visit_formatted'] = dt.strftime('%Y-%m-%d')
                else:
                    patient['last_visit_formatted'] = datetime.now().strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                patient['last_visit_formatted'] = datetime.now().strftime('%Y-%m-%d')
        else:
            patient['last_visit_formatted'] = datetime.now().strftime('%Y-%m-%d')
        
        # Directly fetch tests with reports for this patient
        try:
            response = supabase.table(TESTS_TABLE).select('*').eq('patient_id', patient['id']).execute()
            tests = response.data if response.data else []
            
            # Filter tests to only include those with valid reports
            patient['tests'] = []
            patient['has_report'] = False
            patient['latest_test_with_report'] = None
            
            for test in tests:
                # Check if the test has a report path
                if test.get('report_path') and os.path.exists(test.get('report_path')):
                    # Mark this patient as having a report
                    patient['has_report'] = True
                    
                    # Track the latest test with a report
                    if not patient['latest_test_with_report'] or test['id'] > patient['latest_test_with_report']['id']:
                        patient['latest_test_with_report'] = test
                
                patient['tests'].append(test)
                
            # Debug output
            if patient['has_report']:
                print(f"Patient {patient['name']} has report in test {patient['latest_test_with_report']['id']}")
                print(f"Report path: {patient['latest_test_with_report']['report_path']}")
            else:
                print(f"Patient {patient['name']} has no reports")
                
        except Exception as e:
            print(f"Error getting tests for patient {patient['id']}: {str(e)}")
            patient['tests'] = []
            patient['has_report'] = False
            
    return render_template('dashboard.html', patients=patients, now=datetime.now())

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    doctor_id = session.get('doctor_id')
    doctor = get_doctor_by_id(doctor_id)
    
    if request.method == 'POST':
        # Prepare updates
        updates = {
            'name': request.form['name'],
            'specialty': request.form.get('specialty', ''),
            'phone': request.form.get('phone', '')
        }
        
        # Handle profile image update if provided
        if 'profile_image' in request.files:
            profile_image = request.files['profile_image']
            if profile_image.filename != '':
                # Delete old image if exists
                if doctor['profile_image']:
                    old_image_path = os.path.join(app.config['PROFILE_IMAGES'], doctor['profile_image'].split('/')[-1])
                    if os.path.exists(old_image_path):
                        os.remove(old_image_path)
                
                # Save new image
                filename = secure_filename(f"{uuid.uuid4()}_{profile_image.filename}")
                image_path = os.path.join(app.config['PROFILE_IMAGES'], filename)
                profile_image.save(image_path)
                updates['profile_image'] = f"profile_images/{filename}"
                session['profile_image'] = updates['profile_image']
        
        # Handle password update if provided
        new_password = request.form.get('new_password')
        if new_password:
            current_password = request.form.get('current_password')
            if verify_password(doctor['password'], current_password):
                updates['password'] = hash_password(new_password)
                flash('Password updated successfully!', 'success')
            else:
                flash('Current password is incorrect. Password not updated.', 'danger')
        
        # Update in Supabase
        supabase.table(DOCTORS_TABLE).update(updates).eq('id', doctor_id).execute()
        
        # Update session
        session['doctor_name'] = updates['name']
        
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', doctor=doctor)

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/new_patient', methods=['GET', 'POST'])
@login_required
def new_patient():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        medical_history = request.form['medical_history']
        additional_notes = request.form.get('additional_notes', '')
        
        # Create new patient
        new_patient = {
            'name': name,
            'age': int(age),
            'gender': gender,
            'medical_history': medical_history,
            'additional_notes': additional_notes,
            'doctor_id': session.get('doctor_id'),
            'last_visit': datetime.utcnow().isoformat(),
            'status': 'Pending'
        }
        
        # Insert into Supabase
        response = supabase.table(PATIENTS_TABLE).insert(new_patient).execute()
        
        if response.data:
            patient_id = response.data[0]['id']
            return redirect(url_for('test', patient_id=patient_id))
        else:
            flash('Error creating patient. Please try again.', 'danger')
            return redirect(url_for('new_patient'))
        
    return render_template('new_patient.html')

@app.route('/delete_patient/<int:patient_id>')
@login_required
def delete_patient(patient_id):
    # Get the patient to confirm they exist
    patient = get_patient_by_id(patient_id)
    
    if not patient:
        flash('Patient not found', 'danger')
        return redirect(url_for('dashboard'))
    
    # Check if the patient belongs to the current doctor
    if patient['doctor_id'] != session.get('doctor_id'):
        flash('You do not have permission to delete this patient', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        # First, get all tests associated with this patient
        tests_response = supabase.table(TESTS_TABLE).select('*').eq('patient_id', patient_id).execute()
        
        if tests_response.data:
            # Delete all test records for this patient
            for test in tests_response.data:
                # Delete the report file if it exists
                if test.get('report_path') and os.path.exists(test.get('report_path')):
                    try:
                        os.remove(test.get('report_path'))
                        print(f"Deleted report file: {test.get('report_path')}")
                    except Exception as e:
                        print(f"Error deleting report file: {str(e)}")
                        
                # Delete test record from database
                supabase.table(TESTS_TABLE).delete().eq('id', test['id']).execute()
                print(f"Deleted test record ID: {test['id']}")
        
        # Finally, delete the patient record
        supabase.table(PATIENTS_TABLE).delete().eq('id', patient_id).execute()
        print(f"Deleted patient ID: {patient_id}")
        
        flash(f"Patient '{patient['name']}' and all associated data have been deleted", 'success')
    except Exception as e:
        print(f"Error deleting patient: {str(e)}")
        flash(f"Error deleting patient: {str(e)}", 'danger')
    
    return redirect(url_for('dashboard'))

@app.route('/test/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def test(patient_id):
    patient = get_patient_by_id(patient_id)
    
    if not patient:
        flash('Patient not found', 'danger')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        test_type = request.form['test_type']
        
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
            
        file = request.files['image']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image with the AI model
        if test_type == 'chest_xray':
            result, result_image_path = process_chest_xray(file_path)
        elif test_type == 'lung_cancer':
            result, result_image_path = process_lung_cancer(file_path)
        elif test_type == 'covid_19':
            result, result_image_path = process_covid_19(file_path)
        else:
            result, result_image_path = {'message': 'Unknown test type'}, None
            
        # Create a new test record
        new_test = {
            'test_type': test_type,
            'image_path': filename,  # Store just the filename, not the full path
            'result': json.dumps(result),
            'result_image_path': result_image_path.split('/')[-1] if result_image_path and '/' in result_image_path else 
                              (result_image_path.split('\\')[-1] if result_image_path else None),
            'patient_id': patient_id,
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Insert into Supabase
        response = supabase.table(TESTS_TABLE).insert(new_test).execute()
        
        if response.data:
            test_id = response.data[0]['id']
            
            # Update patient status
            supabase.table(PATIENTS_TABLE).update({'status': 'In Progress'}).eq('id', patient_id).execute()
            
            # Add the id to new_test for report generation
            new_test['id'] = test_id
            
            # Parse the result for the AI summary
            try:
                result_data = json.loads(new_test['result']) if isinstance(new_test['result'], str) else new_test['result']
            except:
                result_data = {}
            
            # Generate AI summary
            print("Generating AI summary for the test...")
            ai_summary = generate_ai_summary(patient, new_test, result_data)
            if ai_summary:
                print(f"AI summary generated successfully (first 100 chars): {ai_summary[:100]}...")
                new_test['ai_summary'] = ai_summary
                
                # Store AI summary in memory but don't try to save to database yet
                # We'll add this column to the database schema later
            
            # Generate report
            report_generated = False
            try:
                print("Attempting to generate report...")
                report_path = generate_report(patient, new_test)
                print(f"Generated report path: {report_path}")
                
                if report_path and os.path.exists(report_path):
                    # Update the test with the report path
                    update_response = supabase.table(TESTS_TABLE).update({
                        'report_path': report_path
                    }).eq('id', test_id).execute()
                    
                    print(f"Test updated with report: {update_response.data}")
                    report_generated = True
                else:
                    print("Report generation failed or file doesn't exist")
            except Exception as e:
                print(f"Error in report generation process: {str(e)}")
            
            # Always mark patient as completed, regardless of report generation status
            try:
                status_update = supabase.table(PATIENTS_TABLE).update({'status': 'Completed'}).eq('id', patient_id).execute()
                print(f"Patient status updated to Completed: {status_update.data}")
            except Exception as e:
                print(f"Error updating patient status: {str(e)}")
                # One more attempt with basic error handling
                try:
                    supabase.table(PATIENTS_TABLE).update({'status': 'Completed'}).eq('id', patient_id).execute()
                    print("Patient status updated to Completed on second attempt")
                except:
                    print("Failed to update patient status even on second attempt")
                
            return redirect(url_for('test_result', test_id=test_id))
        else:
            flash('Error creating test. Please try again.', 'danger')
            return redirect(url_for('test', patient_id=patient_id))
        
    return render_template('test.html', patient=patient)

@app.route('/test_result/<int:test_id>')
@login_required
def test_result(test_id):
    test = get_test_by_id(test_id)
    
    if not test:
        flash('Test not found', 'danger')
        return redirect(url_for('dashboard'))
    
    # Add debug information about the image path
    if 'image_path' in test:
        print(f"Original image path: {test['image_path']}")
        
    patient = get_patient_by_id(test['patient_id'])
    result = json.loads(test['result']) if isinstance(test['result'], str) else test['result']
    
    # Generate AI summary on-the-fly for viewing (don't store in database yet)
    try:
        print("Generating AI summary for viewing...")
        ai_summary = generate_ai_summary(patient, test, result)
        if ai_summary:
            # Just store in memory for rendering
            test['ai_summary'] = ai_summary
            print("AI summary generated for display")
    except Exception as e:
        print(f"Error generating AI summary: {str(e)}")
    
    return render_template('test_result.html', test=test, patient=patient, result=result)

@app.route('/view_report/<int:test_id>')
@login_required
def view_report(test_id):
    print(f"Attempting to view report for test ID: {test_id}")
    
    # Get the test directly from Supabase to ensure we have the most up-to-date data
    response = supabase.table(TESTS_TABLE).select('*').eq('id', test_id).execute()
    if not response.data:
        flash('Test not found', 'danger')
        return redirect(url_for('dashboard'))
    
    test = response.data[0]
    print(f"Found test: {test}")
    
    if not test.get('report_path'):
        flash('No report available for this test', 'danger')
        return redirect(url_for('dashboard'))
    
    report_path = test['report_path']
    print(f"Report path: {report_path}")
    
    # Check if the file exists
    if not os.path.exists(report_path):
        print(f"Report file not found on server at path: {report_path}")
        
        # Try to regenerate the report
        try:
            patient = get_patient_by_id(test['patient_id'])
            print(f"Attempting to regenerate report for patient {patient['name']}")
            
            # Try to regenerate the report
            new_report_path = generate_report(patient, test)
            
            if new_report_path and os.path.exists(new_report_path):
                # Update the test with the new report path
                supabase.table(TESTS_TABLE).update({
                    'report_path': new_report_path
                }).eq('id', test_id).execute()
                
                report_path = new_report_path
                print(f"Successfully regenerated report at {new_report_path}")
            else:
                flash('Report file not found and could not be regenerated', 'danger')
                return redirect(url_for('dashboard'))
        except Exception as e:
            print(f"Error regenerating report: {str(e)}")
            flash(f'Report file not found and regeneration failed: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))
    
    print(f"Report file exists, attempting to send")
        
    try:
        # Get patient info for the report name
        patient = get_patient_by_id(test['patient_id'])
        patient_name = patient['name'] if patient else 'Unknown'
        
        # Create a more descriptive filename
        filename = f"Medical_Report_{patient_name.replace(' ', '_')}_Test{test_id}.pdf"
        
        # Send the file
        return send_file(report_path, as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"Error sending report: {str(e)}")
        flash(f'Error downloading report: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/add_notes/<int:test_id>', methods=['POST'])
@login_required
def add_notes(test_id):
    test = get_test_by_id(test_id)
    
    if not test:
        flash('Test not found', 'danger')
        return redirect(url_for('dashboard'))
        
    doctor_notes = request.form['doctor_notes']
    
    # Update test with doctor's notes
    updates = {'doctor_notes': doctor_notes}
    
    # Regenerate report with doctor's notes
    patient = get_patient_by_id(test['patient_id'])
    
    # Update test object with doctor_notes for report generation
    test_with_notes = test.copy()
    test_with_notes['doctor_notes'] = doctor_notes
    
    # Parse the result for the AI summary
    try:
        result_data = json.loads(test['result']) if isinstance(test['result'], str) else test['result']
    except:
        result_data = {}
    
    # Regenerate AI summary with doctor's notes
    print("Regenerating AI summary with doctor's notes...")
    ai_summary = generate_ai_summary(patient, test_with_notes, result_data)
    if ai_summary:
        print(f"New AI summary generated (first 100 chars): {ai_summary[:100]}...")
        # Just store in memory for report generation
        test_with_notes['ai_summary'] = ai_summary
        # Don't try to update the database with ai_summary yet
    
    # Generate updated report
    report_path = generate_report(patient, test_with_notes)
    if report_path:
        updates['report_path'] = report_path
    
    # Update in Supabase
    supabase.table(TESTS_TABLE).update(updates).eq('id', test_id).execute()
    
    # Ensure patient is marked as completed
    try:
        supabase.table(PATIENTS_TABLE).update({'status': 'Completed'}).eq('id', test['patient_id']).execute()
        print(f"Patient {test['patient_id']} status updated to Completed after adding notes")
    except Exception as e:
        print(f"Error updating patient status after adding notes: {str(e)}")
    
    flash('Notes added successfully', 'success')
    
    return redirect(url_for('test_result', test_id=test_id))

def process_chest_xray(image_path):
    """
    Process chest X-ray using the optimized best_chest_model.h5
    Based on reference implementation in models/chest/1.py
    Threshold: 0.35 (optimized for this model)
    """
    if not TENSORFLOW_AVAILABLE:
        return {'error': 'TensorFlow not installed. AI features are disabled.'}, None
        
    if chest_disease_model is None:
        return {'error': 'Model not loaded. Please check if best_chest_model.h5 exists.'}, None
        
    try:
        import cv2
        import numpy as np
        
        # 14 Disease labels - MUST match training order
        DISEASES = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
        
        # Optimized threshold for this model
        THRESHOLD = 0.35
        
        print(f"🔬 Processing chest X-ray: {image_path}")
        
        # Load and preprocess image - EXACTLY as in reference code
        img = cv2.imread(image_path)
        if img is None:
            return {'error': f'Failed to load image: {image_path}'}, None
            
        # Convert BGR to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        # Add batch dimension: (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        
        print(f"✅ Image preprocessed: shape {img.shape}, range [{img.min():.2f}, {img.max():.2f}]")
        
        # Predict using the model
        print("🤖 Running AI prediction...")
        predictions = chest_disease_model.predict(img, verbose=0)[0]
        print(f"✅ Predictions received: {len(predictions)} disease probabilities")
        
        # Process results
        all_predictions = {}
        above_threshold = {}
        detected_diseases = []
        
        for i, disease in enumerate(DISEASES):
            probability = float(predictions[i])
            confidence_percent = probability * 100
            
            all_predictions[disease] = probability
            
            if probability >= THRESHOLD:
                above_threshold[disease] = probability
                detected_diseases.append((disease, confidence_percent))
        
        # Sort by probability (highest first)
        all_predictions = dict(sorted(all_predictions.items(), 
                                     key=lambda item: item[1], 
                                     reverse=True))
        
        above_threshold = dict(sorted(above_threshold.items(), 
                                     key=lambda item: item[1], 
                                     reverse=True))
        
        # Create top 3 conditions
        top_conditions = [
            {'condition': disease, 'probability': prob}
            for disease, prob in list(all_predictions.items())[:3]
        ]
        
        # Log results
        if len(detected_diseases) == 0:
            print("✅ NO DISEASES DETECTED - X-ray appears normal")
        else:
            print(f"⚠️  {len(detected_diseases)} DISEASE(S) DETECTED:")
            for disease, conf in detected_diseases[:5]:  # Show top 5
                print(f"   • {disease}: {conf:.1f}%")
        
        # Build result object
        result = {
            'top_conditions': top_conditions,
            'all_predictions': all_predictions,
            'above_threshold': above_threshold,
            'threshold_used': THRESHOLD,
            'total_detected': len(detected_diseases),
            'model_info': 'best_chest_model.h5 (94.90% accuracy)'
        }
        
        # Generate heatmap visualization
        print("🎨 Generating heatmap visualization...")
        result_image_path = generate_heatmap(image_path, chest_disease_model)
        
        return result, result_image_path
        
    except Exception as e:
        print(f"❌ Error in processing chest X-ray: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, None

def process_lung_cancer(image_path):
    """
    Process lung cancer CT scan using stage2_best.h5 model
    Binary classification: Malignant vs Non-malignant
    Based on reference: models/lung cancer/simple_predict.py
    Accuracy: 96.8%
    """
    if not TENSORFLOW_AVAILABLE:
        return {'error': 'TensorFlow not installed. AI features are disabled.'}, None
        
    # Load model if not already loaded
    global lung_cancer_model
    if 'lung_cancer_model' not in globals() or lung_cancer_model is None:
        lung_cancer_model = load_lung_cancer_model()
    
    if lung_cancer_model is None:
        return {'error': 'Lung cancer model not loaded. Please check if stage2_best.h5 exists.'}, None
        
    try:
        import cv2
        import numpy as np
        from tensorflow import keras
        
        print(f"🔬 Processing lung cancer CT scan: {image_path}")
        
        # Preprocessing pipeline - EXACT as reference (simple_predict.py)
        # Step 1: Load image
        img = keras.utils.load_img(str(image_path))
        arr = keras.utils.img_to_array(img)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        
        # Step 2: RGB to Grayscale
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        
        # Step 3: Resize to 512x512
        resized = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Step 4: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        
        # Step 5: Normalize to [0, 1]
        normalized = enhanced.astype(np.float32) / 255.0
        
        # Step 6: Stack to 3 channels (model expects 3-channel input)
        rgb3 = np.stack([normalized, normalized, normalized], axis=-1)
        
        # Step 7: Add batch dimension
        x = np.expand_dims(rgb3, axis=0)
        
        print(f"✅ Image preprocessed: shape {x.shape}, range [{x.min():.2f}, {x.max():.2f}]")
        
        # Predict using the model
        print("🤖 Running AI prediction...")
        probs = lung_cancer_model.predict(x, verbose=0)[0]
        
        # probs = [p_benign, p_malignant, p_normal]
        p_benign = float(probs[0])
        p_malignant = float(probs[1])
        p_normal = float(probs[2])
        p_non_malignant = p_benign + p_normal
        
        # Binary classification: Malignant vs Non-malignant
        if p_malignant > 0.5:
            prediction = "Malignant"
            confidence = p_malignant
        else:
            prediction = "Non-malignant"
            confidence = p_non_malignant
        
        print(f"✅ Prediction: {prediction} ({confidence * 100:.1f}% confidence)")
        print(f"   Benign: {p_benign * 100:.1f}%")
        print(f"   Malignant: {p_malignant * 100:.1f}%")
        print(f"   Normal: {p_normal * 100:.1f}%")
        
        # Build result object
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'Benign': p_benign,
                'Malignant': p_malignant,
                'Normal': p_normal,
                'Non-malignant': p_non_malignant
            },
            'model_info': 'stage2_best.h5 (96.8% accuracy)',
            'scan_type': 'CT Scan'
        }
        
        # Generate visualization (heatmap for lung cancer)
        print("🎨 Generating visualization...")
        result_image_path = generate_lung_cancer_heatmap(image_path, enhanced)
        
        return result, result_image_path
        
    except Exception as e:
        print(f"❌ Error in processing lung cancer CT scan: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, None

def generate_lung_cancer_heatmap(image_path, processed_image):
    """Generate a visualization showing the processed CT scan"""
    try:
        import cv2
        import numpy as np
        
        # Create a side-by-side visualization
        # Load original
        original = cv2.imread(image_path)
        if original is None:
            return None
            
        # Resize original to match processed
        original_resized = cv2.resize(original, (512, 512))
        
        # Convert processed (grayscale) to BGR for concatenation
        processed_bgr = cv2.cvtColor(processed_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Create side-by-side comparison
        comparison = np.hstack([original_resized, processed_bgr])
        
        # Save the visualization
        filename = f"lung_processed_{os.path.basename(image_path)}"
        viz_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(viz_path, comparison)
        
        print(f"✅ Visualization saved: {viz_path}")
        return viz_path
        
    except Exception as e:
        print(f"Error generating lung cancer visualization: {str(e)}")
        return None

def generate_heatmap(image_path, model):
    try:
        # Try to import OpenCV and NumPy
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("OpenCV or NumPy not found. Heatmap generation disabled.")
            return None
            
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
            
        img = cv2.resize(img, (224, 224))
        
        # Create a simulated heatmap overlay (for demonstration)
        # In a real application with a working model, you'd use Grad-CAM
        
        # Generate a simple heat pattern
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Create a synthetic heatmap focused on a random area of the image
        import random
        focus_x = random.randint(w // 4, 3 * w // 4)
        focus_y = random.randint(h // 4, 3 * h // 4)
        
        heatmap = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                # Distance from focus point
                dist = np.sqrt((x - focus_x)**2 + (y - focus_y)**2)
                # Intensity decreases with distance
                intensity = max(0, 255 - int(dist * 1.5))
                heatmap[y, x] = intensity
        
        # Apply color map
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        superimposed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        
        # Save the heatmap image
        filename = f"heatmap_{os.path.basename(image_path)}"
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(heatmap_path, superimposed_img)
        
        return heatmap_path
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        return None

def generate_report(patient, test):
    if FPDF is None:
        print("FPDF not available. Report generation disabled.")
        return None
    
    # Always create the reports directory    
    os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
    
    # Generate a placeholder file path even before starting
    patient_id = patient.get('id', 'unknown')
    test_id = test.get('id', 'unknown')
    filename = f"report_{patient_id}_{test_id}.pdf"
    report_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
        
    try:
        # Debug information
        print(f"Generating report for patient: {patient['name']}, ID: {patient['id']}")
        print(f"Test ID: {test.get('id')}, Test type: {test.get('test_type')}")
        
        # Parse the test results
        try:
            if isinstance(test.get('result'), str):
                result_data = json.loads(test.get('result', '{}'))
            else:
                result_data = test.get('result', {})
        except Exception as e:
            print(f"Error parsing test results: {str(e)}")
            result_data = {}
        
        # Generate AI summary
        print("Generating AI summary using local LLM...")
        ai_summary = generate_ai_summary(patient, test, result_data)
        print(f"Generated AI summary: {ai_summary[:100]}...")
        
        # Helper function to sanitize text for PDF (remove Unicode characters)
        def sanitize_for_pdf(text):
            """Remove Unicode characters that can't be encoded in latin-1"""
            if not text:
                return ""
            # Convert to string if not already
            text = str(text)
            # Replace common Unicode characters with ASCII equivalents
            replacements = {
                '•': '-', '→': '->', '–': '-', '—': '-',
                '"': '"', '"': '"', ''': "'", ''': "'",
                '…': '...', '°': ' degrees', '×': 'x',
                '≥': '>=', '≤': '<=', '±': '+/-'
            }
            for unicode_char, ascii_char in replacements.items():
                text = text.replace(unicode_char, ascii_char)
            # Remove any remaining non-ASCII characters
            return text.encode('ascii', 'ignore').decode('ascii')
        
        # Create a PDF report
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'MediMind AI - Medical Report', 0, 1, 'C')
        
        # Patient information
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Patient Information", 0, 1)
        pdf.set_font('Arial', '', 10)
        
        # Safely access patient information with sanitization
        pdf.cell(0, 6, sanitize_for_pdf(f"Name: {patient.get('name', 'Unknown')}"), 0, 1)
        pdf.cell(0, 6, sanitize_for_pdf(f"Age: {patient.get('age', 'Unknown')}"), 0, 1)
        pdf.cell(0, 6, sanitize_for_pdf(f"Gender: {patient.get('gender', 'Unknown')}"), 0, 1)
        pdf.cell(0, 6, sanitize_for_pdf(f"Medical History: {patient.get('medical_history', 'None')}"), 0, 1)
        
        # Test results
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, sanitize_for_pdf(f"Test Results - {test.get('test_type', 'Unknown').replace('_', ' ').title()}"), 0, 1)
        pdf.set_font('Arial', '', 10)
        
        # Format and add the test results
        try:
            result = json.loads(test.get('result', '{}'))
            if 'error' in result:
                pdf.cell(0, 6, sanitize_for_pdf(f"Error: {result['error']}"), 0, 1)
            elif 'prediction' in result:
                # Simple prediction (Breast Cancer, COVID-19, Lung Cancer)
                pdf.cell(0, 6, sanitize_for_pdf(f"Prediction: {result['prediction']}"), 0, 1)
                pdf.cell(0, 6, sanitize_for_pdf(f"Confidence: {result.get('confidence', 0)*100:.2f}%"), 0, 1)
            elif 'top_conditions' in result:
                # Chest X-Ray (14 diseases)
                pdf.cell(0, 6, "Top 3 Detected Conditions:", 0, 1)
                for item in result['top_conditions']:
                    pdf.cell(0, 6, sanitize_for_pdf(f"  - {item.get('condition', 'Unknown')}: {item.get('probability', 0)*100:.2f}%"), 0, 1)
                
                # Add threshold and model info
                if 'threshold_used' in result:
                    pdf.ln(3)
                    pdf.set_font('Arial', 'I', 9)
                    pdf.cell(0, 6, sanitize_for_pdf(f"Detection Threshold: {result.get('threshold_used', 0.5)*100:.0f}%"), 0, 1)
                    if 'model_info' in result:
                        pdf.cell(0, 6, sanitize_for_pdf(f"Model: {result.get('model_info', 'N/A')}"), 0, 1)
                    pdf.set_font('Arial', '', 10)
                
                # Add all detected diseases (above threshold)
                if 'above_threshold' in result and result['above_threshold']:
                    pdf.ln(5)
                    pdf.set_font('Arial', 'B', 11)
                    pdf.cell(0, 6, sanitize_for_pdf(f"All Detected Conditions ({len(result['above_threshold'])} found):"), 0, 1)
                    pdf.set_font('Arial', '', 10)
                    for disease, prob in result['above_threshold'].items():
                        pdf.cell(0, 6, sanitize_for_pdf(f"  - {disease}: {prob*100:.2f}%"), 0, 1)
                elif 'total_detected' in result and result['total_detected'] == 0:
                    pdf.ln(3)
                    pdf.set_font('Arial', 'B', 11)
                    pdf.cell(0, 6, "Result: No diseases detected - X-ray appears normal", 0, 1)
                    pdf.set_font('Arial', '', 10)
            else:
                pdf.cell(0, 6, "No specific results found in the test data.", 0, 1)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            pdf.cell(0, 6, f"Error parsing test results: {str(e)}", 0, 1)
        
        # Add images
        image_path = test.get('image_path')
        if image_path:
            # Convert filename to full path
            full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
            print(f"Checking for original image at: {full_image_path}")
            
            if os.path.exists(full_image_path):
                pdf.add_page()
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, "Original Image", 0, 1)
                try:
                    print(f"Adding original image from path: {full_image_path}")
                    pdf.image(full_image_path, x=10, y=50, w=90)
                except Exception as img_err:
                    print(f"Error adding image to report: {str(img_err)}")
                    pdf.cell(0, 10, f"Error adding image: {str(img_err)}", 0, 1)
            else:
                print(f"Original image not found at: {full_image_path}")
        
        result_image_path = test.get('result_image_path')
        if result_image_path:
            # Convert filename to full path
            full_result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_image_path)
            print(f"Checking for result image at: {full_result_image_path}")
            
            if os.path.exists(full_result_image_path):
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, "Analysis Visualization", 0, 1)
                try:
                    print(f"Adding result image from path: {full_result_image_path}")
                    pdf.image(full_result_image_path, x=110, y=50, w=90)
                except Exception as img_err:
                    print(f"Error adding heatmap to report: {str(img_err)}")
                    pdf.cell(0, 10, f"Error adding heatmap: {str(img_err)}", 0, 1)
            else:
                print(f"Result image not found at: {full_result_image_path}")
        
        # Doctor's notes
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Doctor's Notes", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, sanitize_for_pdf(test.get('doctor_notes', "No notes provided.")))
        
        # Add AI Summary Section - use either a provided summary or generate one on-the-fly
        summary_text = None
        
        # Check if we have an AI summary already
        if test.get('ai_summary'):
            summary_text = test.get('ai_summary')
        # Otherwise, try to generate one now
        else:
            try:
                # Make sure result_data is available
                if not result_data and test.get('result'):
                    try:
                        if isinstance(test.get('result'), str):
                            result_data = json.loads(test.get('result'))
                        else:
                            result_data = test.get('result')
                    except:
                        result_data = {}
                
                # Only try to generate if we have the Ollama integration
                if OLLAMA_AVAILABLE and result_data:
                    summary_text = generate_ai_summary(patient, test, result_data)
                    if summary_text:
                        print("Generated AI summary for report")
            except Exception as e:
                print(f"Error generating AI summary for report: {str(e)}")
                
        if summary_text:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "AI Assisted Summary", 0, 1)
            
            pdf.set_font('Arial', '', 10)
            
            # Format the AI summary for the PDF
            # Split by lines and handle markdown-style formatting
            lines = summary_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    pdf.ln()
                    continue
                
                # Replace Unicode characters with ASCII-safe alternatives
                # This prevents 'latin-1' codec errors
                line = line.replace('•', '-')  # Bullet point
                line = line.replace('→', '->')  # Arrow
                line = line.replace('–', '-')   # En dash
                line = line.replace('—', '-')   # Em dash
                line = line.replace('"', '"')   # Smart quotes
                line = line.replace('"', '"')
                line = line.replace(''', "'")
                line = line.replace(''', "'")
                line = line.replace('…', '...')  # Ellipsis
                
                # Remove any remaining non-ASCII characters
                line = line.encode('ascii', 'ignore').decode('ascii')
                    
                # Handle markdown headings
                if line.startswith('# '):
                    pdf.set_font('Arial', 'B', 12)
                    pdf.multi_cell(0, 6, line[2:])
                    pdf.set_font('Arial', '', 10)
                elif line.startswith('## '):
                    pdf.set_font('Arial', 'B', 11)
                    pdf.multi_cell(0, 6, line[3:])
                    pdf.set_font('Arial', '', 10)
                elif line.startswith('### '):
                    pdf.set_font('Arial', 'BI', 10)
                    pdf.multi_cell(0, 6, line[4:])
                    pdf.set_font('Arial', '', 10)
                # Handle markdown lists
                elif line.startswith('- ') or line.startswith('* '):
                    pdf.multi_cell(0, 6, '  - ' + line[2:])  # Use ASCII dash instead of bullet
                elif line[0:2].isdigit() and line[2:4] in ['. ', ') ']:
                    pdf.multi_cell(0, 6, line)
                else:
                    pdf.multi_cell(0, 6, line)
            
            pdf.ln()
            pdf.set_font('Arial', 'I', 8)
            pdf.multi_cell(0, 4, "Note: This AI summary is generated to assist healthcare professionals and should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment decisions.")
        
        # Add date of report
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
        
        # Save the PDF
        patient_id = patient.get('id', 'unknown')
        test_id = test.get('id', 'unknown')
        filename = f"report_{patient_id}_{test_id}.pdf"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
        
        # Make sure the REPORTS_FOLDER exists
        os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
        
        print(f"Saving report to: {report_path}")
        pdf.output(report_path)
        
        # Verify the file was created
        if os.path.exists(report_path):
            print(f"Report successfully created at {report_path}")
            return report_path
        else:
            print(f"Failed to create report at {report_path}")
            
            # Create a simple emergency report if the main one failed
            try:
                simple_pdf = FPDF()
                simple_pdf.add_page()
                simple_pdf.set_font('Arial', 'B', 16)
                simple_pdf.cell(0, 10, 'MediMind AI - Basic Medical Report', 0, 1, 'C')
                simple_pdf.set_font('Arial', '', 12)
                simple_pdf.cell(0, 10, f"Patient: {patient.get('name')}", 0, 1)
                simple_pdf.cell(0, 10, f"Test Type: {test.get('test_type', 'Unknown').replace('_', ' ').title()}", 0, 1)
                simple_pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
                simple_pdf.cell(0, 10, "This is a basic report. The detailed report could not be generated.", 0, 1)
                simple_pdf.output(report_path)
                
                if os.path.exists(report_path):
                    print(f"Created simple emergency report at {report_path}")
                    return report_path
            except Exception as e:
                print(f"Failed to create emergency report: {str(e)}")
            
            return None
            
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        
        # Try to generate a simple emergency report
        try:
            simple_pdf = FPDF()
            simple_pdf.add_page()
            simple_pdf.set_font('Arial', 'B', 16)
            simple_pdf.cell(0, 10, 'MediMind AI - Basic Medical Report', 0, 1, 'C')
            simple_pdf.set_font('Arial', '', 12)
            simple_pdf.cell(0, 10, f"Patient: {patient.get('name')}", 0, 1)
            simple_pdf.cell(0, 10, f"Test Type: {test.get('test_type', 'Unknown').replace('_', ' ').title()}", 0, 1)
            simple_pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
            simple_pdf.cell(0, 10, f"Error: {str(e)}", 0, 1)
            simple_pdf.output(report_path)
            
            if os.path.exists(report_path):
                print(f"Created simple emergency report at {report_path}")
                return report_path
        except:
            pass
            
        return None


# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Check if the filename contains path separators and extract just the filename if needed
    if '/' in filename:
        filename = filename.split('/')[-1]
    elif '\\' in filename:
        filename = filename.split('\\')[-1]
        
    # For debugging
    print(f"Serving file: {os.path.join(app.config['UPLOAD_FOLDER'], filename)}")
    
    # Check if file exists
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
    
    return send_file(file_path)


if __name__ == '__main__':
    # Ensure the upload folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROFILE_IMAGES'], exist_ok=True)
    
    app.run(debug=True)
