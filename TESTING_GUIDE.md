# 🧪 MediMind AI - Complete Testing Guide

## 📋 Overview

This guide walks you through testing both updated models:
1. **Chest X-Ray** - 14 disease classification (94.90% accuracy)
2. **Lung Cancer CT Scan** - Malignant/Non-malignant classification (96.8% accuracy)

---

## 🚀 Quick Start

### Step 1: Run Model Tests

**Test Chest X-Ray Model:**
```powershell
python test_chest_model.py
```

**Expected Output:**
```
========================================
Chest Disease Model - Comprehensive Test
========================================

✅ Model file exists
✅ Model loaded successfully
✅ Model architecture verified
✅ Image preprocessing working
✅ Predictions working correctly
✅ All 14 disease classes present

🎉 ALL TESTS PASSED!
```

---

**Test Lung Cancer Model:**
```powershell
python test_lung_cancer_model.py
```

**Expected Output:**
```
========================================
Lung Cancer Model - Comprehensive Test
========================================

✅ Model file: EXISTS
✅ TensorFlow: LOADED
✅ Model architecture: BUILT
✅ Model weights: LOADED
✅ Classification labels: 3 classes
✅ Preprocessing: WORKING
✅ Predictions: WORKING
✅ Binary classification: WORKING

🎉 ALL TESTS PASSED!
```

---

### Step 2: Start the Application

```powershell
python app.py
```

**Expected Output:**
```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

---

### Step 3: Test via Web Interface

#### **3.1 Create Account**
1. Go to `http://127.0.0.1:5000`
2. Click "Sign Up"
3. Fill in doctor details:
   - Full Name: "Dr. Test User"
   - Email: test@medimind.ai
   - Password: Test123!
   - Specialization: "Radiology"
   - License Number: "RAD12345"
4. Click "Sign Up"

#### **3.2 Login**
1. Use credentials from signup
2. Should redirect to dashboard

#### **3.3 Add Patient**
1. Click "New Patient"
2. Fill in details:
   - Full Name: "Test Patient"
   - Date of Birth: 1990-01-01
   - Gender: Male/Female
   - Phone: +1234567890
   - Email: patient@test.com
3. Click "Add Patient"

---

## 🫁 Testing Lung Cancer Model

### Test 1: Upload CT Scan

1. From dashboard, click "Run Test"
2. Select the patient you created
3. Click on **"Lung Cancer CT Scan"** card
4. Upload a CT scan image

**Where to get test images:**
- Use any lung CT scan from medical imaging databases
- LIDC-IDRI dataset
- Or use sample from `uploads/` folder

5. Click "Start Analysis"

### Test 2: Verify Results

**Check for:**
- ✅ Prediction shows "Malignant" or "Non-malignant"
- ✅ Confidence percentage displayed
- ✅ Detailed probabilities table (Benign, Malignant, Normal, Non-malignant)
- ✅ Model info shows "stage2_best.h5 (96.8% accuracy)"
- ✅ AI-generated summary appears
- ✅ Original and processed images displayed
- ✅ "Download Report" button works

### Test 3: Check PDF Report

1. Click "Download Report"
2. Open downloaded PDF
3. Verify:
   - Patient information correct
   - Test type: "Lung Cancer CT Scan"
   - Result shows correct classification
   - AI summary included
   - Doctor information present
   - Timestamp accurate

---

## 🫀 Testing Chest X-Ray Model

### Test 1: Upload Chest X-Ray

1. From dashboard, click "Run Test"
2. Select a patient
3. Click on **"Chest X-Ray Analysis"** card
4. Upload a chest X-ray image

**Where to get test images:**
- NIH ChestX-ray14 dataset
- Use sample from `uploads/` folder
- Public chest X-ray databases

5. Click "Start Analysis"

### Test 2: Verify Results

**Check for:**
- ✅ Up to 14 different diseases detected
- ✅ Confidence scores for each disease
- ✅ Only diseases with >35% confidence shown
- ✅ Progress bars for each detected disease
- ✅ Color-coded severity (Red: High, Orange: Medium, Yellow: Low)
- ✅ AI-generated comprehensive summary
- ✅ Model info shows "best_chest_model.h5 (94.90% accuracy)"
- ✅ Heatmap visualization

### Test 3: Check Disease Detection

**14 Possible Diseases:**
1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural Thickening
13. Pneumonia
14. Pneumothorax

**Verify:**
- Multiple diseases can be detected
- Only diseases >35% threshold shown
- Sorted by confidence (highest first)

---

## 🧪 Advanced Testing

### Test Different Image Types

#### For Chest X-Ray:
- **Frontal view** (PA/AP)
- **Lateral view**
- **Portable X-rays**
- Different image qualities

#### For Lung Cancer:
- **Axial CT slices** (best)
- **Coronal CT slices**
- **Sagittal CT slices**
- Different window settings

### Test Edge Cases

1. **Very low quality images**
   - Should still process but lower confidence

2. **Wrong image type**
   - X-ray for lung cancer test → Wrong predictions
   - CT scan for chest X-ray → Wrong predictions

3. **Non-medical images**
   - Should process but give nonsensical results

4. **Multiple tests for same patient**
   - Check history saves correctly
   - Verify all tests appear in patient profile

---

## 📊 Performance Benchmarks

### Expected Processing Times:

| Test Type | Model Load | Image Processing | Prediction | Total |
|-----------|------------|------------------|------------|-------|
| Chest X-Ray | ~2-5s | ~0.1s | ~0.5s | **~3-6s** |
| Lung Cancer | ~3-8s | ~0.3s | ~0.7s | **~4-9s** |

*Times may vary based on hardware*

---

## ✅ Verification Checklist

### Models:
- [ ] `test_chest_model.py` → All tests pass
- [ ] `test_lung_cancer_model.py` → All tests pass
- [ ] Both models load without errors on app start

### Web Interface:
- [ ] Can create account
- [ ] Can login
- [ ] Dashboard shows correctly
- [ ] Can add patient
- [ ] Can edit patient
- [ ] Profile page works

### Chest X-Ray Test:
- [ ] Can upload X-ray image
- [ ] Processes within 10 seconds
- [ ] Shows multiple diseases if present
- [ ] Confidence scores accurate
- [ ] AI summary generates
- [ ] PDF downloads correctly
- [ ] Heatmap visualizes

### Lung Cancer Test:
- [ ] Can upload CT scan
- [ ] Processes within 10 seconds
- [ ] Shows Malignant/Non-malignant
- [ ] Detailed probabilities display
- [ ] AI summary appropriate for result
- [ ] PDF downloads correctly
- [ ] Visualization shows preprocessing

### Database:
- [ ] Tests saved to database
- [ ] Patient history accessible
- [ ] Results persist after logout/login
- [ ] Can view past test results

---

## 🐛 Troubleshooting

### Issue: Model tests fail

**Symptoms:**
```
❌ Model file not found
```

**Solution:**
```powershell
# Check model files exist
ls models/chest/best_chest_model.h5
ls "models/lung cancer/stage2_best.h5"
```

---

### Issue: App won't start

**Symptoms:**
```
Error: No module named 'tensorflow'
```

**Solution:**
```powershell
# Install dependencies
pip install -r requirements.txt
```

---

### Issue: Wrong predictions

**For Chest X-Ray:**
- Ensure using actual chest X-ray (not CT)
- Check image is frontal view
- Verify image quality

**For Lung Cancer:**
- Ensure using CT scan (not X-ray)
- Check image shows lung tissue
- Prefer axial slices

---

### Issue: Low confidence

**Causes:**
- Poor image quality
- Wrong image type
- Image too small/large
- Corrupted file

**Solutions:**
- Use higher quality images
- Verify correct test type
- Try different image

---

### Issue: Database errors

**Symptoms:**
```
supabase.exceptions.APIError
```

**Solution:**
```powershell
# Check Supabase credentials
python setup_supabase.py
```

---

## 📈 Expected Accuracy

### Chest X-Ray Model:
- **Overall Accuracy**: 94.90%
- **Disease Detection**: Multi-label (can detect multiple)
- **Threshold**: 35% (optimized for early detection)
- **Best Performance**: Cardiomegaly, Effusion, Mass

### Lung Cancer Model:
- **Overall Accuracy**: 96.8%
- **Malignant Recall**: 94.6%
- **Non-malignant Specificity**: 99.1%
- **Binary Classification**: Malignant vs Non-malignant

---

## 🎯 Test Scenarios

### Scenario 1: Healthy Patient
**Input**: Normal chest X-ray  
**Expected**: No diseases detected or very low probabilities (<35%)

### Scenario 2: Pneumonia Patient
**Input**: Chest X-ray with pneumonia  
**Expected**: Pneumonia detected with high confidence (>70%)

### Scenario 3: Malignant Lung Tumor
**Input**: CT scan showing malignant tumor  
**Expected**: "Malignant" classification with >80% confidence

### Scenario 4: Benign Lung Nodule
**Input**: CT scan with benign nodule  
**Expected**: "Non-malignant" classification

### Scenario 5: Multiple Chest Conditions
**Input**: X-ray with multiple abnormalities  
**Expected**: Multiple diseases detected above threshold

---

## 📝 Testing Log Template

```
Date: ___________
Tester: _________

✅ Model Tests:
  - Chest: [ ] Pass [ ] Fail
  - Lung:  [ ] Pass [ ] Fail

✅ Web Interface:
  - Signup: [ ] Pass [ ] Fail
  - Login:  [ ] Pass [ ] Fail
  - Dashboard: [ ] Pass [ ] Fail

✅ Chest X-Ray Test:
  - Upload: [ ] Pass [ ] Fail
  - Process: [ ] Pass [ ] Fail
  - Results: [ ] Pass [ ] Fail
  - PDF: [ ] Pass [ ] Fail

✅ Lung Cancer Test:
  - Upload: [ ] Pass [ ] Fail
  - Process: [ ] Pass [ ] Fail
  - Results: [ ] Pass [ ] Fail
  - PDF: [ ] Pass [ ] Fail

Issues Found:
_________________________
_________________________

Notes:
_________________________
_________________________
```

---

## 🎉 Success Criteria

### All tests pass when:
1. ✅ Both model test scripts complete successfully
2. ✅ Application starts without errors
3. ✅ Can perform end-to-end test (signup → add patient → run test → view results)
4. ✅ Both chest X-ray and lung cancer tests work
5. ✅ Results are medically sensible
6. ✅ PDFs generate correctly
7. ✅ Database persists data
8. ✅ AI summaries generate

---

**Ready to Test!** 🚀

Start with the model tests, then move to web interface testing. Good luck! 🍀
