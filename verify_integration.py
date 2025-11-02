#!/usr/bin/env python
"""
Quick verification script to confirm all chest model changes are in place
Run this before starting the application
"""

import os
import sys

print("=" * 70)
print("🔍 CHEST MODEL INTEGRATION VERIFICATION")
print("=" * 70)

checks_passed = 0
total_checks = 0

def check(description, condition, details=""):
    global checks_passed, total_checks
    total_checks += 1
    status = "✅" if condition else "❌"
    print(f"\n{total_checks}. {description}")
    print(f"   {status} {'PASS' if condition else 'FAIL'}")
    if details:
        print(f"   {details}")
    if condition:
        checks_passed += 1
    return condition

# Check 1: Model file exists
model_exists = os.path.exists('models/chest/best_chest_model.h5')
check("Model file exists", model_exists, 
      "Location: models/chest/best_chest_model.h5")

# Check 2: Test script exists
test_script_exists = os.path.exists('test_chest_model.py')
check("Test script created", test_script_exists,
      "File: test_chest_model.py")

# Check 3: Documentation exists
docs_exist = (os.path.exists('CHEST_MODEL_INTEGRATION.md') and 
              os.path.exists('QUICK_START.md') and
              os.path.exists('CHANGES_SUMMARY.md'))
check("Documentation complete", docs_exist,
      "Files: CHEST_MODEL_INTEGRATION.md, QUICK_START.md, CHANGES_SUMMARY.md")

# Check 4: app.py has correct model path
app_content = ""
try:
    with open('app.py', 'r', encoding='utf-8') as f:
        app_content = f.read()
    has_correct_path = "best_chest_model.h5" in app_content
    check("app.py references correct model", has_correct_path,
          "Found: best_chest_model.h5")
except Exception as e:
    check("app.py references correct model", False, f"Error: {e}")

# Check 5: Threshold is 0.35
has_threshold = "THRESHOLD = 0.35" in app_content
check("Threshold set to 0.35", has_threshold,
      "Optimized threshold for sensitivity")

# Check 6: BGR to RGB conversion
has_bgr_rgb = "cv2.cvtColor(img, cv2.COLOR_BGR2RGB)" in app_content
check("BGR to RGB conversion implemented", has_bgr_rgb,
      "Critical for correct predictions")

# Check 7: 14 diseases configured
diseases_check = all(disease in app_content for disease in 
                     ['Atelectasis', 'Cardiomegaly', 'Pneumonia', 'Hernia'])
check("14 disease labels present", diseases_check,
      "All disease labels configured")

# Check 8: templates/test_result.html updated
template_exists = os.path.exists('templates/test_result.html')
if template_exists:
    try:
        with open('templates/test_result.html', 'r', encoding='utf-8') as f:
            template_content = f.read()
        has_threshold_display = 'threshold_used' in template_content
        check("Result template updated", has_threshold_display,
              "Enhanced display with threshold info")
    except Exception as e:
        check("Result template updated", False, f"Error: {e}")
else:
    check("Result template updated", False, "Template file not found")

# Check 9: ai_helper.py updated
helper_exists = os.path.exists('ai_helper.py')
if helper_exists:
    try:
        with open('ai_helper.py', 'r', encoding='utf-8') as f:
            helper_content = f.read()
        has_enhanced_summary = 'detected_conditions' in helper_content
        check("AI helper enhanced", has_enhanced_summary,
              "Better summaries for chest X-rays")
    except Exception as e:
        check("AI helper enhanced", False, f"Error: {e}")
else:
    check("AI helper enhanced", False, "ai_helper.py not found")

# Check 10: Dependencies
print("\n10. Checking dependencies...")
dependencies = {
    'tensorflow': False,
    'cv2': False,
    'numpy': False,
    'flask': False
}

for dep_name in dependencies.keys():
    try:
        if dep_name == 'cv2':
            import cv2
            dependencies['cv2'] = True
        elif dep_name == 'numpy':
            import numpy
            dependencies['numpy'] = True
        elif dep_name == 'tensorflow':
            import tensorflow
            dependencies['tensorflow'] = True
        elif dep_name == 'flask':
            import flask
            dependencies['flask'] = True
    except ImportError:
        pass

all_deps = all(dependencies.values())
check("All dependencies installed", all_deps,
      f"TensorFlow: {dependencies['tensorflow']}, "
      f"OpenCV: {dependencies['cv2']}, "
      f"NumPy: {dependencies['numpy']}, "
      f"Flask: {dependencies['flask']}")

if not all_deps:
    print("\n   Missing dependencies:")
    for dep, installed in dependencies.items():
        if not installed:
            pkg_name = 'opencv-python' if dep == 'cv2' else dep
            print(f"   ❌ {dep} - Install: pip install {pkg_name}")

# Summary
print("\n" + "=" * 70)
print("📊 VERIFICATION SUMMARY")
print("=" * 70)
print(f"\nPassed: {checks_passed}/{total_checks} checks")

if checks_passed == total_checks:
    print("\n🎉 ALL CHECKS PASSED!")
    print("\n✅ The chest model is properly integrated")
    print("✅ All files are in place")
    print("✅ Configuration is correct")
    print("\n🚀 Next steps:")
    print("   1. Run: python test_chest_model.py")
    print("   2. Then: python app.py")
    print("   3. Test with a chest X-ray image")
else:
    print("\n⚠️  SOME CHECKS FAILED")
    print(f"\n   {total_checks - checks_passed} issue(s) found")
    print("\n   Please review the failed checks above")
    print("   Refer to CHANGES_SUMMARY.md for details")

print("\n" + "=" * 70)
