# Lung Cancer CT Classifier – Malignant vs Non‑malignant (Information & Quick Start)

This document explains what the model is, how it was trained, how it performs, and how to use it in this project or integrate it elsewhere.

## Overview

- Task: Binary decision on chest CT slices: Malignant vs Non‑malignant (Non‑malignant = Benign + Normal).
- Best checkpoint: `checkpoints/stage2_best.h5`
- Backbone: EfficientNetB3 (ImageNet init, include_top=False) + custom classification head.
- Input size: 512×512×3 (channels are stacked grayscale after preprocessing).
- Preprocessing: RGB→Grayscale → Resize 512 → CLAHE → Normalize to [0,1] → Stack to 3 channels.

## Model architecture (training-time, 3-class)

- Base: `EfficientNetB3(include_top=False, weights='imagenet', input_shape=(512,512,3))`
- Head:
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense(512, relu, L2=0.001) → Dropout(0.5)
  - BatchNormalization
  - Dense(256, relu, L2=0.001) → Dropout(0.4)
  - BatchNormalization
  - Dense(128, relu, L2=0.001) → Dropout(0.3)
  - BatchNormalization
  - Dense(3, softmax)

## Training pipeline (summary)

Two-stage fine-tuning (see `step5train(final).py` and updates in Step 7 docs):
- Stage 1 (head training):
  - Freeze base; train the custom head.
  - Moderate augmentation.
- Stage 2 (fine-tuning):
  - Unfreeze the later part of EfficientNet (trainable_from ≈ 150).
  - Lower learning rate; gradient clipping used in later scripts.
  - More aggressive augmentation once stable.

Notes:
- Loss: categorical cross-entropy; some experiments used focal loss when addressing class imbalance.
- Class imbalance: separate scripts exist to augment Benign and use class weights.

## Data preprocessing (critical for accuracy)

Consistent preprocessing is required at training and inference:
- Convert RGB to grayscale
- Resize to 512×512 (cv2.INTER_AREA)
- CLAHE (clipLimit=2.0, tileGridSize=(8,8))
- Normalize to [0,1]
- Stack gray into 3 channels

Tools:
- Training/val creation: `step2_preprocessing_augmentation.py`
- Inference (3-class): `1_final_check.py` applies the same preprocessing by default.
- Inference (binary): `0_final_check.py` applies the same preprocessing by default.

## Validation performance (Malignant vs Non‑malignant)

Evaluated on `preprocessed_data/val` (219 images; Non‑malignant = Benign 24 + Normal 83 = 107; Malignant = 112):
- `checkpoints/stage2_best.h5` (binary mapping at inference)
  - Malignant recall: 106/112 ≈ 94.6%
  - Non‑malignant specificity: 106/107 ≈ 99.1%
  - Overall binary accuracy: (106 + 106) / 219 ≈ 96.8%

This binary mode is enabled via `0_final_check.py` and is recommended when only Malignant vs Not is needed.

## How to use (in this project)

- Binary predictions (Malignant vs Non‑malignant):
  - Script: `0_final_check.py`
  - Quick default run (uses DEFAULT_WEIGHTS and DEFAULT_IMAGES at top of file):
```powershell
python .\0_final_check.py --smoke_test --print_info
```
  - Custom paths and folder accuracy:
```powershell
python .\0_final_check.py --weights "checkpoints\stage2_best.h5" --images "D:\path\to\folder" --assume_label Non-malignant --output_csv "binary_preds.csv"
```
  - Thresholding:
```powershell
python .\0_final_check.py --weights "checkpoints\stage2_best.h5" --images "D:\path\to\folder" --threshold 0.7
```

- Evaluation on val/test sets:
  - Script: `eval_benign_boost_on_test.py` (works for any weights; now supports fallback to val):
```powershell
python .\eval_benign_boost_on_test.py --weights "checkpoints\stage2_best.h5" --test_dir "preprocessed_data\test" --fallback_to_val
```
  - Outputs CSVs in `evaluation_results/` with model and split in the filename.

## Integration into another project

- Copy `0_final_check.py` and the `.h5` weights into your project.
- In code:
```python
from pathlib import Path
from 0_final_check import LungCancerBinaryPredictor

predictor = LungCancerBinaryPredictor("checkpoints/stage2_best.h5")
print(predictor.get_info())
res = predictor.predict_image(Path("path/to/image.png"), threshold=0.5)
print(res["binary_label"], res["p_malignant"], res["p_non_malignant"]) 
```
- Or use it as a CLI (see examples above). The script documents classes, preprocessing, and decision rules.

## Requirements

- Python 3.10+
- TensorFlow 2.14 (or compatible 2.x)
- numpy, opencv-python, pillow, pandas, scikit-learn (for evaluation scripts)
- On Windows PowerShell, paths should be quoted if they contain spaces.

Install from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

## Known limitations

- Binary mapping doesn’t separate Benign vs Normal; both are reported as Non‑malignant.
- The `.h5` files are weights-only; we rebuild the architecture in scripts before loading.
- For best results, always use the training-like preprocessing at inference.

## Next steps

- If you need explicit Benign vs Normal separation in the future, retraining or fine‑tuning is required; for binary detection, current setup is sufficient.

## Time to train

- Full end-to-end training time depends heavily on hardware (CPU vs GPU) and is not measured end-to-end in this workspace.
- Expect hours on CPU; significantly faster on a modern GPU. Evaluations in this repo complete in seconds per split.

## File index (key scripts)

- `0_information.md`: This document.
- `0_final_check.py`: Portable binary predictor (Malignant vs Non‑malignant) with CLI & importable class.
- `eval_benign_boost_on_test.py`: Evaluate any weights on a val/test split and save metrics.
- `step2_preprocessing_augmentation.py`: Build `preprocessed_data/train` and `preprocessed_data/val` with preprocessing & augmentation.
- `step6_evaluation.py`: Detailed evaluation of a given model on validation/test data.
- `checkpoints/stage2_best.h5`: Best available checkpoint (weights-only) used in this project.

---

If you want this turned into a PDF or a shorter README variant for sharing, I can generate that too.
