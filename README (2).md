# Dental Faster R-CNN Project

This repository contains two scripts for training and evaluating a Faster R-CNN model for dental image analysis.

## Files Included
- `model_test_alg.py` — test/evaluation script (removed duplicates, replaced hard paths with config variables).
- `train_faster_rcnn_with_kv_loss.py` — training script (placeholders for dataset paths, fixed end call, small config).
- `requirements.txt` — dependency list for environment setup.

## Modifications Made
- Removed duplicated function/class definitions in `model_test_alg.py`.
- Replaced hard-coded, user-specific file paths with top-level configuration variables.
- Added `if __name__ == '__main__'` guard to both scripts and warnings for missing paths.
- Replaced literal batch sizes with `BATCH_SIZE` variables to allow easier tuning.
- Provided placeholder dataset/model paths — **please update them before running**.

## Next Steps (Recommended)
1. Update `MODEL_PATH`, `VAL_IMG_DIR`, `VAL_XML_DIR` and training dataset paths in the configuration blocks.
2. Install dependencies using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Run a small smoke test (1–2 batches) to ensure shapes and device mapping are correct.