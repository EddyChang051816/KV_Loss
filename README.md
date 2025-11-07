# Dental Faster R-CNN Project (Cleaned, Quick Upload)

This repository contains two cleaned scripts provided by the user, prepared for a quick GitHub upload (basic-clean level -- **Option A**).

**Files created**
- `model_test_alg_clean.py` — cleaned test/eval script (removed duplicates, replaced hard paths with config variables).
- `train_faster_rcnn_with_kv_loss_clean.py` — cleaned training script (placeholders for dataset paths, fixed end call, small config).

**What I changed (high level)**
- Removed duplicated function/class definitions in `model_test_alg.py`.
- Replaced hard-coded, user-specific file paths with top-level configuration variables.
- Added `if __name__ == '__main__'` guard to both scripts and warnings for missing paths.
- Replaced literal batch sizes with `BATCH_SIZE` variables to ease tuning.
- Provided placeholder dataset/model paths — **please update them before running**.

**Next steps (recommended)**
1. Update `MODEL_PATH`, `VAL_IMG_DIR`, `VAL_XML_DIR` and training dataset paths in the configuration blocks.
2. Create a `requirements.txt` (suggested: torch, torchvision, pillow, tqdm, pandas, scikit-learn, matplotlib, torchmetrics).
3. Run a small smoke test (1-2 batches) to ensure shapes and device mapping are correct.