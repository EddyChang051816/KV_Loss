# Dental Faster R-CNN Project

This repository contains two scripts for training and evaluating a Faster R-CNN model for dental image analysis.

## Files Included
- `model_test_alg.py` — script for running model testing and evaluation.
- `train_faster_rcnn_with_kv_loss.py` — script for training the Faster R-CNN model with KV loss.
- `requirements.txt` — dependency list for environment setup.

## Setup Instructions

### 1️⃣ Install Dependencies
Recommend using a virtual environment before installation:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Update Configuration
Before running the scripts, update the configuration section at the top of each file:

| Script | Variables to Update |
|--------|--------------------------|
| `model_test_alg.py` | `MODEL_PATH`, `VAL_IMG_DIR`, `VAL_XML_DIR` |
| `train_faster_rcnn_with_kv_loss.py` | `TRAIN_IMG_DIR`, `TRAIN_XML_DIR`, `TEST_IMG_DIR`, `TEST_XML_DIR` |

### 3️⃣ Run Testing / Evaluation

```bash
python model_test_alg.py
```

### 4️⃣ Run Training

```bash
python train_faster_rcnn_with_kv_loss.py
```

Training results (metrics and model weights) will be saved to the paths specified in the script.