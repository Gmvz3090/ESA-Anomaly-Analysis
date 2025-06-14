# 🚀 ESA-Anomaly-Analysis

An advanced anomaly detection pipeline for multivariate satellite telemetry, using a GRU-based autoencoder with attention and a downstream Random Forest classifier.

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-informational)]()
[![Model](https://img.shields.io/badge/Model-GRU%20%2B%20Attention-green)]()
[![Status](https://img.shields.io/badge/Status-Trained%20%26%20Ready-brightgreen)]()

---

## 📦 Overview

This project provides a complete and ready-to-use anomaly detection solution for ESA-like satellite telemetry data. It combines:

- GRU + Attention Autoencoder
- MSE-based anomaly filtering
- Random Forest trained on latent representations
- Window-based anomaly reporting with high recall and 0% false positives

---

## 🧠 What’s Inside

- ✅ Ready-trained model (80% recall, 0% false positives)
- ✅ Easy retraining on your own data
- ✅ Windowed anomaly reports in `report.csv`

---

## 📁 Folder Structure

```
ESA-Anomaly-Analysis/
├── 3_months.train.csv         # Training data (from MediaFire)
├── 3_months.test.csv          # Testing data (from MediaFire)
├── attention_gru_autoencoder.py
├── training_gru_attn.py
├── export_attn_features.py
├── train_rf_on_latent.py
├── test_rf_downstream.py
├── models/
├── run_pipeline.sh
├── README.md
└── requirements.txt
```

---

## 📥 Download Demo Dataset

Test and training data used in this project:

🔗 https://www.mediafire.com/file/t98hqv8nir414pe/DataRar.rar/file

After downloading and extracting, make sure the following files are in the root directory:

- `3_months.train.csv`
- `3_months.test.csv`

---

## 🔄 Full Pipeline Usage

```bash
# Step 1: Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Step 2: Install requirements
pip install -r requirements.txt

# Step 3: Run the full pipeline
bash run_pipeline.sh
```

Or run each step manually:

```bash
python training_gru_attn.py
python export_attn_features.py
python train_rf_on_latent.py
python test_rf_downstream.py
```

---

## 📊 Output: `report.csv`

Anomalies are reported in sliding windows of 1000 samples:

```
range,status,indices
0-999,OK,
1000-1999,ANOMALY,1099,1123
2000-2999,OK,
```

---

## 📚 Source Datasets

This project is based on:
- 📡 [Zenodo: ESA Mission1](https://zenodo.org/records/12528696)
- 🛠 [ESA-ADB preprocessing repo](https://github.com/kplabs-pl/ESA-ADB)

---

## 👤 Author

Built by **PXRLO**  
Feel free to contribute, open issues, or suggest improvements.

---


---

## 🏗️ Own Training (Pipeline)

If you'd like to train the model yourself from scratch using your own healthy telemetry data:

### 🔧 Prepare training data:
- Format: `.csv` with numerical columns (`channel_1`, ..., `telecommand_*`)
- Make sure it contains **no anomalies** (or remove anomaly rows)

### 🧪 Steps:

```bash
# Train the autoencoder model
python training_gru_attn.py

# Extract latent vectors + errors from test data
python export_attn_features.py

# Train Random Forest classifier on high-error samples
python train_rf_on_latent.py

# Generate report based on trained RF
python test_rf_downstream.py
```

You can also run:

```bash
bash run_pipeline.sh
```

This runs the full pipeline on the demo dataset.

---

## ⚡ Use This Model (Pretrained)

If you simply want to **use the pretrained model** to analyze new telemetry:

### 📁 Prepare your `.csv` file:
- Format: same columns as `3_months.test.csv` (no `is_anomaly_*` needed)
- Example: `my_telemetry.csv`

### 🚀 Analyze:

1. Replace file path in `export_attn_features.py`:
   ```python
   TEST = "my_telemetry.csv"
   ```

2. Run feature extraction:
   ```bash
   python export_attn_features.py
   ```

3. Run detection with pretrained RandomForest:
   ```bash
   python test_rf_downstream.py
   ```

Check `report.csv` for window-based anomaly summary.

---

## 🧠 How the Model Works

This pipeline uses a hybrid approach:

### 1. GRU + Attention Autoencoder
- Learns to reconstruct time windows of telemetry (sliding sequences)
- Uses **GRU** to process sequences and **attention** to focus on the most relevant time steps
- Calculates **MSE** per sequence (and per channel)

### 2. MSE-Based Filtering
- Only samples with high total MSE (`>10`) are considered as *suspicious*

### 3. Downstream Random Forest
- Latent vectors (`z_0` to `z_31`) from encoder are used to train a Random Forest
- Classifies whether high-error sample is truly anomalous or not

### 4. Reporting
- All predictions are grouped into **1000-sample windows**
- Each window is flagged as:
  - `OK` (if no anomaly detected)
  - `ANOMALY` with exact row indices

This design:
- Reduces false positives dramatically (0%)
- Retains high recall (80%+)
- Enables real-world batch monitoring

---
