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
