# 🚀 ESA-Analysis – Anomaly Detection with GRU + Attention + Downstream Classifier

A complete anomaly detection pipeline for multivariate telemetry data using:
- GRU-based Autoencoder with attention
- RobustScaler preprocessing
- MSE reconstruction error filtering
- Random Forest classifier trained on latent vectors
- Window-based anomaly reporting

---

## 📂 Project Structure

```
ESA-Analysis/
├── 3_months.train.csv         # Training data (anomaly-free)
├── 3_months.test.csv          # Test data (with labels)
├── attention_gru_autoencoder.py
├── training_gru_attn.py       # Train GRU+Attention model
├── export_attn_features.py    # Extract MSE + latent vectors
├── train_rf_on_latent.py      # Train Random Forest
├── test_rf_downstream.py      # Evaluate + report anomalies
├── models/                    # Trained models and thresholds
├── requirements.txt
└── report.csv                 # Final output (window-based report)
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/ESA-Analysis.git
cd ESA-Analysis

python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 📊 How It Works

### 1. Prepare the data
Ensure the following files exist in the root:
- `3_months.train.csv`: anomaly-free training data
- `3_months.test.csv`: labeled test data with `is_anomaly_*` columns

---

### 2. Train GRU + Attention Autoencoder

```bash
python training_gru_attn.py
```

- Trains on sliding windows from healthy data
- Saves model to `models/gru_attn.pt`
- Computes 99th percentile MSE threshold

---

### 3. Extract features from test set

```bash
python export_attn_features.py
```

- Extracts:
  - Total MSE
  - Per-channel MSE
  - Latent vector `z_0`–`z_31`
- Saves to `attn_features.csv`

---

### 4. Train downstream classifier (RandomForest)

```bash
python train_rf_on_latent.py
```

- Uses only sequences with `mse_total > 10`
- Trains RF to distinguish true vs. false positives
- Saves to `models/rf_downstream.pkl`

---

### 5. Generate final anomaly report

```bash
python test_rf_downstream.py
```

- Classifies sequences with RF
- Groups predictions into 1000-sample windows
- Marks a window as:
  - `OK` — if no anomalies detected
  - `ANOMALY` — if any anomaly detected
- Output: `report.csv` with per-window status

---

## 📁 Example: `report.csv`

```
range,status,indices
0-999,OK,
1000-1999,ANOMALY,1088,1123
2000-2999,OK,
```

---

## 🎯 Performance (example)

- Precision: **100%** (no false positives)
- Recall: **80%** (846/852 anomalies detected)
- Window-wise alerting for operational use

---

## 📦 Dependencies

```txt
torch>=2.0
numpy
pandas
scikit-learn
joblib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## ✨ Credits

Built by [Your Name] for ESA-like anomaly detection challenges.  
Trained and tested on multivariate telemetry with synthetic anomaly injection.

---

## 📜 License

MIT License
