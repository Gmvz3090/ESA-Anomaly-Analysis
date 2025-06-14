# 🛰️ ESA Anomaly Analysis

A complete pipeline for **unsupervised anomaly detection** in satellite telemetry using:
- GRU + Attention AutoEncoder (sequence reconstruction)
- MSE-based anomaly scoring
- Downstream Random Forest classifier (on latent vectors)
- Windowed analysis + CSV reporting

---

## 🔧 Quickstart (Zero-to-Result)

1. **Clone the project**
```bash
git clone https://github.com/PXRLO/ESA-Anomaly-Analysis.git
cd ESA-Anomaly-Analysis
```

2. **Download training/test data**  
📦 From MediaFire:  
https://www.mediafire.com/file/t98hqv8nir414pe/DataRar.rar/file

Place `3_months.train.csv` and `3_months.test.csv` in the root of the project.

3. **Create environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

4. **Run the pipeline on the provided data**
```bash
python training_gru_attn.py             # Train GRU+Attention AutoEncoder
python export_attn_features.py          # Extract MSE + latent features
python train_rf_on_latent.py            # Train downstream Random Forest
python test_rf_downstream.py            # Generate report.csv
```

---

## ⚠️ Important: When You Must Retrain RF

> If you run `export_attn_features.py` on new data (even test data), you **must re-train the Random Forest classifier** using:

```bash
python train_rf_on_latent.py
```

Otherwise, `test_rf_downstream.py` will fail to detect anomalies due to mismatch in feature distributions.

---

## 🧠 How It Works

1. **GRU AutoEncoder + Attention**
   - Learns to reconstruct 10-timestep windows of telemetry
   - Computes MSE per channel and sequence
   - Captures latent representation `z_0 ... z_31`

2. **Anomaly Scoring**
   - Only samples with MSE > 10.0 are flagged as "suspicious"

3. **Downstream Classifier**
   - Trained on `z_0 ... z_31` to detect true anomalies
   - Uses Random Forest for robustness and precision

4. **Windowed Reporting**
   - Results grouped into 1000-row blocks
   - Report in `report.csv`: shows which windows are anomalous

---

## 📦 File Overview

| File                        | Purpose                                 |
|-----------------------------|------------------------------------------|
| `training_gru_attn.py`      | Train GRU + attention autoencoder       |
| `export_attn_features.py`   | Generate latent vectors and MSE         |
| `train_rf_on_latent.py`     | Train RandomForest on high-error data   |
| `test_rf_downstream.py`     | Predict and generate final `report.csv` |
| `attn_features.csv`         | Intermediate features for classification |
| `models/*.pt / .pkl / .txt` | Saved PyTorch + RF + threshold           |

---

## ✅ Example Output: `report.csv`

```csv
range,status,indices
0-999,OK,
1000-1999,ANOMALY,1103,1129,1171
...
```

---

## 🧪 Evaluation Example

After testing, you'll see in the console:

```
✅ Report saved to: report.csv

📈 Evaluation:
   ✅ True Positives: 683
   ⚠️  False Positives: 0
   ❌ False Negatives: 169
   🎯 Precision: 1.000
   📉 Recall:    0.802
   🏆 F1-score:  0.890
```

---

## 🙋 FAQ

**Q: Can I use this model on my own telemetry data?**  
✅ Yes! Just convert your data to `.csv` format with `channel_*` columns and follow the same steps.

**Q: What if I don’t have `is_anomaly_*` columns?**  
No problem — they are only used for evaluation. The model works without them.

**Q: Is GPU required?**  
Not at all. The pipeline works on CPU too (slower training, same accuracy).

---

## 🏁 Summary

> 🎉 After training, this model achieves:
> - **80%+ recall**
> - **0% false positives**
> - Fully windowed anomaly reports
> - Easy to retrain or extend

---

**Author:** [PXRLO](https://github.com/PXRLO)  
**License:** MIT  
**Built with:** PyTorch, scikit-learn, pandas, NumPy  


---

## 📚 Citation

The data trained on, and tested on is ESA-ADB, the data was processed using their repository.

> Krzysztof Kotowski, Christoph Haskamp, Jacek Andrzejewski, Bogdan Ruszczak, Jakub Nalepa, Daniel Lakey, Peter Collins, Aybike Kolmas, Mauro Bartesaghi, Jose Martínez-Heras, and Gabriele De Canio.  
> **European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry.** arXiv, 2024.  
> [https://doi.org/10.48550/arXiv.2406.17826](https://doi.org/10.48550/arXiv.2406.17826)

