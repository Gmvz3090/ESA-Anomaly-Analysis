import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

FEATURES_CSV = "attn_features.csv"
MODEL_PATH = "models/rf_downstream.pkl"
WINDOW_SIZE = 1000
REPORT_PATH = "report.csv"

df = pd.read_csv(FEATURES_CSV)
df = df[df["mse_total"] > 10.0]

X = df[[c for c in df.columns if c.startswith("z_")]]
y_true = df["is_anomaly"]
clf = joblib.load(MODEL_PATH)
y_pred = clf.predict(X)

rows = []
for i in range(0, len(y_pred), WINDOW_SIZE):
    y_window = y_pred[i:i+WINDOW_SIZE]
    start = i
    end = min(i + WINDOW_SIZE - 1, len(y_pred) - 1)
    if y_window.sum() > 0:
        indices = [str(idx) for idx, val in enumerate(y_window, start=i) if val == 1]
        rows.append({"range": f"{start}-{end}", "status": "ANOMALY", "indices": ",".join(indices)})
    else:
        rows.append({"range": f"{start}-{end}", "status": "OK", "indices": ""})

pd.DataFrame(rows).to_csv(REPORT_PATH, index=False)
print(f"✅ Report saved to: {REPORT_PATH}")
