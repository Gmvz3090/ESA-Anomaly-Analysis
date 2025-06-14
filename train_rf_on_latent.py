import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

features = pd.read_csv("attn_features.csv")

filtered = features[features["mse_total"] > 10.0]
print(f"🔍 Trenujemy na {len(filtered)} próbek z dużym błędem")

X = filtered[[f"z_{i}" for i in range(32)]]
y = filtered["is_anomaly"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

acc = clf.score(X_val, y_val)
print(f"✅ Dokładność na walidacji: {acc:.3f}")

joblib.dump(clf, "models/rf_downstream.pkl")
print("💾 Zapisano: models/rf_downstream.pkl")

