import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

print("🎯 ESA Anomaly Detection - Final Testing...")

parser = argparse.ArgumentParser(description="Test Random Forest with different modes")
parser.add_argument("--mode", choices=["low", "medium", "high"], default="medium",
                   help="Detection sensitivity mode")
args = parser.parse_args()

thresholds = {
    "low": 0.5,     
    "medium": 0.8, 
    "high": 0.95   
}

threshold = thresholds[args.mode]
print(f"🛰️ Running in {args.mode.upper()} mode (threshold: {threshold})")

# Load test data features
print("📁 Loading features...")
try:
    features = pd.read_csv("attn_features.csv")
    print(f"Total sequences: {len(features)}")
except FileNotFoundError:
    print("❌ attn_features.csv not found! Run export_attn_features.py first.")
    exit(1)

# Load trained model
print("🤖 Loading trained model...")
try:
    clf = joblib.load("models/rf_downstream.pkl")
except FileNotFoundError:
    print("❌ models/rf_downstream.pkl not found! Run train_rf_on_latent.py first.")
    exit(1)

# Load feature names
try:
    with open("models/rf_features.txt", "r") as f:
        selected_features = [line.strip() for line in f]
    print(f"Model expects {len(selected_features)} features")
except FileNotFoundError:
    print("❌ models/rf_features.txt not found! Run train_rf_on_latent.py first.")
    exit(1)

# Check if all required features exist
missing_features = [f for f in selected_features if f not in features.columns]
if missing_features:
    print(f"❌ Missing features: {missing_features}")
    print("Available features:", list(features.columns))
    exit(1)

X = features[selected_features]
y = features["is_anomaly"]

print(f"✅ Data loaded: {X.shape[0]} samples, {y.sum()} anomalies")

# Use same split as training for consistency
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y))

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

print(f"📊 Test set: {len(X_test)} samples, {y_test.sum()} anomalies")

# Validate input data
if X_test.empty:
    print("❌ Test set is empty!")
    exit(1)

if X_test.isnull().any().any():
    print("⚠️ Found NaN values in test data, filling with 0...")
    X_test = X_test.fillna(0)

# Make predictions
print("🔮 Making predictions...")
try:
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)
    print(f"✅ Predictions completed")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_test dtypes: {X_test.dtypes.unique()}")
    exit(1)

# Comprehensive evaluation
print(f"\n📈 EVALUATION RESULTS ({args.mode.upper()} MODE):")
print("=" * 60)

# Basic metrics
try:
    print("🎯 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))
except Exception as e:
    print(f"❌ Classification report failed: {e}")

# Confusion matrix
try:
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:  # 2x2 matrix
        tn, fp, fn, tp = cm.ravel()
    else:
        print("⚠️ Unusual confusion matrix shape:", cm.shape)
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        tn = np.sum((y_test == 0) & (y_pred == 0))

    print(f"\n🔍 CONFUSION MATRIX:")
    print(f"True Negatives (TN):  {tn:6d} (correct normal)")
    print(f"False Positives (FP): {fp:6d} (false alarms)")
    print(f"False Negatives (FN): {fn:6d} (missed anomalies)")
    print(f"True Positives (TP):  {tp:6d} (caught anomalies)")

    # Detailed metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n📊 DETAILED METRICS:")
    print(f"Precision:    {precision:.3f} ({precision:.1%})")
    print(f"Recall:       {recall:.3f} ({recall:.1%})")
    print(f"F1-Score:     {f1:.3f}")
    print(f"Specificity:  {specificity:.3f}")

    # AUC Score
    if len(np.unique(y_test)) > 1:
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC-ROC:      {auc_score:.3f}")
        except Exception as e:
            print(f"AUC-ROC:      Error calculating ({e})")
    else:
        print("AUC-ROC:      N/A (only one class in test set)")

    # Operational metrics
    print(f"\n🛰️ OPERATIONAL METRICS:")
    false_alarm_rate = fp / len(y_test) * 100
    detection_rate = tp / y_test.sum() * 100 if y_test.sum() > 0 else 0

    print(f"False Alarm Rate:     {false_alarm_rate:.2f}% of all data")
    print(f"Detection Rate:       {detection_rate:.1f}% of anomalies caught")
    print(f"Missed Anomalies:     {fn} out of {y_test.sum()} total")

    # Mode-specific interpretation
    print(f"\n💡 {args.mode.upper()} MODE INTERPRETATION:")
    if args.mode == "low":
        print(f"   • High sensitivity: Catches {recall:.0%} of anomalies")
        print(f"   • Alert frequency: {(tp + fp) / len(y_test) * 1000:.1f} per 1000 periods")
        print(f"   • Best for: Critical mission phases")
    elif args.mode == "medium":
        print(f"   • Balanced operation: {precision:.0%} precision, {recall:.0%} recall")
        print(f"   • Alert frequency: {(tp + fp) / len(y_test) * 1000:.1f} per 1000 periods")
        print(f"   • Best for: Standard operations")
    elif args.mode == "high":
        print(f"   • High precision: {precision:.0%} of alerts are real")
        print(f"   • Conservative: Only {false_alarm_rate:.2f}% false alarm rate")
        print(f"   • Best for: Routine operations")

except Exception as e:
    print(f"❌ Metrics calculation failed: {e}")

print(f"\n✅ Testing completed in {args.mode.upper()} mode")
print(f"🎯 Threshold used: {threshold}")
