import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

def evaluate_mode(mode):
    """Evaluate specific mode performance on ESA satellite telemetry data"""
    
    
    thresholds = {
        "low": 0.5,     
        "medium": 0.8,  
        "high": 0.95    
    }
    
    threshold = thresholds[mode]
    
    print(f"🛰️ ESA Satellite Anomaly Detection - {mode.upper()} MODE")
    print("=" * 60)
    print(f"Detection threshold: {threshold}")
    print()
    
    
    print("📁 Loading model and features...")
    rf_model = joblib.load("models/rf_downstream.pkl")
    
    with open("models/rf_features.txt", "r") as f:
        feature_names = [line.strip() for line in f]
    
    features = pd.read_csv("attn_features.csv")
    X = features[feature_names]
    y = features["is_anomaly"]
    
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    
    print(f"Test set: {len(X_test):,} samples, {y_test.sum()} anomalies")
    print()
    
    
    probabilities = rf_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities > threshold).astype(int)
    
    
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    
    print(f"📈 PERFORMANCE METRICS:")
    print(f"Precision:        {precision:.3f} ({precision:.1%})")
    print(f"Recall:           {recall:.3f} ({recall:.1%})")  
    print(f"F1-Score:         {f1:.3f}")
    print()
    
    print(f"🔍 CONFUSION MATRIX:")
    print(f"True Positives:   {tp:4d} (correctly detected anomalies)")
    print(f"False Positives:  {fp:4d} (false alarms)")
    print(f"False Negatives:  {fn:4d} (missed anomalies)")
    print(f"True Negatives:   {tn:4d} (correctly identified normal)")
    print()
    
    print(f"🛰️ OPERATIONAL IMPACT:")
    total_alerts = tp + fp
    false_alarm_rate = fp / len(y_test) * 100
    alerts_per_1000 = total_alerts / len(y_test) * 1000
    
    print(f"Total Alerts:     {total_alerts} (out of {len(y_test):,} samples)")
    print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
    print(f"Alerts per 1000:  {alerts_per_1000:.1f} monitoring periods")
    print(f"Missed Anomalies: {fn} out of {y_test.sum()} total")
    print()
    
    
    if mode == "low":
        print(f"💡 LOW MODE - High Sensitivity:")
        print(f"   • Best for: Critical mission phases")
        print(f"   • Catches {recall:.0%} of anomalies")
        print(f"   • Generates {alerts_per_1000:.1f} alerts per 1000 periods")
        print(f"   • Use when: Maximum anomaly detection required")
        
    elif mode == "medium":
        print(f"💡 MEDIUM MODE - Balanced Operation:")
        print(f"   • Best for: Standard satellite operations")
        print(f"   • {precision:.0%} of alerts are real anomalies")
        print(f"   • {recall:.0%} anomaly detection rate")
        print(f"   • Use when: Good balance needed")
        
    elif mode == "high":
        print(f"💡 HIGH MODE - High Precision:")
        print(f"   • Best for: Routine operations")
        print(f"   • {precision:.0%} of alerts are real anomalies")
        print(f"   • Only {false_alarm_rate:.2f}% false alarm rate")
        print(f"   • Use when: Minimal operator interruption needed")
    
    print()
    print(f"✅ Evaluation completed for {mode.upper()} mode")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESA Satellite Anomaly Detection System")
    parser.add_argument("--mode", choices=["low", "medium", "high"], required=True,
                       help="Detection sensitivity mode to evaluate")
    
    args = parser.parse_args()
    evaluate_mode(args.mode)
