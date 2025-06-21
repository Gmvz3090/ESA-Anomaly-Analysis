import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

def save_detailed_results(features, predictions, probabilities, y_true, y_pred, 
                         test_indices, mode, threshold, output_dir="results"):
    """
    Zapisuje tylko wykryte anomalie w prostym formacie JSON
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    detected_anomalies = []
    
    for i, idx in enumerate(test_indices):
        if y_pred[i] == 1: 
            timestamp_id = test_indices[i] 
            
            detected_anomalies.append({
                "timestamp_id": int(timestamp_id),
                "is_detected_anomaly": True,
                "confidence": float(probabilities[i])
            })
    
    detected_anomalies.sort(key=lambda x: x["timestamp_id"])
    
    anomaly_report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "threshold": float(threshold),
            "total_detected_anomalies": len(detected_anomalies)
        },
        "detected_anomalies": detected_anomalies
    }
    
    # Zapisz plik z wykrytymi anomaliami
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    anomaly_file = f"{output_dir}/detected_anomalies_{mode}_{timestamp}.json"
    
    with open(anomaly_file, 'w', encoding='utf-8') as f:
        json.dump(anomaly_report, f, indent=2, ensure_ascii=False)
    
    # Zachowaj summary z metrykami (dla Ciebie)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    summary_file = f"{output_dir}/summary_{mode}_{timestamp}.json"
    summary = {
        "mode": mode,
        "threshold": float(threshold),
        "performance": {
            "precision": float(precision),
            "recall": float(recall),
            "detected_anomalies": int(tp + fp),  
            "true_anomalies_in_data": int(tp + fn)  
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return anomaly_file, summary_file

def get_classification_type(true_label, pred_label):
    """Określa typ klasyfikacji"""
    if true_label == 1 and pred_label == 1:
        return "true_positive"
    elif true_label == 0 and pred_label == 0:
        return "true_negative"
    elif true_label == 0 and pred_label == 1:
        return "false_positive"
    else:
        return "false_negative"

def get_risk_level(probability, threshold):
    """Ocenia poziom ryzyka na podstawie prawdopodobieństwa"""
    if probability >= 0.95:
        return "CRITICAL"
    elif probability >= threshold:
        return "HIGH"
    elif probability >= threshold * 0.7:
        return "MEDIUM"
    elif probability >= threshold * 0.3:
        return "LOW"
    else:
        return "MINIMAL"

def get_mode_recommendation(precision, recall, false_positives, total_samples):
    """Generuje rekomendacje na podstawie wyników"""
    false_alarm_rate = false_positives / total_samples
    
    if precision < 0.3:
        return "Consider HIGH mode - too many false alarms"
    elif recall < 0.7 and false_alarm_rate < 0.05:
        return "Consider LOW mode - missing too many anomalies"
    elif 0.4 <= precision <= 0.6 and 0.8 <= recall <= 0.95:
        return "MEDIUM mode optimal for your use case"
    else:
        return "Current mode performance acceptable"

def print_enhanced_analysis(summary_data, detected_timestamps, mode):
    """Wyświetla rozszerzoną analizę wyników"""
    print(f"\n🎯 ENHANCED ANALYSIS ({mode.upper()} MODE):")
    print("=" * 70)
    
    perf = summary_data["performance"]
    
    print(f"📊 DETECTION EFFECTIVENESS:")
    print(f"   • Anomalies detected:   {perf['detected_anomalies']:3d}")
    print(f"   • Alert reliability:    {perf['precision']*100:.1f}%")
    print(f"   • Coverage:             {perf['recall']*100:.1f}%")
    
    if detected_timestamps:
        print(f"\n📋 DETECTED ANOMALY TIMESTAMPS:")
        print(f"   • First detected: {min(detected_timestamps)}")
        print(f"   • Last detected:  {max(detected_timestamps)}")
        print(f"   • Sample IDs:     {detected_timestamps[:5]}{'...' if len(detected_timestamps) > 5 else ''}")

# Main execution
print("🎯 ESA Anomaly Detection - Enhanced Final Testing...")

parser = argparse.ArgumentParser(description="Test Random Forest with comprehensive analysis")
parser.add_argument("--mode", choices=["low", "medium", "high"], default="medium",
                   help="Detection sensitivity mode")
parser.add_argument("--output-dir", default="results",
                   help="Output directory for detailed results")
parser.add_argument("--save-results", action="store_true", default=True,
                   help="Save detailed JSON results")
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
    
    # Data quality check
    if features.isnull().any().any():
        print("⚠️ Found NaN values, filling with appropriate defaults...")
        # Fill numeric columns with 0, categorical with mode
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
except FileNotFoundError:
    print("❌ attn_features.csv not found! Run export_attn_features.py first.")
    exit(1)
except Exception as e:
    print(f"❌ Error loading features: {e}")
    exit(1)

# Load trained model
print("🤖 Loading trained model...")
try:
    clf = joblib.load("models/rf_downstream.pkl")
    print(f"✅ Model loaded: {type(clf).__name__}")
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

# Validate features
missing_features = [f for f in selected_features if f not in features.columns]
if missing_features:
    print(f"❌ Missing features: {missing_features}")
    print("Available features:", list(features.columns))
    exit(1)

# Prepare data
X = features[selected_features]
y = features["is_anomaly"]

print(f"✅ Data loaded: {X.shape[0]} samples, {y.sum()} anomalies ({y.mean()*100:.2f}%)")

# Consistent split with training
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y))

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

print(f"📊 Test set: {len(X_test)} samples, {y_test.sum()} anomalies ({y_test.mean()*100:.2f}%)")

# Final validation
if X_test.empty:
    print("❌ Test set is empty!")
    exit(1)

# Make predictions
print("🔮 Making predictions...")
try:
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)
    print(f"✅ Predictions completed")
    print(f"   • Probability range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
    print(f"   • Predictions above threshold: {y_pred.sum()}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_test dtypes: {X_test.dtypes.unique()}")
    exit(1)

# Standard evaluation
print(f"\n📈 EVALUATION RESULTS ({args.mode.upper()} MODE):")
print("=" * 60)

try:
    print("🎯 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))
    
    # Enhanced confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n🔍 CONFUSION MATRIX ANALYSIS:")
        print(f"True Negatives (TN):  {tn:6d} (correctly identified normal)")
        print(f"False Positives (FP): {fp:6d} (false alarms - normal classified as anomaly)")
        print(f"False Negatives (FN): {fn:6d} (missed anomalies - anomaly classified as normal)")
        print(f"True Positives (TP):  {tp:6d} (correctly caught anomalies)")
        
        # AUC Score
        if len(np.unique(y_test)) > 1:
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
                print(f"\n📊 AUC-ROC Score: {auc_score:.3f}")
            except Exception as e:
                print(f"AUC-ROC calculation failed: {e}")
    
    # Save detailed results if requested
    if args.save_results:
        print(f"\n💾 Saving detailed results...")
        try:
            anomaly_file, summary_file = save_detailed_results(
                features, y_pred, y_pred_proba, y_test, y_pred, 
                test_idx, args.mode, threshold, args.output_dir
            )
            print(f"✅ Anomaly detections saved: {anomaly_file}")
            print(f"✅ Summary saved: {summary_file}")
            
            # Load and display enhanced analysis
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            with open(anomaly_file, 'r', encoding='utf-8') as f:
                anomaly_data = json.load(f)
                detected_timestamps = [a["timestamp_id"] for a in anomaly_data["detected_anomalies"]]
            
            print(f"\n📋 Wykryte anomalie (weryfikacja sortowania):")
            if detected_timestamps:
                print(f"   • Pierwsze 5: {detected_timestamps[:5]}")
                print(f"   • Ostatnie 5: {detected_timestamps[-5:]}")
                print(f"   • Czy posortowane: {detected_timestamps == sorted(detected_timestamps)}")
            else:
                print("   • Brak wykrytych anomalii")
            
            print_enhanced_analysis(summary_data, detected_timestamps, args.mode)
            
        except Exception as e:
            print(f"❌ Failed to save detailed results: {e}")
            import traceback
            traceback.print_exc()

except Exception as e:
    print(f"❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n✅ Testing completed in {args.mode.upper()} mode")
print(f"🎯 Threshold used: {threshold}")
print(f"📁 Results saved in: {args.output_dir}/")
