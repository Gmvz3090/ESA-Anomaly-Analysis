import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score
import joblib
import numpy as np

print("🌲 TRAINING RANDOM FOREST (FIXED VERSION)...")

features = pd.read_csv("attn_features.csv")
print(f"📊 Total samples: {len(features)}")
print(f"📊 Anomalies: {features['is_anomaly'].sum()} ({features['is_anomaly'].mean()*100:.2f}%)")

print("✅ Using ALL data (no arbitrary filtering)")

latent_cols = [f"latent_{i}" for i in range(32)]  
mse_cols = [col for col in features.columns if col.startswith('mse_channel_')][:20]  # Top 20
attention_cols = ['attention_max', 'attention_std', 'attention_entropy']

selected_features = ['mse_total'] + mse_cols + latent_cols + attention_cols
selected_features = [col for col in selected_features if col in features.columns]

print(f"📋 Selected {len(selected_features)} features:")
print(f"   - MSE features: {len([f for f in selected_features if 'mse' in f])}")
print(f"   - Latent features: {len(latent_cols)}")
print(f"   - Attention features: {len([f for f in selected_features if 'attention' in f])}")

X = features[selected_features]
y = features["is_anomaly"]

print(f"🔍 Feature matrix shape: {X.shape}")
print(f"🔍 Target distribution: Normal={np.sum(y==0)}, Anomaly={np.sum(y==1)}")

print("🔄 Creating stratified train/test split...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"📊 Train: {len(X_train)} samples, {y_train.sum()} anomalies ({y_train.mean()*100:.2f}%)")
print(f"📊 Test: {len(X_test)} samples, {y_test.sum()} anomalies ({y_test.mean()*100:.2f}%)")

# Check if we have anomalies in test
if y_test.sum() == 0:
    print("⚠️ No anomalies in test set - using manual split...")
    # Manual approach
    anomaly_indices = np.where(y == 1)[0]
    normal_indices = np.where(y == 0)[0]
    
    # Take 20% of anomalies for test
    n_test_anomalies = max(1, len(anomaly_indices) // 5)
    test_anomaly_idx = np.random.choice(anomaly_indices, n_test_anomalies, replace=False)
    
    # Take 20% of normal samples for test
    n_test_normal = len(normal_indices) // 5
    test_normal_idx = np.random.choice(normal_indices, n_test_normal, replace=False)
    
    test_idx = np.concatenate([test_anomaly_idx, test_normal_idx])
    train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"📊 Manual split - Train: {len(X_train)} samples, {y_train.sum()} anomalies")
    print(f"📊 Manual split - Test: {len(X_test)} samples, {y_test.sum()} anomalies")

# Train Random Forest
print("🌲 Training Random Forest...")
clf = RandomForestClassifier(
    n_estimators=200,  
    max_depth=10,    
    class_weight="balanced", 
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)
print("✅ Training completed!")

# Evaluation
print("\n📊 EVALUATION:")

# Training performance
train_pred = clf.predict(X_train)
train_f1 = f1_score(y_train, train_pred)
print(f"🎯 Training F1: {train_f1:.3f}")

# Test performance
test_pred = clf.predict(X_test)
test_f1 = f1_score(y_test, test_pred)
print(f"🎯 Test F1: {test_f1:.3f}")

print(f"\n📈 DETAILED TEST RESULTS:")
print(classification_report(y_test, test_pred))

# Feature importance
print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
feature_importance = list(zip(selected_features, clf.feature_importances_))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for i, (feat, importance) in enumerate(feature_importance[:10]):
    feature_type = "MSE" if feat.startswith('mse') else "LATENT" if feat.startswith('latent') else "ATTENTION"
    print(f"{i+1:2d}. {feat:20s} ({feature_type:9s}): {importance:.4f}")

# Save model
joblib.dump(clf, "models/rf_downstream.pkl")

# Also save feature names for later use
with open("models/rf_features.txt", "w") as f:
    for feat in selected_features:
        f.write(feat + "\n")

print(f"\n💾 SAVED:")
print(f"✅ Model: models/rf_downstream.pkl")
print(f"✅ Features: models/rf_features.txt")

# Summary
print(f"\n🎉 SUMMARY:")
print(f"📊 Features used: {len(selected_features)}")
print(f"🎯 Test F1 Score: {test_f1:.3f}")
print(f"🏆 Best feature type: {feature_importance[0][0]}")

