#!/bin/bash
echo "🧪 Setting up virtual environment..."
python -m venv .venv
source .venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🧠 Training GRU + Attention model..."
python training_gru_attn.py

echo "📤 Extracting features and MSE from test set..."
python export_attn_features.py

echo "🌲 Training downstream RandomForest classifier..."
python train_rf_on_latent.py

echo "📈 Generating anomaly report..."
python test_rf_downstream.py

echo "✅ Pipeline complete. See report.csv for results."
