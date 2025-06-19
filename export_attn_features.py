import torch
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset
from attention_gru_autoencoder import AttentionGRUAutoEncoder


TEST = "3_months.test.csv"
SCALER = "models/rs_scaler.pkl"
MODEL = "models/gru_attn_best.pt"
THRESHOLD_FILE = "models/gru_attn_thr.txt"
WIN = 10
HIDDEN, LATENT = 64, 32
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("Loading test data...")
df = pd.read_csv(TEST)
print(f"Test data shape: {df.shape}")


anom = [c for c in df.columns if c.startswith("is_anomaly_")]
print(f"Found {len(anom)} anomaly columns")


all_labels = (df[anom].sum(axis=1) > 0).astype(int).values
print(f"Total anomalies in test data: {all_labels.sum()} ({all_labels.mean()*100:.2f}%)")


X_all = df.drop(columns=["timestamp"] + anom, errors="ignore").values
print(f"Full feature matrix shape: {X_all.shape}")


print("Creating sequences...")
sequences = np.stack([X_all[i:i+WIN] for i in range(len(X_all)-WIN)])
labels = all_labels[WIN:]  

print(f"Sequences shape: {sequences.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Anomalies in sequence labels: {labels.sum()} ({labels.mean()*100:.2f}%)")


print("\n🔧 APPLYING SCALING TO ALL DATA...")
sc = joblib.load(SCALER)


seq_reshaped = sequences.reshape(-1, sequences.shape[-1])
seq_scaled_reshaped = sc.transform(seq_reshaped)
seq_scaled = seq_scaled_reshaped.reshape(sequences.shape)

print(f"Scaled sequence range: [{seq_scaled.min():.3f}, {seq_scaled.max():.3f}]")
print(f"Scaled sequence mean: {seq_scaled.mean():.3f}, std: {seq_scaled.std():.3f}")


print("Loading threshold...")
with open(THRESHOLD_FILE, 'r') as f:
    original_threshold = float(f.read().strip())
print(f"Original threshold: {original_threshold:.6f}")


print("Loading model...")
model = AttentionGRUAutoEncoder(X_all.shape[1], hidden_dim=HIDDEN, latent_dim=LATENT).to(device)
model.load_state_dict(torch.load(MODEL, map_location=device))
model.eval()


print("\n🧪 TESTING MODEL SCALE...")
test_sample = torch.tensor(seq_scaled[:100]).float().to(device)
with torch.no_grad():
    test_output = model(test_sample)
    test_mse = ((test_output - test_sample) ** 2).mean(dim=(1,2))
    print(f"Sample MSE range: [{test_mse.min():.6f}, {test_mse.max():.6f}]")
    print(f"Sample MSE mean: {test_mse.mean():.6f}")



mse_scale_factor = test_mse.mean().item() / original_threshold
print(f"MSE scale factor: {mse_scale_factor:.2f}")

if mse_scale_factor > 10 or mse_scale_factor < 0.1:
    print("🚨 Large scale difference detected - adjusting threshold...")
    
    adjusted_threshold = np.percentile(test_mse.cpu().numpy(), 95)
    print(f"Adjusted threshold (95th percentile): {adjusted_threshold:.6f}")
    threshold = adjusted_threshold
else:
    threshold = original_threshold
    print(f"Using original threshold: {threshold:.6f}")


test_loader = DataLoader(
    TensorDataset(torch.tensor(seq_scaled).float()), 
    batch_size=BATCH_SIZE, 
    shuffle=False
)


print("\nExtracting features...")
errors_total = []
errors_per_channel = []
latent_vectors = []
attention_weights = []

with torch.no_grad():
    for batch_idx, (batch,) in enumerate(test_loader):
        batch = batch.to(device)
        
        
        output = model(batch)
        
        
        mse_per_sample_channel = ((output - batch) ** 2).mean(dim=1)  
        mse_per_sample = mse_per_sample_channel.mean(dim=1)  
        
        errors_total.extend(mse_per_sample.cpu().numpy())
        errors_per_channel.extend(mse_per_sample_channel.cpu().numpy())
        
        
        latent, attn_weights = model.get_latent_representation(batch)
        latent_vectors.extend(latent)
        attention_weights.extend(attn_weights)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} sequences...")


errors_total = np.array(errors_total)
attention_weights = np.array(attention_weights)

print(f"Extracted features for {len(errors_total)} sequences")


print("Creating feature dataframe...")


base_features = pd.DataFrame({
    "idx": np.arange(len(errors_total)),
    "mse_total": errors_total,
    "is_anomaly": labels,
    "is_high_error": (errors_total > threshold).astype(int)
})


mse_channel_df = pd.DataFrame(
    errors_per_channel,
    columns=[f"mse_channel_{i}" for i in range(X_all.shape[1])]
)


latent_df = pd.DataFrame(
    latent_vectors,
    columns=[f"latent_{i}" for i in range(LATENT)]
)


attention_stats_df = pd.DataFrame({
    'attention_max': attention_weights.max(axis=1),
    'attention_min': attention_weights.min(axis=1),
    'attention_std': attention_weights.std(axis=1),
    'attention_entropy': -np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=1)
})

attention_timestep_df = pd.DataFrame(
    attention_weights,
    columns=[f'attention_t{t}' for t in range(WIN)]
)


df_out = pd.concat([
    base_features,
    mse_channel_df,
    latent_df,
    attention_stats_df,
    attention_timestep_df
], axis=1)


print(f"\n📊 FEATURE EXTRACTION SUMMARY:")
print(f"Total sequences: {len(df_out)}")
print(f"True anomalies: {df_out['is_anomaly'].sum()} ({df_out['is_anomaly'].mean()*100:.2f}%)")
print(f"High error samples (MSE > threshold): {df_out['is_high_error'].sum()} ({df_out['is_high_error'].mean()*100:.2f}%)")




output_file = "attn_features.csv"
df_out.to_csv(output_file, index=False)
print(f"\n✅ Features saved to: {output_file}")
print(f"📊 Feature matrix shape: {df_out.shape}")
print(f"MSE features: {X_all.shape[1] + 1}")
print(f"Latent features: {LATENT}")
print(f"Attention features: {WIN + 4}")
print(f"Total features: {df_out.shape[1] - 3}") 
