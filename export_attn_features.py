import torch
import pandas as pd
import numpy as np
import joblib
from attention_gru_autoencoder import AttentionGRUAutoEncoder

TEST = "3_months.test.csv"
SCALER = "models/rs_scaler.pkl"
MODEL = "models/gru_attn.pt"
WIN = 10
LATENT = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(TEST)
anom = [c for c in df.columns if c.startswith("is_anomaly_")]
labels = (df[anom].sum(axis=1) > 0).astype(int).values[WIN:]
X = df.drop(columns=["timestamp"] + anom, errors="ignore").values

sc = joblib.load(SCALER)
Xs = sc.transform(X)
seq = np.stack([Xs[i:i+WIN] for i in range(len(Xs)-WIN)])
tensor = torch.tensor(seq).float().to(device)

model = AttentionGRUAutoEncoder(X.shape[1], hidden_dim=64, latent_dim=LATENT).to(device)
model.load_state_dict(torch.load(MODEL, map_location=device))
model.eval()

errors_total = []
errors_per_channel = []
latent_vectors = []

with torch.no_grad():
    for batch in torch.split(tensor, 128):
        batch = batch.to(device)
        out = model(batch)
        err = ((out - batch) ** 2).mean(dim=1)
        errors_total.extend(err.mean(dim=1).cpu().numpy())
        errors_per_channel.extend(err.cpu().numpy())

        h, _ = model.encoder(batch)
        h_mean = h.mean(dim=1)
        z = model.latent(h_mean)
        latent_vectors.extend(z.cpu().numpy())

df_out = pd.DataFrame({
    "idx": np.arange(len(errors_total)),
    "mse_total": errors_total,
    "is_anomaly": labels
})

for i in range(X.shape[1]):
    df_out[f"mse_{i}"] = [row[i] for row in errors_per_channel]

for j in range(LATENT):
    df_out[f"z_{j}"] = [z[j] for z in latent_vectors]

df_out.to_csv("attn_features.csv", index=False)
print("✅ Saved to attn_features.csv")