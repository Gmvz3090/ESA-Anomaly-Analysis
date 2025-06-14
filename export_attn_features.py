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

model = AttentionGRUAutoEncoder(X.shape[1], hidden_dim=64, latent_dim=LATENT)
model.load_state_dict(torch.load(MODEL, map_location=device))
model.eval()

errors = []
latents = []
with torch.no_grad():
    for batch in torch.split(tensor, 128):
        out = model(batch)
        err = ((out - batch) ** 2).mean(dim=1)
        errors.extend(err.mean(dim=1).cpu().numpy())
        latents.extend(model.latent.weight.data.cpu().numpy())

errors = np.array(errors)
df_out = pd.DataFrame({
    "idx": np.arange(len(errors)),
    "mse_total": errors,
    "is_anomaly": labels
})

for i in range(X.shape[1]):
    df_out[f"mse_{i}"] = err[:, i].cpu().numpy()

for j in range(LATENT):
    df_out[f"z_{j}"] = [z[j].item() for z in model.latent(model.encoder(tensor)[0].mean(dim=1))]

df_out.to_csv("attn_features.csv", index=False)
print("✅ Saved to attn_features.csv")
