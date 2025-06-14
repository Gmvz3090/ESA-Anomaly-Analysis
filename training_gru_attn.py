from torch import nn
import torch, pandas as pd, numpy as np, joblib, os
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from attention_gru_autoencoder import AttentionGRUAutoEncoder

TRAIN = "3_months.train.csv"
SCALER, MODEL, THR = "models/rs_scaler.pkl", "models/gru_attn.pt", "models/gru_attn_thr.txt"
EPOCHS, BS, WIN = 30, 64, 10
HIDDEN, LATENT = 64, 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(TRAIN)
anom = [c for c in df if c.startswith("is_anomaly_")]
df = df[df[anom].sum(1) == 0]
X = df.drop(columns=["timestamp"] + anom, errors="ignore").values

sc = RobustScaler()
Xs = sc.fit_transform(X)
joblib.dump(sc, SCALER)

seq = np.stack([Xs[i:i+WIN] for i in range(len(Xs)-WIN)])
loader = DataLoader(TensorDataset(torch.tensor(seq).float()), batch_size=BS, shuffle=True)

model = AttentionGRUAutoEncoder(X.shape[1], HIDDEN, LATENT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.SmoothL1Loss()

for e in range(EPOCHS):
    model.train()
    tot = 0
    for (b,) in loader:
        b = b.to(device)
        opt.zero_grad()
        out = model(b)
        l = loss_fn(out, b)
        l.backward()
        opt.step()
        tot += l.item()
    print(f"Epoch {e+1}/{EPOCHS}  Loss {tot/len(loader):.4f}")

model.eval()
errs = []
for (b,) in loader:
    b = b.to(device)
    with torch.no_grad():
        errs.extend(((model(b) - b)**2).mean((1,2)).cpu().numpy())

thr = np.percentile(errs, 99)
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL)
with open(THR, "w") as f:
    f.write(str(thr))
print(f"Model saved to {MODEL}")
print(f"Threshold saved to {THR} (value: {thr:.6f})")
print(f"Scaler saved to {SCALER}")
