import torch, pandas as pd, numpy as np, joblib
from torch.utils.data import DataLoader, TensorDataset
from attention_gru_autoencoder import AttentionGRUAutoEncoder

TEST = "3_months.test.csv"
SCALER, MODEL, THR = "models/rs_scaler.pkl", "models/gru_attn.pt", "models/gru_attn_thr.txt"
WIN, BS, LATENT = 10, 64, 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(TEST)
anom = [c for c in df if c.startswith("is_anomaly_")]
flags = (df[anom].sum(1)>0).values[WIN:]
X = df.drop(columns=["timestamp"]+anom, errors="ignore").values

sc = joblib.load(SCALER); Xs=sc.transform(X)
seq=np.stack([Xs[i:i+WIN] for i in range(len(Xs)-WIN)])
loader = DataLoader(TensorDataset(torch.tensor(seq).float()), batch_size=BS)

model=AttentionGRUAutoEncoder(X.shape[1],64,LATENT).to(device)
model.load_state_dict(torch.load(MODEL,map_location=device))
model.eval()

with open(THR) as f: thr=float(f.read())
errs=[]
for (b,) in loader:
    b=b.to(device)
    with torch.no_grad():
        errs.extend(((model(b)-b)**2).mean((1,2)).cpu().numpy())

errs=np.array(errs)
d=errs>thr
tp=(d & flags).sum(); fp=(d & ~flags).sum(); fn=(~d & flags).sum()
print(f"Threshold {thr:.3f}  TP={tp}  FP={fp}  FN={fn}")
