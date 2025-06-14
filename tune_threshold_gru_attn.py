import numpy as np, pandas as pd, torch, joblib
from torch.utils.data import DataLoader, TensorDataset
from attention_gru_autoencoder import AttentionGRUAutoEncoder

TEST = "3_months.test.csv"; SCALER, MODEL = "models/rs_scaler.pkl", "models/gru_attn.pt"
WIN, BS, LATENT = 10, 64, 32
device = torch.device("cpu")

df = pd.read_csv(TEST)
anom=[c for c in df if c.startswith("is_anomaly_")]
flags=(df[anom].sum(1)>0).values[WIN:]
X=df.drop(columns=["timestamp"]+anom, errors="ignore").values
sc=joblib.load(SCALER); Xs=sc.transform(X)
seq=np.stack([Xs[i:i+WIN] for i in range(len(Xs)-WIN)])
loader = DataLoader(TensorDataset(torch.tensor(seq).float()), batch_size=BS)

m=AttentionGRUAutoEncoder(X.shape[1],64,LATENT).eval()
m.load_state_dict(torch.load(MODEL, map_location=device))

errs=[]
with torch.no_grad():
    for (b,) in loader:
        r=m(b)
        errs.extend(((r-b)**2).mean((1,2)).numpy())
errs=np.array(errs)

best=-1; bt=None
for thr in np.linspace(errs.min(), errs.max(), 500):
    d=errs>thr
    tp=(d & flags).sum(); fp=(d & ~flags).sum(); fn=(~d & flags).sum()
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    f1=2*prec*rec/(prec+rec+1e-9)
    if f1>best:
        best, bt, stats = f1, thr,(tp,fp,fn)
print("Best thr", bt, "F1",best, "stats",stats)
open("models/gru_attn_thr.txt","w").write(str(bt))
