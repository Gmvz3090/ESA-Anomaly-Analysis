from torch import nn
import torch, pandas as pd, numpy as np, joblib, os
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from attention_gru_autoencoder import AttentionGRUAutoEncoder

TRAIN = "3_months.train.csv"
SCALER, MODEL, THR = "models/rs_scaler.pkl", "models/gru_attn.pt", "models/gru_attn_thr.txt"
EPOCHS, BS, WIN = 50, 64, 10  # Increased epochs for early stopping
HIDDEN, LATENT = 64, 32
VAL_SPLIT = 0.2  # 20% for validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading and preprocessing data...")
df = pd.read_csv(TRAIN)
print(f"Original data shape: {df.shape}")

anom = [c for c in df if c.startswith("is_anomaly_")]
print(f"Found {len(anom)} anomaly columns")
anomaly_count = df[anom].sum(1).sum()
print(f"Total anomalies in training data: {anomaly_count}")

df_clean = df[df[anom].sum(1) == 0]
print(f"Clean data shape: {df_clean.shape}")

# Extract features
X = df_clean.drop(columns=["timestamp"] + anom, errors="ignore").values
print(f"Feature matrix shape: {X.shape}")

# Scale features
sc = RobustScaler()
Xs = sc.fit_transform(X)
joblib.dump(sc, SCALER)
print(f"Scaler saved to {SCALER}")

print(f"Creating sequences with window size {WIN}...")
seq = np.stack([Xs[i:i+WIN] for i in range(len(Xs)-WIN)])
print(f"Total sequences: {seq.shape}")

split_idx = int(len(seq) * (1 - VAL_SPLIT))
train_seq = seq[:split_idx]
val_seq = seq[split_idx:]

print(f"Training sequences: {train_seq.shape}")
print(f"Validation sequences: {val_seq.shape}")

train_loader = DataLoader(
    TensorDataset(torch.tensor(train_seq).float()), 
    batch_size=BS, 
    shuffle=False 
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(val_seq).float()), 
    batch_size=BS, 
    shuffle=False
)

print("Initializing model...")
model = AttentionGRUAutoEncoder(X.shape[1], HIDDEN, LATENT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', patience=3, factor=0.5
)
loss_fn = nn.SmoothL1Loss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("Starting training...")
best_val_loss = float('inf')
patience = 7
patience_counter = 0
train_losses = []
val_losses = []

os.makedirs("models", exist_ok=True)

for e in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0
    for (b,) in train_loader:
        b = b.to(device)
        opt.zero_grad()
        out = model(b)
        l = loss_fn(out, b)
        l.backward()
        opt.step()
        train_loss += l.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (b,) in val_loader:
            b = b.to(device)
            out = model(b)
            l = loss_fn(out, b)
            val_loss += l.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {e+1:3d}/{EPOCHS}  Train: {train_loss:.6f}  Val: {val_loss:.6f}")
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL.replace('.pt', '_best.pt'))
        print(f"  → New best validation loss: {val_loss:.6f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {e+1} (patience={patience})")
            break

# Load best model
print("Loading best model...")
model.load_state_dict(torch.load(MODEL.replace('.pt', '_best.pt')))

print("Calculating threshold on training data...")
model.eval()
train_errors = []

with torch.no_grad():
    for (b,) in train_loader:
        b = b.to(device)
        out = model(b)
        mse = ((out - b)**2).mean((1,2))
        train_errors.extend(mse.cpu().numpy())

train_errors = np.array(train_errors)

percentiles = [90, 95, 97, 99, 99.5]
thresholds = {}
for p in percentiles:
    thresholds[p] = np.percentile(train_errors, p)

print(f"\nMSE:")
print(f"  Mean: {np.mean(train_errors):.6f}")
print(f"  Std:  {np.std(train_errors):.6f}")
print(f"  Min:  {np.min(train_errors):.6f}")
print(f"  Max:  {np.max(train_errors):.6f}")

for p, thr in thresholds.items():
    anomaly_rate = np.mean(train_errors > thr) * 100

selected_threshold = thresholds[95]
print(f"\nSelected threshold (95th percentile): {selected_threshold:.6f}")

val_errors = []
with torch.no_grad():
    for (b,) in val_loader:
        b = b.to(device)
        out = model(b)
        mse = ((out - b)**2).mean((1,2))
        val_errors.extend(mse.cpu().numpy())

val_errors = np.array(val_errors)
val_anomaly_rate = np.mean(val_errors > selected_threshold) * 100
print(f"Validation anomaly rate with selected threshold: {val_anomaly_rate:.2f}%")

torch.save(model.state_dict(), MODEL)
with open(THR, "w") as f:
    f.write(str(selected_threshold))

np.save("models/train_losses.npy", train_losses)
np.save("models/val_losses.npy", val_losses)
np.save("models/train_errors.npy", train_errors)
np.save("models/val_errors.npy", val_errors)

print(f"\nFiles saved:")
print(f"  Model: {MODEL}")
print(f"  Best model: {MODEL.replace('.pt', '_best.pt')}")
print(f"  Threshold: {THR}")
print(f"  Scaler: {SCALER}")
print(f"  Training history: models/train_losses.npy, models/val_losses.npy")
print(f"  Error distributions: models/train_errors.npy, models/val_errors.npy")

print(f"\n✅ Training completed!")
print(f"📊 Best validation loss: {best_val_loss:.6f}")
print(f"🎯 Final threshold: {selected_threshold:.6f}")
