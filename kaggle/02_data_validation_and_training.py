"""
Data Validation and Quick Training — Kaggle Notebook
=====================================================
Load collected GPU telemetry data, validate quality (no NaN, reasonable
ranges), run a quick training loop, and save the model + metrics.

This notebook is designed to run on Kaggle with GPU runtime.

Requirements:
  - Kaggle notebook with GPU runtime (T4, P100, or V100)
  - Upload collected telemetry CSV or use the Alibaba trace data

Output files:
  - validated_features.csv   — cleaned feature data
  - quick_model.pt           — trained model checkpoint
  - training_metrics.json    — loss curves and final metrics
  - validation_report.json   — data quality report
"""

# %% [markdown]
# # Energivanu — Data Validation & Quick Training
#
# Validate collected telemetry data quality and run a quick training
# iteration to verify the pipeline end-to-end.

# %% Cell 1: Install dependencies
# !pip install torch pandas numpy matplotlib scikit-learn 2>/dev/null

# %% Cell 2: Imports
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")

# %% Cell 3: Configuration
# ---- Feature names (must match data.py exactly) ----
FEATURE_NAMES = [
    "facility_mw",
    "power_roc",
    "power_roc2",
    "power_roll_mean",
    "power_roll_std",
    "gpu_avg_power_norm",
    "gpu_max_power_norm",
    "gpu_avg_temp_norm",
    "gpu_max_temp_norm",
    "gpu_avg_util_norm",
    "gpu_avg_mem_util_norm",
    "cpu_util_est_norm",
    "hour_sin",
    "hour_cos",
    "is_allreduce",
]

N_FEATURES = len(FEATURE_NAMES)
SEQ_LEN = 30
PRED_HORIZON = 10
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
STRIDE = 50

# ---- Reasonable value ranges for validation ----
FEATURE_RANGES = {
    "facility_mw":            (0.0, 1000.0),
    "power_roc":              (-100.0, 100.0),
    "power_roc2":             (-50.0, 50.0),
    "power_roll_mean":        (0.0, 1000.0),
    "power_roll_std":         (0.0, 200.0),
    "gpu_avg_power_norm":     (0.0, 1.5),
    "gpu_max_power_norm":     (0.0, 1.5),
    "gpu_avg_temp_norm":      (0.0, 1.0),
    "gpu_max_temp_norm":      (0.0, 1.0),
    "gpu_avg_util_norm":      (0.0, 1.0),
    "gpu_avg_mem_util_norm":  (0.0, 1.0),
    "cpu_util_est_norm":      (0.0, 1.0),
    "hour_sin":               (-1.0, 1.0),
    "hour_cos":               (-1.0, 1.0),
    "is_allreduce":           (0.0, 1.0),
}

print(f"Features: {N_FEATURES}")
print(f"Seq len:  {SEQ_LEN}, Pred horizon: {PRED_HORIZON}")


# %% Cell 4: Data Loading
def find_data_file() -> str:
    """Locate the telemetry/training features CSV."""
    candidates = [
        "data/collections/run_latest/training_features.csv",
        "data/collections/training_features.csv",
        "data/alibaba/training_features.csv",
        "training_features.csv",
        "gpu_telemetry.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Search for any CSV with feature columns
    for root, dirs, files in os.walk("data"):
        for f in sorted(files, reverse=True):
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                try:
                    head = pd.read_csv(path, nrows=2)
                    if all(col in head.columns for col in FEATURE_NAMES[:5]):
                        return path
                except Exception:
                    continue

    return ""


data_file = find_data_file()
if data_file:
    print(f"Found data: {data_file}")
    raw_df = pd.read_csv(data_file)
    print(f"Shape: {raw_df.shape}")
else:
    print("⚠️  No data file found. Generating synthetic data for demo.")
    print("   In production, upload your telemetry CSV to the notebook.")
    # Generate synthetic data for validation demo
    np.random.seed(42)
    n_rows = 5000
    ts = pd.date_range("2024-01-01", periods=n_rows, freq="1s")
    raw_df = pd.DataFrame({
        "facility_mw": 150 + 30 * np.sin(np.linspace(0, 4 * np.pi, n_rows)) + np.random.normal(0, 3, n_rows),
        "power_roc": np.random.normal(0, 2, n_rows),
        "power_roc2": np.random.normal(0, 1, n_rows),
        "power_roll_mean": 150 + 30 * np.sin(np.linspace(0, 4 * np.pi, n_rows)),
        "power_roll_std": np.abs(np.random.normal(5, 2, n_rows)),
        "gpu_avg_power_norm": np.clip(0.7 + np.random.normal(0, 0.1, n_rows), 0, 1.2),
        "gpu_max_power_norm": np.clip(0.8 + np.random.normal(0, 0.1, n_rows), 0, 1.2),
        "gpu_avg_temp_norm": np.clip(0.75 + np.random.normal(0, 0.05, n_rows), 0, 1),
        "gpu_max_temp_norm": np.clip(0.80 + np.random.normal(0, 0.05, n_rows), 0, 1),
        "gpu_avg_util_norm": np.clip(0.85 + np.random.normal(0, 0.1, n_rows), 0, 1),
        "gpu_avg_mem_util_norm": np.clip(0.6 + np.random.normal(0, 0.15, n_rows), 0, 1),
        "cpu_util_est_norm": np.full(n_rows, 0.4),
        "hour_sin": np.sin(2 * np.pi * np.arange(n_rows) % 86400 / 86400),
        "hour_cos": np.cos(2 * np.pi * np.arange(n_rows) % 86400 / 86400),
        "is_allreduce": (np.random.random(n_rows) > 0.9).astype(float),
        "timestamp": ts,
    })
    print(f"Generated synthetic data: {raw_df.shape}")


# %% Cell 5: Data Validation
print("=" * 60)
print("  Data Quality Validation")
print("=" * 60)

validation_report = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "data_file": data_file or "(synthetic)",
    "total_rows": len(raw_df),
    "checks": {},
    "passed": True,
}

# Check 1: Required columns present
missing_cols = [c for c in FEATURE_NAMES if c not in raw_df.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    validation_report["checks"]["missing_columns"] = {
        "status": "FAIL",
        "missing": missing_cols,
    }
    validation_report["passed"] = False
else:
    print("✅ All 15 feature columns present")
    validation_report["checks"]["missing_columns"] = {"status": "PASS"}

# Check 2: No NaN values
nan_counts = raw_df[FEATURE_NAMES].isna().sum()
total_nans = nan_counts.sum()
if total_nans > 0:
    print(f"⚠️  Found {total_nans} NaN values — will fill")
    for col in FEATURE_NAMES:
        if nan_counts[col] > 0:
            print(f"   {col}: {nan_counts[col]} NaNs")
    validation_report["checks"]["nan_values"] = {
        "status": "WARN",
        "total_nans": int(total_nans),
        "by_column": {k: int(v) for k, v in nan_counts.items() if v > 0},
    }
    # Fill NaN
    raw_df[FEATURE_NAMES] = raw_df[FEATURE_NAMES].fillna(0.0)
else:
    print("✅ No NaN values")
    validation_report["checks"]["nan_values"] = {"status": "PASS", "total_nans": 0}

# Check 3: No infinite values
inf_counts = np.isinf(raw_df[FEATURE_NAMES].values).sum(axis=0)
total_inf = inf_counts.sum()
if total_inf > 0:
    print(f"⚠️  Found {total_inf} infinite values — will clip")
    validation_report["checks"]["inf_values"] = {
        "status": "WARN",
        "total_inf": int(total_inf),
    }
    raw_df[FEATURE_NAMES] = raw_df[FEATURE_NAMES].replace([np.inf, -np.inf], 0.0)
else:
    print("✅ No infinite values")
    validation_report["checks"]["inf_values"] = {"status": "PASS", "total_inf": 0}

# Check 4: Value ranges
range_violations = {}
for col in FEATURE_NAMES:
    if col in raw_df.columns:
        lo, hi = FEATURE_RANGES[col]
        below = (raw_df[col] < lo).sum()
        above = (raw_df[col] > hi).sum()
        if below > 0 or above > 0:
            range_violations[col] = {"below": int(below), "above": int(above)}

if range_violations:
    print(f"⚠️  Range violations in {len(range_violations)} features:")
    for col, counts in range_violations.items():
        lo, hi = FEATURE_RANGES[col]
        print(f"   {col}: {counts['below']} below {lo}, {counts['above']} above {hi}")
    validation_report["checks"]["value_ranges"] = {
        "status": "WARN",
        "violations": range_violations,
    }
else:
    print("✅ All features within expected ranges")
    validation_report["checks"]["value_ranges"] = {"status": "PASS"}

# Check 5: Minimum data length
min_rows_needed = SEQ_LEN + PRED_HORIZON + STRIDE
if len(raw_df) < min_rows_needed:
    print(f"❌ Insufficient data: {len(raw_df)} rows (need ≥{min_rows_needed})")
    validation_report["checks"]["data_length"] = {
        "status": "FAIL",
        "rows": len(raw_df),
        "min_needed": min_rows_needed,
    }
    validation_report["passed"] = False
else:
    print(f"✅ Sufficient data: {len(raw_df)} rows")
    validation_report["checks"]["data_length"] = {
        "status": "PASS",
        "rows": len(raw_df),
    }

# Check 6: Statistical summary
print("\nFeature statistics:")
stats = raw_df[FEATURE_NAMES].describe().T[["mean", "std", "min", "max"]]
print(stats.to_string())

validation_report["feature_stats"] = {
    col: {
        "mean": float(stats.loc[col, "mean"]),
        "std": float(stats.loc[col, "std"]),
        "min": float(stats.loc[col, "min"]),
        "max": float(stats.loc[col, "max"]),
    }
    for col in FEATURE_NAMES if col in stats.index
}

# Save validated data
os.makedirs("data/validated", exist_ok=True)
raw_df[FEATURE_NAMES].to_csv("data/validated/validated_features.csv", index=False)
print(f"\n✅ Validated features saved to data/validated/validated_features.csv")

# Save validation report
with open("data/validated/validation_report.json", "w") as f:
    json.dump(validation_report, f, indent=2, default=str)
print("✅ Validation report saved to data/validated/validation_report.json")


# %% Cell 6: Sequence Creation
def create_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    pred_horizon: int = PRED_HORIZON,
    stride: int = STRIDE,
):
    """Create training sequences from feature DataFrame."""
    features = df[FEATURE_NAMES].values.astype(np.float32)

    # Target: facility_mw (index 0) for the prediction horizon
    facility_mw = features[:, 0]

    # Signal labels: 0=hold, 1=power increasing, 2=power decreasing
    power_change = np.diff(facility_mw, prepend=facility_mw[0])
    signals = np.zeros(len(facility_mw), dtype=np.int64)
    signals[power_change > 0.5] = 1
    signals[power_change < -0.5] = 2

    X, Y_power, Y_signal = [], [], []
    for i in range(0, len(features) - seq_len - pred_horizon, stride):
        X.append(features[i:i + seq_len])
        Y_power.append(facility_mw[i + seq_len:i + seq_len + pred_horizon])
        Y_signal.append(signals[i + seq_len])

    return (
        np.array(X, dtype=np.float32),
        np.array(Y_power, dtype=np.float32),
        np.array(Y_signal, dtype=np.int64),
    )


X, Y_power, Y_signal = create_sequences(raw_df)
print(f"Sequences: {X.shape[0]}")
print(f"X shape:   {X.shape}")  # (N, seq_len, 15)
print(f"Y_power:   {Y_power.shape}")  # (N, pred_horizon)
print(f"Y_signal:  {Y_signal.shape}")  # (N,)
print(f"Signal distribution: hold={np.sum(Y_signal==0)}, "
      f"discharge={np.sum(Y_signal==1)}, charge={np.sum(Y_signal==2)}")


# %% Cell 7: Dataset and DataLoader
class TelemetryDataset(Dataset):
    """PyTorch Dataset for telemetry sequences."""

    def __init__(self, X, Y_power, Y_signal):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_power = torch.tensor(Y_power, dtype=torch.float32)
        self.Y_signal = torch.tensor(Y_signal, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_power[idx], self.Y_signal[idx]


# Train/val split
n = len(X)
indices = np.random.permutation(n)
split = int(n * 0.85)
train_idx, val_idx = indices[:split], indices[split:]

train_ds = TelemetryDataset(X[train_idx], Y_power[train_idx], Y_signal[train_idx])
val_ds = TelemetryDataset(X[val_idx], Y_power[val_idx], Y_signal[val_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")


# %% Cell 8: Simple TCN Model for Quick Training
class SimpleTCNBlock(nn.Module):
    """Single temporal convolution block."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self._padding = padding

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]  # causal trim
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        res = self.downsample(x)
        return self.relu(out + res)


class QuickPEBModel(nn.Module):
    """
    Lightweight TCN model for quick validation training.

    Input:  (batch, seq_len, 15)
    Output: power_pred (batch, pred_horizon), signal_logits (batch, 3)
    """

    def __init__(self, n_features=15, seq_len=30, pred_horizon=10,
                 channels=(32, 64), kernels=(5, 3), dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        layers = []
        in_ch = n_features
        for out_ch, k in zip(channels, kernels):
            layers.append(SimpleTCNBlock(in_ch, out_ch, k, dilation=1, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.power_head = nn.Linear(channels[-1], pred_horizon)
        self.signal_head = nn.Linear(channels[-1], 3)

    def forward(self, x):
        # x: (B, T, F) → (B, F, T)
        h = x.transpose(1, 2)
        h = self.tcn(h)
        h = self.pool(h).squeeze(-1)  # (B, C)
        return self.power_head(h), self.signal_head(h)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuickPEBModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} parameters on {device}")


# %% Cell 9: Training Loop
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
power_loss_fn = nn.MSELoss()
signal_loss_fn = nn.CrossEntropyLoss()

history = {"train_loss": [], "val_loss": [], "val_power_mae": []}

print(f"\nTraining for {EPOCHS} epochs...")
print("-" * 60)

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_losses = []
    for xb, yb_power, yb_signal in train_loader:
        xb = xb.to(device)
        yb_power = yb_power.to(device)
        yb_signal = yb_signal.to(device)

        pred_power, pred_signal = model(xb)
        loss = power_loss_fn(pred_power, yb_power) + 0.5 * signal_loss_fn(pred_signal, yb_signal)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    scheduler.step()

    # Validate
    model.eval()
    val_losses, val_maes = [], []
    with torch.no_grad():
        for xb, yb_power, yb_signal in val_loader:
            xb = xb.to(device)
            yb_power = yb_power.to(device)
            yb_signal = yb_signal.to(device)

            pred_power, pred_signal = model(xb)
            loss = power_loss_fn(pred_power, yb_power) + 0.5 * signal_loss_fn(pred_signal, yb_signal)
            val_losses.append(loss.item())
            val_maes.append(torch.mean(torch.abs(pred_power - yb_power)).item())

    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    avg_mae = np.mean(val_maes)
    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    history["val_power_mae"].append(avg_mae)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"MAE: {avg_mae:.4f} MW")

print("-" * 60)
print(f"Final val loss: {history['val_loss'][-1]:.4f}")
print(f"Final val MAE:  {history['val_power_mae'][-1]:.4f} MW")


# %% Cell 10: Save Model and Metrics
os.makedirs("models", exist_ok=True)

# Save model
torch.save({
    "model_state_dict": model.state_dict(),
    "n_features": N_FEATURES,
    "seq_len": SEQ_LEN,
    "pred_horizon": PRED_HORIZON,
    "channels": (32, 64),
    "kernels": (5, 3),
    "final_val_loss": history["val_loss"][-1],
    "final_val_mae": history["val_power_mae"][-1],
}, "models/quick_model.pt")
print("✅ Model saved to models/quick_model.pt")

# Save training metrics
metrics = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "n_sequences": len(X),
    "train_sequences": len(train_ds),
    "val_sequences": len(val_ds),
    "final_train_loss": history["train_loss"][-1],
    "final_val_loss": history["val_loss"][-1],
    "final_val_mae_mw": history["val_power_mae"][-1],
    "history": history,
}
with open("models/training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("✅ Metrics saved to models/training_metrics.json")


# %% Cell 11: Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss curves
axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"], label="Val")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE curve
axes[1].plot(history["val_power_mae"], color="orange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE (MW)")
axes[1].set_title("Validation Power MAE")
axes[1].grid(True, alpha=0.3)

# Feature distributions
feature_means = raw_df[FEATURE_NAMES].mean()
axes[2].barh(FEATURE_NAMES, feature_means.values)
axes[2].set_xlabel("Mean Value")
axes[2].set_title("Feature Distributions")
axes[2].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("data/validated/validation_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plots saved to data/validated/validation_plots.png")


# %% Cell 12: Summary
print()
print("=" * 60)
print("  Energivanu Data Validation & Quick Training Summary")
print("=" * 60)
print(f"  Data rows:       {len(raw_df)}")
print(f"  Sequences:       {len(X)}")
print(f"  Features:        {N_FEATURES}")
print(f"  Validation:      {'✅ PASSED' if validation_report['passed'] else '❌ ISSUES'}")
print(f"  Final val loss:  {history['val_loss'][-1]:.4f}")
print(f"  Final val MAE:   {history['val_power_mae'][-1]:.4f} MW")
print(f"  Model:           models/quick_model.pt")
print(f"  Metrics:         models/training_metrics.json")
print(f"  Report:          data/validated/validation_report.json")
print("=" * 60)
