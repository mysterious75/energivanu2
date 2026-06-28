"""
Train Energivanu PEB model on synthetic data for out-of-box demo.

Usage:
    python -m energivanu.train_demo
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .model import EnergivanuPEB


class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=30, pred_horizon=10):
        # Generate synthetic sequence
        t = np.linspace(0, 50 * np.pi, num_samples + seq_len + pred_horizon)
        base_power = np.sin(t) * 50 + 200 # MW
        noise = np.random.normal(0, 2, len(t))
        power = base_power + noise

        X, Yp, Ys = [], [], []
        for i in range(num_samples):
            # Create 15 features
            features = np.zeros((seq_len, 15))
            features[:, 0] = power[i:i+seq_len] # facility_mw
            features[:, 1] = np.gradient(features[:, 0]) # power_roc
            features[:, 2] = np.gradient(features[:, 1]) # power_roc2
            features[:, 3] = np.convolve(features[:, 0], np.ones(5)/5, mode='same') # roll_mean
            features[:, 4] = 2.0 # roll_std approx
            features[:, 5] = features[:, 0] / 700.0 # gpu_avg_power
            features[:, 6] = features[:, 0] / 700.0 * 1.1 # gpu_max_power
            features[:, 7] = 0.5 # temp
            features[:, 8] = 0.6 # temp max
            features[:, 9] = 0.8 # util
            features[:, 10] = 0.8 # mem util
            features[:, 11] = 0.4 # cpu util
            features[:, 12] = np.sin(t[i:i+seq_len]) # hour sin
            features[:, 13] = np.cos(t[i:i+seq_len]) # hour cos
            features[:, 14] = 0.0 # is_allreduce

            target_p = power[i+seq_len:i+seq_len+pred_horizon]

            # signal
            change = target_p[0] - features[-1, 0]
            if change > 0.5:
                signal = 1
            elif change < -0.5:
                signal = 2
            else:
                signal = 0

            X.append(features)
            Yp.append(target_p)
            Ys.append(signal)

        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.Yp = torch.tensor(np.array(Yp), dtype=torch.float32)
        self.Ys = torch.tensor(np.array(Ys), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Yp[idx], self.Ys[idx]


def train():
    SEQ_LEN = 30
    PRED_HORIZON = 10
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = SyntheticDataset(num_samples=2000, seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON)
    val_ds = SyntheticDataset(num_samples=500, seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    n_features = 15

    model = EnergivanuPEB(
        n_features=n_features,
        seq_len=SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        tcn_channels=[32, 64, 128],
        tcn_kernels=[5, 3, 3],
        attention_heads=8,
        attention_dim=128,
        hidden_dims=[256, 128],
        n_signal_classes=3,
        dropout=0.1,
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    power_loss_fn = nn.MSELoss()
    signal_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_model_demo.pt")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_batches = 0
        t0 = time.time()

        for X, Yp, Ys in train_loader:
            X, Yp, Ys = X.to(device), Yp.to(device), Ys.to(device)

            power_pred, signal_logits = model(X)
            loss_power = power_loss_fn(power_pred, Yp)
            loss_signal = signal_loss_fn(signal_logits, Ys)
            loss = loss_power + 0.5 * loss_signal

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train = train_loss / train_batches

        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X, Yp, Ys in val_loader:
                X, Yp, Ys = X.to(device), Yp.to(device), Ys.to(device)
                power_pred, signal_logits = model(X)
                loss_power = power_loss_fn(power_pred, Yp)
                loss_signal = signal_loss_fn(signal_logits, Ys)
                loss = loss_power + 0.5 * loss_signal
                val_loss += loss.item()
                val_batches += 1
                all_preds.append(power_pred.cpu().numpy())
                all_targets.append(Yp.cpu().numpy())

        avg_val = val_loss / val_batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        mape = np.mean(np.abs((all_preds - all_targets) / (all_targets + 1e-8))) * 100
        dt = time.time() - t0

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"MAPE: {mape:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f} | {dt:.1f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val,
                "val_mape": mape,
                "config": {
                    "n_features": n_features,
                    "seq_len": SEQ_LEN,
                    "pred_horizon": PRED_HORIZON,
                    "tcn_channels": [32, 64, 128],
                    "tcn_kernels": [5, 3, 3],
                    "attention_heads": 8,
                    "attention_dim": 128,
                    "hidden_dims": [256, 128],
                    "n_signal_classes": 3,
                    "dropout": 0.1,
                },
            }, best_path)
            print(f"  -> Saved best model (val_loss={avg_val:.4f})")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved to: {best_path}")


if __name__ == "__main__":
    train()
