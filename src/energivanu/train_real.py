# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Train Energivanu PEB model on real H100 data from York University.

Usage:
    python -m energivanu.train_real
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import build_dataloaders
from .model import EnergivanuPEB


def train():
    SEQ_LEN = 30
    PRED_HORIZON = 10
    BATCH_SIZE = 64
    NUM_GPUS = 200000
    EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, n_features = build_dataloaders(
        seq_len=SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        batch_size=BATCH_SIZE,
        num_gpus=NUM_GPUS,
    )

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
        os.path.dirname(os.path.abspath(__file__)), "..", "models", "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_model_real.pt")

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
