# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Real H100 Data Processor — Process York University dataset for training.
=====================================================
- Loads real 8-GPU H100 node data (20ms resolution)
- Extracts power/temperature/utilization features
- Scales to facility-level (MW) for training
- Creates sequences for TCN model
"""

import glob
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


def load_node_data(filepath: str) -> pd.DataFrame:
    """Load a single H100 node CSV file and compute derived features."""
    df = pd.read_csv(filepath)

    gpu_power_cols = [f"gpu{i}_power_W" for i in range(8)]
    gpu_temp_cols = [f"gpu{i}_temp_C" for i in range(8)]
    gpu_util_cols = [f"gpu{i}_utilization_percent" for i in range(8)]
    gpu_mem_cols = [f"gpu{i}_mem_utilization" for i in range(8)]

    df["node_power_W"] = df[gpu_power_cols].sum(axis=1)
    df["gpu_avg_power"] = df[gpu_power_cols].mean(axis=1)
    df["gpu_max_power"] = df[gpu_power_cols].max(axis=1)
    df["gpu_std_power"] = df[gpu_power_cols].std(axis=1)
    df["gpu_avg_temp"] = df[gpu_temp_cols].mean(axis=1)
    df["gpu_max_temp"] = df[gpu_temp_cols].max(axis=1)
    df["gpu_avg_util"] = df[gpu_util_cols].mean(axis=1)
    df["gpu_avg_mem_util"] = df[gpu_mem_cols].mean(axis=1)
    df["cpu_power_estimated_W"] = df["cpu_utilization_percent"] * 2.5
    df["system_power_W"] = df["node_power_W"] + df["cpu_power_estimated_W"]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.minute / 60)

    df["power_roc"] = df["node_power_W"].diff().fillna(0)
    df["power_roc2"] = df["power_roc"].diff().fillna(0)

    window = 250
    df["power_roll_mean"] = df["node_power_W"].rolling(window, min_periods=1).mean()
    df["power_roll_std"] = df["node_power_W"].rolling(window, min_periods=1).std().fillna(0)

    df["is_allreduce"] = (
        (df["gpu_avg_util"] > 80) & (df["gpu_avg_mem_util"] < 30)
    ).astype(float)

    return df


def scale_to_facility(node_power_W: np.ndarray, num_gpus: int = 200000,
                      gpus_per_node: int = 8) -> np.ndarray:
    """Scale single-node power (W) to facility-level power (MW)."""
    num_nodes = num_gpus / gpus_per_node
    return node_power_W * num_nodes / 1e6


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = 30,
    pred_horizon: int = 10,
    num_gpus: int = 200000,
    gpus_per_node: int = 8,
    stride: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training sequences from real H100 node data.

    Returns:
        X: (N, seq_len, 15) feature sequences
        Y_power: (N, pred_horizon) power targets in MW
        Y_signal: (N,) signal labels (0=hold, 1=discharge, 2=charge)
    """
    node_power = df["node_power_W"].values
    facility_mw = scale_to_facility(node_power, num_gpus, gpus_per_node)

    features = np.column_stack([
        facility_mw,
        df["power_roc"].values * num_gpus / gpus_per_node / 1e6,
        df["power_roc2"].values * num_gpus / gpus_per_node / 1e6,
        scale_to_facility(df["power_roll_mean"].values, num_gpus, gpus_per_node),
        scale_to_facility(df["power_roll_std"].values, num_gpus, gpus_per_node),
        df["gpu_avg_power"].values / 700.0,
        df["gpu_max_power"].values / 700.0,
        df["gpu_avg_temp"].values / 100.0,
        df["gpu_max_temp"].values / 100.0,
        df["gpu_avg_util"].values / 100.0,
        df["gpu_avg_mem_util"].values / 100.0,
        df["cpu_utilization_percent"].values / 100.0,
        df["hour_sin"].values,
        df["hour_cos"].values,
        df["is_allreduce"].values,
    ])

    targets_mw = facility_mw.copy()
    power_change = np.diff(targets_mw, prepend=targets_mw[0])
    signals = np.zeros(len(targets_mw), dtype=int)
    signals[power_change > 0.5] = 1
    signals[power_change < -0.5] = 2

    X, Y_power, Y_signal = [], [], []
    for i in range(0, len(features) - seq_len - pred_horizon, stride):
        X.append(features[i:i + seq_len])
        Y_power.append(targets_mw[i + seq_len:i + seq_len + pred_horizon])
        Y_signal.append(signals[i + seq_len])

    return (
        np.array(X, dtype=np.float32),
        np.array(Y_power, dtype=np.float32),
        np.array(Y_signal, dtype=np.int64),
    )


class RealH100Dataset(Dataset):
    """PyTorch Dataset for real H100 data sequences."""

    def __init__(self, X: np.ndarray, Y_power: np.ndarray, Y_signal: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_power = torch.tensor(Y_power, dtype=torch.float32)
        self.Y_signal = torch.tensor(Y_signal, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_power[idx], self.Y_signal[idx]


def build_dataloaders(
    seq_len: int = 30,
    pred_horizon: int = 10,
    batch_size: int = 64,
    num_gpus: int = 200000,
    data_dir: Optional[str] = None,
    val_split: float = 0.15,
    stride: int = 50,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build train/val dataloaders from real H100 LLM data.

    Returns:
        train_loader, val_loader, n_features
    """
    if data_dir is None:
        data_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
            "real_h100", "High-resolution-AI-Data-Center-Training-Workloads-Dataset_FigShare"
        ))

    files = sorted(glob.glob(os.path.join(
        data_dir, "Node_Dataset", "Text Generation LLMs", "H100", "**", "*.csv"
    ), recursive=True))

    print(f"Found {len(files)} H100 LLM files")

    all_X, all_Yp, all_Ys = [], [], []
    for f in files:
        print(f"  Loading {os.path.basename(f)}...")
        df = load_node_data(f)
        X, Yp, Ys = create_sequences(df, seq_len, pred_horizon, num_gpus, stride=stride)
        print(f"    -> {len(X)} sequences, power range: {Yp.min():.1f}-{Yp.max():.1f} MW")
        all_X.append(X)
        all_Yp.append(Yp)
        all_Ys.append(Ys)

    X = np.concatenate(all_X, axis=0)
    Yp = np.concatenate(all_Yp, axis=0)
    Ys = np.concatenate(all_Ys, axis=0)

    print(f"\nTotal: {len(X)} sequences")
    print(f"Power range: {Yp.min():.1f} - {Yp.max():.1f} MW")

    indices = np.random.permutation(len(X))
    split = int(len(X) * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = RealH100Dataset(X[train_idx], Yp[train_idx], Ys[train_idx])
    val_ds = RealH100Dataset(X[val_idx], Yp[val_idx], Ys[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_loader, val_loader, X.shape[2]
