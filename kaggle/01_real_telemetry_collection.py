"""
Real GPU Telemetry Collection — Kaggle Notebook Template
=========================================================
Copy this script into a Kaggle notebook cell-by-cell or run as a single script.

What it does:
  1. Installs dependencies (torch, codecarbon, pandas, matplotlib)
  2. Creates a synthetic training workload that mimics LLM training
  3. Collects GPU telemetry (power, temperature, utilization) every second
  4. Tracks energy consumption with CodeCarbon
  5. Saves all data to CSV for downstream analysis
  6. Visualizes power traces, temperature, utilization, and energy cost

Requirements:
  - Kaggle notebook with GPU runtime (T4, P100, or V100)
  - Enable GPU in: Settings → Accelerator → GPU

Output files:
  - gpu_telemetry.csv      — per-second GPU metrics
  - energy_report.csv      — per-epoch energy + cost
  - carbon_report.csv      — CodeCarbon emissions data
  - telemetry_summary.json — aggregate statistics
"""

# %% [markdown]
# # Energivanu — Real GPU Telemetry Collection
#
# Collect real GPU power, temperature, and utilization data during model training.
# This data feeds the Energivanu Predictive Energy Buffer (PEB) model.

# %% Cell 1: Install dependencies
# !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
# !pip install codecarbon pandas matplotlib numpy 2>/dev/null

# %% Cell 2: Imports
import csv
import json
import math
import os
import subprocess
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
    print(f"GPU count:       {torch.cuda.device_count()}")

# Try importing CodeCarbon
try:
    from codecarbon import EmissionsTracker

    CODECARBON_AVAILABLE = True
    print("CodeCarbon:      available ✓")
except ImportError:
    CODECARBON_AVAILABLE = False
    print("CodeCarbon:      not available (fallback estimator)")


# %% Cell 3: GPU Telemetry Collector
class GpuTelemetryCollector:
    """
    Collects GPU metrics via nvidia-smi at 1-second intervals.

    Metrics collected per GPU:
      - power_draw_w:     Power consumption in watts
      - temperature_c:    GPU core temperature
      - gpu_util_pct:     SM utilization percentage
      - mem_util_pct:     Memory utilization percentage
      - sm_clock_mhz:     SM clock frequency
      - mem_clock_mhz:    Memory clock frequency
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.data = []
        self._running = False
        self._start_time = None

    def _query_nvidia_smi(self) -> list:
        """Query nvidia-smi for per-GPU metrics."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,power.draw,temperature.gpu,"
                    "utilization.gpu,utilization.memory,"
                    "clocks.current.graphics,clocks.current.memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []

            rows = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 7:
                    try:
                        rows.append({
                            "gpu_id": int(parts[0]),
                            "power_w": float(parts[1]),
                            "temp_c": float(parts[2]),
                            "gpu_util_pct": float(parts[3]),
                            "mem_util_pct": float(parts[4]),
                            "sm_clock_mhz": float(parts[5]),
                            "mem_clock_mhz": float(parts[6]),
                        })
                    except (ValueError, IndexError):
                        continue
            return rows
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return []

    def collect_once(self) -> list:
        """Collect one round of telemetry data."""
        now = datetime.now(timezone.utc)
        unix_ts = time.time()
        rows = self._query_nvidia_smi()

        samples = []
        for row in rows:
            sample = {
                "timestamp": now.isoformat(),
                "unix_ts": unix_ts,
                "elapsed_s": unix_ts - self._start_time if self._start_time else 0,
                **row,
            }
            samples.append(sample)
            self.data.append(sample)

        return samples

    def start(self):
        """Start collection (marks t=0)."""
        self._start_time = time.time()
        self._running = True
        print(f"Telemetry collection started at {datetime.now(timezone.utc).isoformat()}")

    def stop(self):
        """Stop collection."""
        self._running = False
        total = len(self.data)
        elapsed = time.time() - self._start_time if self._start_time else 0
        print(f"Telemetry collection stopped. {total} samples in {elapsed:.1f}s")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected data to a pandas DataFrame."""
        return pd.DataFrame(self.data)

    def to_csv(self, filepath: str = "gpu_telemetry.csv"):
        """Save collected data to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} telemetry samples to {filepath}")
        return filepath


# %% Cell 4: Training Workload (Synthetic LLM-like)
class SyntheticLLMDataset(Dataset):
    """
    Synthetic dataset that mimics LLM training compute patterns.
    Generates random tensors to exercise the GPU.
    """

    def __init__(self, num_samples: int = 500, input_dim: int = 2048, seq_len: int = 128):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate token embeddings
        x = torch.randn(self.seq_len, self.input_dim)
        y = torch.randint(0, 1000, (self.seq_len,))
        return x, y


class SimpleTransformerBlock(nn.Module):
    """Lightweight transformer block for GPU exercise."""

    def __init__(self, d_model: int = 2048, nhead: int = 16, dim_ff: int = 8192):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


def create_training_model(d_model: int = 2048, n_layers: int = 4, nhead: int = 16):
    """Create a transformer model for GPU workload generation."""
    layers = [SimpleTransformerBlock(d_model, nhead) for _ in range(n_layers)]
    model = nn.Sequential(
        nn.Linear(d_model, d_model),
        *[layer for layer in layers],
        nn.Linear(d_model, 1000),
    )
    return model


# %% Cell 5: Training Loop with Telemetry
def train_with_telemetry(
    num_epochs: int = 3,
    batch_size: int = 8,
    num_samples: int = 200,
    d_model: int = 2048,
    n_layers: int = 4,
    learning_rate: float = 1e-4,
    collect_interval_s: float = 1.0,
):
    """
    Train a model while collecting GPU telemetry.

    Returns:
        Tuple of (telemetry_collector, training_stats, emissions_tracker)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")

    # Create model and data
    model = create_training_model(d_model=d_model, n_layers=n_layers).to(device)
    dataset = SyntheticLLMDataset(num_samples=num_samples, input_dim=d_model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Initialize telemetry collector
    collector = GpuTelemetryCollector(interval_s=collect_interval_s)

    # Initialize CodeCarbon tracker
    emissions_tracker = None
    if CODECARBON_AVAILABLE:
        emissions_tracker = EmissionsTracker(
            project_name="energivanu-telemetry",
            output_dir=".",
            log_level="warning",
        )

    # Training stats
    epoch_stats = []
    collector.start()

    if emissions_tracker:
        emissions_tracker.start()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_start = time.time()

        # Collect telemetry at start of epoch
        collector.collect_once()

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            out = model(x)
            # Reshape for cross-entropy: (B*seq, classes) vs (B*seq,)
            loss = loss_fn(out.view(-1, 1000), y.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1

            # Collect telemetry periodically
            if batch_idx % 5 == 0:
                collector.collect_once()

        scheduler.step()
        epoch_duration = time.time() - epoch_start
        avg_loss = epoch_loss / max(1, epoch_batches)

        # Collect telemetry at end of epoch
        collector.collect_once()

        # Get energy data
        energy_kwh = 0.0
        emissions_kg = 0.0
        if emissions_tracker:
            try:
                emissions_data = emissions_tracker._prepare_emissions_data()
                energy_kwh = float(emissions_data.energy.kWh)
                emissions_kg = float(emissions_data.emissions)
            except Exception:
                pass

        # Estimate power from telemetry
        recent_samples = [s for s in collector.data if s.get("gpu_id") == 0][-10:]
        avg_power = np.mean([s["power_w"] for s in recent_samples]) if recent_samples else 0

        cost_usd = energy_kwh * 0.12  # $0.12/kWh

        stats = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "duration_s": round(epoch_duration, 2),
            "energy_kwh": round(energy_kwh, 6),
            "emissions_kg": round(emissions_kg, 6),
            "cost_usd": round(cost_usd, 6),
            "avg_gpu_power_w": round(avg_power, 1),
        }
        epoch_stats.append(stats)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
            f"Time: {epoch_duration:.1f}s | Power: {avg_power:.0f}W | "
            f"Energy: {energy_kwh:.4f} kWh | Cost: ${cost_usd:.4f}"
        )

    collector.stop()

    if emissions_tracker:
        emissions_tracker.stop()

    return collector, epoch_stats, emissions_tracker


# %% Cell 6: Run Training and Collect Data
print("=" * 70)
print("Energivanu — Real GPU Telemetry Collection")
print("=" * 70)

collector, epoch_stats, emissions_tracker = train_with_telemetry(
    num_epochs=3,
    batch_size=4,
    num_samples=100,
    d_model=1024,       # Smaller model for Kaggle T4
    n_layers=2,
    learning_rate=1e-4,
    collect_interval_s=1.0,
)

# %% Cell 7: Save Data
# Save telemetry to CSV
telemetry_path = collector.to_csv("gpu_telemetry.csv")

# Save epoch energy report
with open("energy_report.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=epoch_stats[0].keys())
    writer.writeheader()
    writer.writerows(epoch_stats)
print(f"Saved energy report to energy_report.csv")

# Export CodeCarbon data if available
if emissions_tracker and CODECARBON_AVAILABLE:
    print("CodeCarbon data saved to emissions.csv (auto-generated)")

# Save summary JSON
summary = {
    "collection_time": datetime.now(timezone.utc).isoformat(),
    "total_samples": len(collector.data),
    "num_gpus": len(set(s["gpu_id"] for s in collector.data)),
    "epochs": len(epoch_stats),
    "total_energy_kwh": sum(s["energy_kwh"] for s in epoch_stats),
    "total_cost_usd": sum(s["cost_usd"] for s in epoch_stats),
    "avg_gpu_power_w": np.mean([s["power_w"] for s in collector.data if s["gpu_id"] == 0]),
    "max_gpu_temp_c": max(s["temp_c"] for s in collector.data),
}
with open("telemetry_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary: {json.dumps(summary, indent=2)}")

# %% Cell 8: Visualization — Power Trace
df = collector.to_dataframe()

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
fig.suptitle("GPU Telemetry During Training", fontsize=14, fontweight="bold")

# Plot per-GPU power
gpu_ids = sorted(df["gpu_id"].unique())
for gpu_id in gpu_ids:
    gpu_df = df[df["gpu_id"] == gpu_id]
    axes[0].plot(gpu_df["elapsed_s"], gpu_df["power_w"], alpha=0.7, label=f"GPU {gpu_id}")
axes[0].set_ylabel("Power (W)")
axes[0].set_title("GPU Power Draw")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(True, alpha=0.3)

# Plot temperature
for gpu_id in gpu_ids:
    gpu_df = df[df["gpu_id"] == gpu_id]
    axes[1].plot(gpu_df["elapsed_s"], gpu_df["temp_c"], alpha=0.7, label=f"GPU {gpu_id}")
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_title("GPU Temperature")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].grid(True, alpha=0.3)

# Plot utilization
for gpu_id in gpu_ids:
    gpu_df = df[df["gpu_id"] == gpu_id]
    axes[2].plot(gpu_df["elapsed_s"], gpu_df["gpu_util_pct"], alpha=0.7, label=f"GPU {gpu_id}")
axes[2].set_ylabel("Utilization (%)")
axes[2].set_title("GPU SM Utilization")
axes[2].legend(loc="upper right", fontsize=8)
axes[2].grid(True, alpha=0.3)

# Plot memory utilization
for gpu_id in gpu_ids:
    gpu_df = df[df["gpu_id"] == gpu_id]
    axes[3].plot(gpu_df["elapsed_s"], gpu_df["mem_util_pct"], alpha=0.7, label=f"GPU {gpu_id}")
axes[3].set_ylabel("Memory Util (%)")
axes[3].set_title("GPU Memory Utilization")
axes[3].set_xlabel("Time (seconds)")
axes[3].legend(loc="upper right", fontsize=8)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("telemetry_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved telemetry_plots.png")

# %% Cell 9: Visualization — Energy and Cost
if epoch_stats:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Energy & Cost Analysis", fontsize=14, fontweight="bold")

    epochs = [s["epoch"] + 1 for s in epoch_stats]
    losses = [s["loss"] for s in epoch_stats]
    energies = [s["energy_kwh"] for s in epoch_stats]
    costs = [s["cost_usd"] for s in epoch_stats]
    powers = [s["avg_gpu_power_w"] for s in epoch_stats]

    axes[0, 0].bar(epochs, losses, color="#2196F3")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(epochs, energies, color="#4CAF50")
    axes[0, 1].set_title("Energy per Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Energy (kWh)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(epochs, costs, color="#FF9800")
    axes[1, 0].set_title("Cost per Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Cost (USD)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(epochs, powers, color="#F44336")
    axes[1, 1].set_title("Average GPU Power")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Power (W)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("energy_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved energy_analysis.png")

# %% Cell 10: Feature Extraction for Energivanu Model
def extract_features_from_telemetry(df: pd.DataFrame, num_gpus_facility: int = 200000) -> pd.DataFrame:
    """
    Extract 15 Energivanu features from raw telemetry data.

    This matches the feature format in src/energivanu/data.py so the collected
    data can be used directly for model training.
    """
    # Aggregate across GPUs per timestamp
    agg = df.groupby("unix_ts").agg({
        "power_w": ["sum", "mean", "max", "std"],
        "temp_c": ["mean", "max"],
        "gpu_util_pct": "mean",
        "mem_util_pct": "mean",
    }).reset_index()

    agg.columns = [
        "unix_ts", "node_power_w", "gpu_avg_power", "gpu_max_power",
        "gpu_std_power", "gpu_avg_temp", "gpu_max_temp",
        "gpu_avg_util", "gpu_avg_mem_util",
    ]
    agg = agg.sort_values("unix_ts").reset_index(drop=True)

    # Scale to facility MW
    num_gpus = df["gpu_id"].nunique()
    scale = num_gpus_facility / num_gpus / 1e6
    agg["facility_mw"] = agg["node_power_w"] * scale

    # Derivatives
    agg["power_roc"] = agg["facility_mw"].diff().fillna(0)
    agg["power_roc2"] = agg["power_roc"].diff().fillna(0)

    # Rolling stats
    agg["power_roll_mean"] = agg["facility_mw"].rolling(5, min_periods=1).mean()
    agg["power_roll_std"] = agg["facility_mw"].rolling(5, min_periods=1).std().fillna(0)

    # Normalized features
    agg["gpu_avg_power_norm"] = agg["gpu_avg_power"] / 700.0
    agg["gpu_max_power_norm"] = agg["gpu_max_power"] / 700.0
    agg["gpu_avg_temp_norm"] = agg["gpu_avg_temp"] / 100.0
    agg["gpu_max_temp_norm"] = agg["gpu_max_temp"] / 100.0
    agg["gpu_avg_util_norm"] = agg["gpu_avg_util"] / 100.0
    agg["gpu_avg_mem_util_norm"] = agg["gpu_avg_mem_util"] / 100.0
    agg["cpu_util_est_norm"] = 0.4  # Estimated

    # Time encoding
    dt_series = pd.to_datetime(agg["unix_ts"], unit="s", utc=True)
    agg["hour_sin"] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
    agg["hour_cos"] = np.cos(2 * np.pi * dt_series.dt.hour / 24)

    # All-reduce heuristic
    agg["is_allreduce"] = (
        (agg["gpu_avg_util"] > 80) & (agg["gpu_avg_mem_util"] < 30)
    ).astype(float)

    return agg


# Extract features
if len(collector.data) > 0:
    raw_df = collector.to_dataframe()
    features_df = extract_features_from_telemetry(raw_df)

    # Save features
    feature_cols = [
        "facility_mw", "power_roc", "power_roc2", "power_roll_mean",
        "power_roll_std", "gpu_avg_power_norm", "gpu_max_power_norm",
        "gpu_avg_temp_norm", "gpu_max_temp_norm", "gpu_avg_util_norm",
        "gpu_avg_mem_util_norm", "cpu_util_est_norm", "hour_sin", "hour_cos",
        "is_allreduce",
    ]
    features_df[feature_cols].to_csv("features_for_training.csv", index=False)
    print(f"\nExtracted {len(features_df)} feature vectors with {len(feature_cols)} features")
    print(f"Saved to features_for_training.csv")
    print(f"\nFeature statistics:")
    print(features_df[feature_cols].describe().round(4))

# %% Cell 11: Summary
print("\n" + "=" * 70)
print("COLLECTION COMPLETE")
print("=" * 70)
print(f"\nOutput files:")
print(f"  gpu_telemetry.csv        — raw per-GPU metrics ({len(collector.data)} rows)")
print(f"  energy_report.csv        — per-epoch energy & cost")
print(f"  telemetry_summary.json   — aggregate statistics")
print(f"  telemetry_plots.png      — power/temp/util visualization")
print(f"  energy_analysis.png      — energy & cost visualization")
if len(collector.data) > 0:
    print(f"  features_for_training.csv — 15-feature vectors for Energivanu model")
print(f"\nNext steps:")
print(f"  1. Download these CSV files from the Kaggle notebook output")
print(f"  2. Place in Energivanu/data/real_telemetry/")
print(f"  3. Use with energivanu.data.build_dataloaders() for model training")
print(f"  4. Train the PEB model: python -m energivanu.train_real")
