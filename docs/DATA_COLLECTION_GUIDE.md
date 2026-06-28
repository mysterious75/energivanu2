# Data Collection Guide

Step-by-step instructions for collecting GPU telemetry data for Energivanu training.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Method 1: nvidia-smi Polling](#method-1-nvidia-smi-polling)
3. [Method 2: Kaggle/Colab Notebook](#method-2-kagglecolab-notebook)
4. [Method 3: Alibaba Dataset](#method-3-alibaba-dataset)
5. [Data Format Requirements](#data-format-requirements)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

The fastest way to get training data:

```bash
# Option A: Use synthetic data (no hardware needed)
python -m energivanu.train_commercial --sources synthetic

# Option B: Collect your own data from a GPU machine
python scripts/collect_data.py --duration 3600 --interval 1

# Option C: Download Alibaba dataset (requires processing)
python scripts/download_alibaba_data.py
```

---

## Method 1: nvidia-smi Polling

Collect GPU telemetry directly from any NVIDIA GPU system.

### Prerequisites

- NVIDIA GPU with drivers installed
- `nvidia-smi` available in PATH
- Python 3.9+

### Step 1: Verify GPU Access

```bash
nvidia-smi
```

You should see a table showing your GPU(s). If this fails, install NVIDIA drivers:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535

# Verify
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,utilization.gpu --format=csv
```

### Step 2: Collect Raw Telemetry

Use `nvidia-smi` in daemon mode to poll at regular intervals:

```bash
# Collect every 1 second for 1 hour (3600 samples)
nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv -l 1 > data/raw_telemetry.csv
```

Or use the Energivanu telemetry module:

```python
from energivanu.telemetry.nvidia_smi_collector import NVIDIASmiCollector

collector = NVIDIASmiCollector(
    gpu_ids=(0, 1, 2, 3, 4, 5, 6, 7),
    collection_interval_s=1.0,
    storage_backend="csv",
    csv_path="data/my_telemetry.csv",
)
collector.start()
# ... run your workload ...
collector.stop()
```

### Step 3: Run a Workload (Optional)

For realistic power patterns, run a training job while collecting:

```bash
# Simple GPU stress test
python -c "
import torch
x = torch.randn(1000, 1000, device='cuda')
for i in range(3600):
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    import time; time.sleep(1)
"

# Or run an actual training job
python train.py --epochs 1
```

### Step 4: Process Raw Data

Convert raw nvidia-smi CSV to training format:

```python
from energivanu.telemetry.format_adapter import FormatAdapter

adapter = FormatAdapter()
adapter.convert(
    input_path="data/raw_telemetry.csv",
    output_path="data/kaggle_t4/kaggle_processed.npz",
    seq_len=30,
    pred_horizon=10,
    stride=50,
)
```

---

## Method 2: Kaggle/Colab Notebook

Use free cloud GPU instances to collect T4 telemetry.

### Step 1: Open Kaggle Notebook

1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Create a new notebook
3. Enable GPU: **Settings → Accelerator → GPU T4 x2**

### Step 2: Copy the Collection Script

Copy the contents of `kaggle/01_real_telemetry_collection.py` into a Kaggle notebook cell. This script:

- Installs required dependencies
- Collects GPU telemetry during a synthetic training loop
- Saves results as CSV
- Generates visualization plots

### Step 3: Run and Download

1. Run all cells in the notebook
2. Download the output CSV from **Data → Output**
3. Place it in `data/kaggle_t4/` in your Energivanu project

### Step 4: Process for Training

```bash
python scripts/process_kaggle_data.py --input data/kaggle_t4/raw_output.csv
```

---

## Method 3: Alibaba Dataset

Use the Alibaba Cluster Trace GPU v2020 dataset (CC BY 4.0).

### Step 1: Download

```bash
# Clone the Alibaba clusterdata repository
git clone https://github.com/alibaba/clusterdata.git /tmp/alibaba_clusterdata

# The GPU trace is in:
# /tmp/alibaba_clusterdata/cluster-trace-gpu-v2020/
```

Or use the download script:

```bash
python scripts/download_alibaba_data.py --output data/alibaba_gpu_trace/
```

### Step 2: Understand the Data

The Alibaba dataset contains these tables:

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `pai_sensor_table` | GPU sensor readings | `gpu_util`, `gpu_memory_util`, `gpu_temp` |
| `pai_instance_table` | Instance metrics | `cpu_util`, `memory_usage` |
| `pai_job_table` | Job metadata | `job_name`, `framework`, `status` |
| `pai_machine_spec` | Hardware specs | `machine_id`, `gpu_type`, `gpu_count` |

### Step 3: Process to Training Format

```bash
python scripts/process_alibaba_data.py \
    --input data/alibaba_gpu_trace/ \
    --output data/alibaba_gpu_trace/alibaba_processed.npz \
    --power-model h100
```

The power model converts utilization to estimated power:

```python
# H100 power model
P_gpu = 70 + 630 * (gpu_util / 100)  # Watts

# Scale to facility level
facility_mw = P_gpu * num_gpus / 1e6
```

### Step 4: Citation

When using Alibaba data, you **must** cite:

```bibtex
@inproceedings{weng2022mlaas,
  title={{MLaaS} in the Wild: Workload Analysis and Scheduling
         in Large-Scale Heterogeneous {GPU} Clusters},
  author={Weng, Qizhen and Xiao, Wencong and Yu, Yinghao and
          Wang, Wei and Wang, Cheng and He, Jian and Li, Yong
          and Zhang, Liping and Lin, Wei and Ding, Yu},
  booktitle={NSDI '22},
  year={2022}
}
```

---

## Data Format Requirements

### Training-Ready Format (.npz)

The training pipeline expects NumPy `.npz` files with these keys:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `X` | `(N, 30, 15)` | float32 | Input feature sequences |
| `Y_power` | `(N, 10)` | float32 | Power targets (MW) |
| `Y_signal` | `(N,)` | int64 | Signal labels (0=hold, 1=discharge, 2=charge) |

### Feature Vector (15 dimensions)

Each timestep must have exactly 15 features in this order:

```
[0]  facility_mw          — Facility power in MW
[1]  power_roc            — Power rate of change (MW/s)
[2]  power_roc2           — Power second derivative (MW/s²)
[3]  power_roll_mean      — Rolling mean power (MW)
[4]  power_roll_std       — Rolling std deviation
[5]  gpu_avg_power_norm   — Avg GPU power / 700W
[6]  gpu_max_power_norm   — Max GPU power / 700W
[7]  gpu_avg_temp_norm    — Avg temperature / 100°C
[8]  gpu_max_temp_norm    — Max temperature / 100°C
[9]  gpu_avg_util_norm    — Avg GPU utilization / 100%
[10] gpu_avg_mem_util_norm — Avg memory utilization / 100%
[11] cpu_util_norm        — CPU utilization / 100%
[12] hour_sin             — Hour of day (sin encoding)
[13] hour_cos             — Hour of day (cos encoding)
[14] is_allreduce         — All-Reduce phase indicator
```

### Raw CSV Format

If collecting with nvidia-smi, the raw CSV should have:

```csv
timestamp, gpu_index, gpu_name, temperature.gpu, power.draw, utilization.gpu, utilization.memory, memory.total, memory.used, memory.free
2026/06/28 12:00:00.000, 0, NVIDIA GeForce RTX 4090, 45, 120.50 W, 85, 70, 24564 MiB, 17194 MiB, 7370 MiB
```

---

## Troubleshooting

### "nvidia-smi: command not found"

```bash
# Check if NVIDIA drivers are installed
ls /usr/bin/nvidia-smi
ls /usr/local/cuda/bin/nvidia-smi

# Add to PATH if needed
export PATH="/usr/local/cuda/bin:$PATH"

# Or install drivers
sudo apt install nvidia-driver-535
```

### "No devices were found"

```bash
# Check if GPU is visible
lspci | grep -i nvidia

# Check driver version
cat /proc/driver/nvidia/version

# Restart nvidia driver
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
```

### "Permission denied" on nvidia-smi

```bash
# Add yourself to the video group
sudo usermod -aG video $USER
# Log out and back in
```

### Kaggle: "GPU not available"

1. Go to notebook **Settings**
2. Set **Accelerator** to **GPU T4 x2**
3. Save and restart the notebook
4. Verify: `!nvidia-smi`

### Alibaba data: "File not found"

The Alibaba dataset is split into multiple parts. Ensure you've downloaded all parts:

```bash
ls data/alibaba_gpu_trace/
# Should contain: pai_sensor_table, pai_instance_table, etc.
```

### Training: "Shape mismatch"

Your data has wrong feature dimensions. Verify:

```python
import numpy as np
data = np.load("data/kaggle_t4/kaggle_processed.npz")
print(data["X"].shape)    # Should be (N, 30, 15)
print(data["Y_power"].shape)  # Should be (N, 10)
print(data["Y_signal"].shape)  # Should be (N,)
```

If shapes are wrong, re-run the format adapter with correct `seq_len` and `pred_horizon`.

### Low MAPE on synthetic, high MAPE on real data

This is expected. Synthetic data uses simplified patterns. For better real-data performance:

1. Use more diverse training data (Alibaba + own collection)
2. Increase training epochs
3. Use data augmentation (noise injection, time warping)
4. Fine-tune on your specific GPU type

---

## Support

- **Issues**: [GitHub Issues](https://github.com/mysterious75/Energivanu/issues)
- **Questions**: Open a Discussion on GitHub
- **Commercial support**: Contact via [@VEDKUMAR98](https://x.com/VEDKUMAR98)
