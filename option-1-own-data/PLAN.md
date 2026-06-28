# Option 1 — Apna Khud Ka Data Collect Karo
## Complete Plan & Execution Guide

> **Goal:** Collect our own GPU telemetry data so we have FULL commercial rights
> to any model trained on it. Zero license restrictions.

---

## 1. Kya Data Chahiye (Feature Requirements)

Existing `data.py` ke hisaab se 15 features chahiye:

| # | Feature | Source | Unit |
|---|---------|--------|------|
| 0 | facility_mw | nvidia-smi power × scale | MW |
| 1 | power_roc | diff(power) | MW/s |
| 2 | power_roc2 | diff(roc) | MW/s² |
| 3 | power_roll_mean | rolling mean | MW |
| 4 | power_roll_std | rolling std | MW |
| 5 | gpu_avg_power_norm | avg power / 700 | 0-1 |
| 6 | gpu_max_power_norm | max power / 700 | 0-1 |
| 7 | gpu_avg_temp_norm | avg temp / 100 | 0-1 |
| 8 | gpu_max_temp_norm | max temp / 100 | 0-1 |
| 9 | gpu_avg_util_norm | avg util / 100 | 0-1 |
| 10 | gpu_avg_mem_util_norm | mem util / 100 | 0-1 |
| 11 | cpu_util_est_norm | estimated | 0-1 |
| 12 | hour_sin | sin(2π×h/24) | -1 to 1 |
| 13 | hour_cos | cos(2π×h/24) | -1 to 1 |
| 14 | is_allreduce | heuristic | 0 or 1 |

**Raw data needed per GPU per second:**
- `power_w` — Power draw in watts
- `temp_c` — Core temperature in °C
- `util_pct` — SM/GPU utilization %
- `mem_util_pct` — Memory utilization %
- `sm_clock_mhz` — SM clock frequency
- `mem_clock_mhz` — Memory clock frequency

---

## 2. Data Collection Strategy

### 2.1 Target GPUs (Priority Order)

| Priority | GPU | Where | Cost | Why |
|----------|-----|-------|------|-----|
| 🥇 P0 | RTX 4090 | Kaggle (free T4), Local | Free-$1600 | Most common consumer GPU |
| 🥈 P0 | T4 | Kaggle, Colab | Free | Free, accessible to everyone |
| 🥉 P1 | A100 | Lambda Cloud, RunPod | $1.10/hr | Datacenter standard |
| P2 | H100 | Lambda, Together.ai | $2-4/hr | Premium, for benchmarks |
| P3 | RTX 3090 | Local, Colab Pro | Free-$800 | Previous gen, still common |

### 2.2 Minimum Viable Dataset

| Metric | Minimum | Good | Production |
|--------|---------|------|------------|
| Duration per GPU | 1 hour | 8 hours | 24+ hours |
| Sampling interval | 1 sec | 1 sec | 0.5-1 sec |
| Num GPUs | 1 | 4-8 | 8+ |
| Total samples | 3,600 | 28,800 | 86,400+ |
| Workload types | 2 | 4 | 6+ |
| File size estimate | ~5 MB | ~50 MB | ~200 MB |

### 2.3 Workload Types to Simulate

| Workload | GPU Util Pattern | Power Pattern | Duration |
|----------|-----------------|---------------|----------|
| **LLM Training** | High (80-95%), periodic dips for all-reduce | High, cyclical | 30+ min |
| **LLM Inference** | Medium (40-70%), bursty | Medium, bursty | 15+ min |
| **Image Training (CNN)** | Steady high (90%+) | Steady high | 20+ min |
| **Idle/Standby** | Low (0-5%) | Low (50-100W) | 10+ min |
| **Mixed Workload** | Variable | Variable | 30+ min |
| **Stress Test** | Max (99%) | Max (TDP) | 10+ min |

### 2.4 Collection Script (Ready to Use)

```python
# Run this on ANY machine with a GPU:
# pip install torch pandas
# python kaggle/01_real_telemetry_collection.py

# Or use the telemetry collector:
from energivanu.telemetry import NvidiaSmiCollector

with NvidiaSmiCollector(
    collection_interval_s=1.0,
    simulation_mode=False,  # Real GPU!
    storage_backend="csv",
    csv_path="my_gpu_data.csv",
) as collector:
    import time
    time.sleep(3600)  # Collect for 1 hour
```

---

## 3. Multi-Source Collection Plan

### Source 1: Kaggle Notebooks (FREE) ⭐
- **GPU:** Tesla T4 (16GB)
- **Cost:** Free (30 hrs/week GPU)
- **How:** Copy `kaggle/01_real_telemetry_collection.py` → Run → Download CSV
- **Limitation:** 1 GPU per notebook, 12-hour max session

### Source 2: Google Colab (FREE)
- **GPU:** T4 or V100 (Colab Pro)
- **Cost:** Free (limited) / $10/mo (Pro)
- **How:** Same Kaggle script works in Colab
- **Limitation:** Session timeouts, disconnections

### Source 3: Lambda Cloud ($1.10/hr)
- **GPU:** A100 40GB/80GB, H100
- **Cost:** ~$1.10/hr (A100), ~$2.50/hr (H100)
- **How:** SSH → install → run collector for 24hrs
- **Best for:** Production-quality datacenter GPU data

### Source 4: RunPod ($0.20-0.40/hr)
- **GPU:** RTX 3090, RTX 4090, A100
- **Cost:** $0.20/hr (3090) to $1.50/hr (A100)
- **How:** Same as Lambda
- **Best for:** Budget datacenter data

### Source 5: Local Machine (FREE)
- **GPU:** Whatever you have
- **Cost:** Free (electricity only)
- **How:** Run collector overnight
- **Best for:** Long-duration, real-world data

### Collection Budget Estimate

| Scenario | GPUs | Hours | Total Cost |
|----------|------|-------|------------|
| Minimum viable | Kaggle T4 × 3 sessions | 9 hrs | $0 |
| Good dataset | Kaggle + Colab + Lambda A100 × 4hrs | 20 hrs | ~$5 |
| Production | Lambda A100 × 24hr + Kaggle × 8hr | 32 hrs | ~$27 |
| Premium (H100 data) | Lambda H100 × 12hr | 12 hrs | ~$30 |

---

## 4. Compatibility Verification

### ✅ Verified: Kaggle Template → `build_dataloaders()`

The Kaggle template (`kaggle/01_real_telemetry_collection.py`) already:
1. Collects raw data: `power_w, temp_c, gpu_util_pct, mem_util_pct, sm_clock_mhz, mem_clock_mhz`
2. Extracts 15 features matching `data.py` format
3. Saves to CSV with correct column names
4. Includes `extract_features_from_telemetry()` function

### Data Pipeline Flow:
```
nvidia-smi → GpuTelemetryCollector → CSV
    ↓
extract_features_from_telemetry() → 15-feature CSV
    ↓
load_node_data() → create_sequences() → DataLoader
    ↓
EnergivanuPEB model training
```

### ⚠️ One Gap to Fix:
The current `data.py` expects columns like `gpu0_power_W`, `gpu1_power_W`, etc. (per-GPU columns).
Our collector saves per-GPU rows. Need a conversion step:

```python
# Add to data.py or create adapter:
def collector_csv_to_training_format(csv_path: str) -> pd.DataFrame:
    """Convert collector CSV (per-GPU rows) to training format (per-GPU columns)."""
    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(
        index='unix_ts', columns='gpu_id',
        values=['power_w', 'temp_c', 'gpu_util_pct', 'mem_util_pct']
    )
    # Rename to match expected format
    # gpu0_power_W, gpu1_power_W, etc.
    ...
```

---

## 5. Legal Analysis — Self-Collected Data

### ✅ FULL RIGHTS — No Restrictions

| Question | Answer |
|----------|--------|
| Who owns the data? | **You** (the person who collected it) |
| Can you use it commercially? | **Yes** |
| Can you distribute models trained on it? | **Yes** |
| Can you use it in advertising? | **Yes** |
| Any attribution required? | **No** (you're the creator) |
| Any restrictions? | **None** |

### ⚠️ Only Caveats:
1. **Cloud provider ToS** — Check if Lambda/RunPod ToS restricts telemetry collection (usually they don't)
2. **nvidia-smi output** — nvidia-smi is NVIDIA's tool, but its output is yours. No restrictions.
3. **Kaggle ToS** — Data you generate on Kaggle is yours. Kaggle gets a license to host it, but you keep full rights.

---

## 6. Execution Timeline

### Week 1: Quick Wins (FREE)
| Day | Action | Output |
|-----|--------|--------|
| 1 | Run Kaggle notebook × 3 sessions | ~9hrs T4 data |
| 2 | Run Colab sessions × 3 | ~9hrs T4/V100 data |
| 3 | Run local GPU overnight | ~8hrs local data |
| 4 | Combine all CSVs, verify features | Combined dataset |
| 5 | Test with `build_dataloaders()` | Validation |

### Week 2: Production Quality ($10-30)
| Day | Action | Output |
|-----|--------|--------|
| 8 | Lambda Cloud A100 × 8hrs | ~8hrs A100 data |
| 9 | Lambda Cloud A100 × 16hrs | ~16hrs A100 data |
| 10 | Different workloads (training, inference, idle) | Diverse data |
| 11 | Combine, clean, validate | Final dataset |
| 12 | Train model, measure MAPE | Benchmark |

### Week 3: Polish
| Day | Action | Output |
|-----|--------|--------|
| 15 | Document collection process | README |
| 16 | Package dataset for distribution | ZIP/tarball |
| 17 | Train final commercial model | ONNX export |
| 18 | Write blog/README about methodology | Marketing material |

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Kaggle session timeout | Medium | Low | Use multiple sessions |
| Insufficient data diversity | Medium | Medium | Collect from multiple GPU types |
| Feature mismatch | Low | High | Verified against data.py |
| Cloud ToS violation | Very Low | High | Read ToS, use nvidia-smi (standard tool) |
| Data quality issues | Medium | Medium | Validate ranges, remove outliers |

---

## 8. Verdict

| Criteria | Rating |
|----------|--------|
| Legal safety | ⭐⭐⭐⭐⭐ (100% safe) |
| Cost | ⭐⭐⭐⭐⭐ ($0-30) |
| Effort | ⭐⭐⭐⭐ (Medium — mostly waiting) |
| Data quality | ⭐⭐⭐⭐ (Good, depends on GPU variety) |
| Commercial readiness | ⭐⭐⭐⭐⭐ (Fully commercial-ready) |

**RECOMMENDATION: DO THIS FIRST. Even if you do Option 2 or 3, collect your own data too. It's cheap, safe, and gives you a fully commercial-ready fallback.**
