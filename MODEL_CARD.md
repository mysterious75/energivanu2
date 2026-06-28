# Model Card — Energivanu PEB

> Following the [Google Model Cards](https://modelcards.withgoogle.com/) format.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | EnergivanuPEB (Predictive Energy Buffer) |
| **Version** | 1.0 |
| **Architecture** | TCN + Multi-Head Attention with Adaptive Domain Normalization |
| **Parameters** | ~338,000 (338K) trainable parameters |
| **Framework** | PyTorch |
| **Input Shape** | `(batch_size, seq_len=30, n_features=15)` |
| **Output** | Power regression: `(batch_size, pred_horizon=10)` + Signal classification: `(batch_size, 3)` |
| **License** | AGPL-3.0-or-later (code), CC BY 4.0 (Alibaba training data attribution) |
| **Contact** | GitHub Issues or [@VEDKUMAR98](https://x.com/VEDKUMAR98) |

### Architecture Breakdown

```
Input (B, 30, 15)
  → Adaptive Domain Normalization (per-feature-group LayerNorm)
  → Input Projection (Linear: 15 → 128)
  → TCN Backbone (3 dilated causal conv blocks: 32→64→128 channels)
  → Multi-Head Self-Attention (8 heads, 128-dim)
  → Weighted Aggregation (last-step + mean-pool with learned gate)
  → Power Head (MLP: 128→256→128→10)
  → Signal Head (MLP: 138→256→128→3)
```

### Input Features (15 total)

| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 0 | facility_mw | Facility-level power (MW) | Raw |
| 1 | power_roc | Rate of change (MW/s) | Raw |
| 2 | power_roc2 | Second derivative (MW/s²) | Raw |
| 3 | power_roll_mean | Rolling mean (MW) | Raw |
| 4 | power_roll_std | Rolling std deviation | Raw |
| 5 | gpu_avg_power_norm | Avg GPU power / 700W | [0, 1] |
| 6 | gpu_max_power_norm | Max GPU power / 700W | [0, 1] |
| 7 | gpu_avg_temp_norm | Avg temperature / 100°C | [0, 1] |
| 8 | gpu_max_temp_norm | Max temperature / 100°C | [0, 1] |
| 9 | gpu_avg_util_norm | Avg GPU utilization / 100% | [0, 1] |
| 10 | gpu_avg_mem_util_norm | Avg memory utilization / 100% | [0, 1] |
| 11 | cpu_util_norm | CPU utilization / 100% | [0, 1] |
| 12 | hour_sin | Hour of day (sin encoding) | [-1, 1] |
| 13 | hour_cos | Hour of day (cos encoding) | [-1, 1] |
| 14 | is_allreduce | All-Reduce phase indicator | {0, 1} |

### Output Heads

1. **Power Regression** (`power_prediction`): Predicts facility power (MW) for the next 10 timesteps.
2. **Signal Classification** (`signal_logits`): Classifies BESS dispatch signal — `hold` (0), `discharge` (1), `charge` (2).

---

## Intended Use

### Primary Use Case
Predicting short-term power consumption patterns in GPU data centers to enable proactive battery energy storage system (BESS) dispatch and peak demand reduction.

### Intended Users
- Data center operators managing GPU training clusters
- Energy management engineers optimizing BESS dispatch
- Researchers studying GPU workload power characteristics

### Out-of-Scope Uses
- **Not a real-time facility management system** — this is a prediction model, not a closed-loop controller.
- **Not validated on real BESS hardware** — all battery dispatch results are simulation-based.
- **Not for safety-critical applications** — do not use as sole input for grid stability decisions.
- **Not for individual GPU power management** — designed for facility-level (MW) predictions.
- **Not validated beyond single-node scale** — prediction accuracy at 100K+ GPU scale is unverified.

---

## Training Data

### Commercial-Safe Sources (Used for Distributed Weights)

| Source | License | Commercial Safe | Samples | Description |
|--------|---------|----------------|---------|-------------|
| Alibaba GPU Trace v2020 | CC BY 4.0 | ✅ Yes | 6,500 GPUs, 2 months | Production GPU cluster traces from Alibaba PAI |
| Self-collected T4 data | Own | ✅ Yes | Variable | nvidia-smi telemetry from Kaggle/Colab T4 instances |
| Synthetic | N/A | ✅ Yes | 2,500 | Procedurally generated sinusoidal power traces |

### Research-Only Sources (NOT Used for Distributed Weights)

| Source | License | Commercial Safe | Description |
|--------|---------|----------------|-------------|
| York University H100 | CC BY-NC-ND 4.0 | ❌ No | Real H100 node telemetry (20ms resolution), used for research/benchmarking only |

### Data Provenance
- The **distributed demo model** (`best_model_demo.pt`) is trained exclusively on synthetic data.
- The **commercial model** (`commercial_best.pt`) is trained on Alibaba + own data only.
- **No York/MIT NC-licensed data** is included in any distributed checkpoint.

---

## Evaluation Metrics

### Demo Model (Synthetic Data)

| Metric | Value | Notes |
|--------|-------|-------|
| Validation MAPE | 4.85% | On synthetic holdout set |
| Validation Loss (MSE) | ~0.02 | Combined power + signal loss |

### Real-Data Benchmark (York H100 — Research Only)

| Metric | Value | Notes |
|--------|-------|-------|
| Validation MAPE | 1.85% | On 15% holdout of 10,800 sequences |
| Validation Loss (MSE) | ~0.005 | Combined power + signal loss |

> ⚠️ The real-data checkpoint is NOT distributed due to CC BY-NC-ND restrictions.

### Simulation Metrics

| Component | Metric | Value | Validated On |
|-----------|--------|-------|-------------|
| BESS Smoothing | Grid std reduction | 30.0% | Synthetic sinusoidal trace |
| Peak Shaving | Peak load reduction | 10.5% | 24-hour TOU profile |
| Phase Staggering | Volatility reduction | 59.0% | 4 synthetic GPU clusters |
| ONNX Inference | Speedup vs PyTorch | ~10x | CPU (hardware-dependent) |

> ⚠️ Simulation metrics are not validated against real facility-scale hardware.

---

## Ethical Considerations

### Data Privacy
- Alibaba trace data is anonymized production telemetry — no PII involved.
- Self-collected data uses only hardware metrics (power, temperature, utilization).
- No user data, job content, or proprietary algorithms are captured.

### Environmental Impact
- **Positive**: The model's purpose is to reduce energy waste and peak demand in data centers.
- **Training cost**: Training is lightweight (~338K parameters, minutes on a single GPU).
- **Inference cost**: ONNX export enables efficient CPU inference without GPU dependency.

### Fairness & Bias
- The model may not generalize to GPU architectures significantly different from those in training data (H100, T4, V100).
- Power prediction accuracy may vary across different data center topologies and cooling systems.
- The synthetic data component uses simplified sinusoidal patterns that may not capture complex real-world workload dynamics.

### Dual Use
- The model could theoretically be used to optimize power consumption for any large-scale computing facility, not just AI training clusters.
- No known harmful dual-use applications.

---

## Limitations

1. **Scale validation**: Predictions at 200K+ GPU facility scale are mathematical projections from single-node data, not empirically verified.
2. **Hardware dependency**: Trained on H100/T4 power profiles. Accuracy on other GPU architectures (A100, MI300, etc.) is untested.
3. **No real BESS integration**: All battery dispatch results are simulation-based.
4. **No live grid integration**: No connection to real-time grid signals (OpenADR, SCED, utility APIs).
5. **No DCGM pipeline**: Uses CSV-based offline telemetry; live NVIDIA DCGM ingestion is not implemented.
6. **Temporal resolution**: Designed for 1-second to 20ms telemetry resolution. May not work well with coarser-grained data.
7. **Single data center**: Not validated for multi-site or geographically distributed deployments.

---

## Citation

If you use Energivanu in your research or commercial deployment, please cite:

```bibtex
@software{energivanu2026,
  title={Energivanu: Open-source ML toolkit for GPU data center power optimization},
  author={Energivanu Contributors},
  year={2026},
  url={https://github.com/mysterious75/Energivanu},
  license={AGPL-3.0-or-later}
}
```

If you use the Alibaba GPU trace data, you must also cite:

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

## Updates

| Date | Version | Changes |
|------|---------|---------|
| 2026-06-28 | 1.0 | Initial model card |
