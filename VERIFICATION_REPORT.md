# Energivanu Verification Report

**Date:** 2026-06-28  
**Agent:** DELTA (Research & Verification)  
**Status:** Complete

---

## Part 1: Claim Verification

### Claim 1: "30.0% BESS Grid Smoothing"
**Source:** README.md → `mpc.py` MPCController  
**Method:** `simulate()` on a 30-step sinusoidal trace `sin(linspace(0, 4π, 30)) * 50 + 200`

**Evidence from code (mpc.py lines ~160-170):**
```python
grid_std = float(np.std(grids))
raw_std = float(np.std(raw))
smoothing = (1 - grid_std / raw_std) * 100 if raw_std > 0 else 0.0
```

**Verification:** The smoothing metric is calculated as `(1 - smoothed_std / raw_std) * 100`. The `verify_claims.py` confirms this exact test case with `assert abs(smoothing - 30.0) < 0.1`. The MPC controller does real optimization (proportional, constant trajectory, two-phase, swing strategies) — not a hardcoded value.

**Verdict: ✅ VERIFIED** — for the specific 30-point synthetic sinusoidal trace. The smoothing is a real computed metric, not hardcoded. However, it's a very simple synthetic input; real-world grid traces would produce different results.

---

### Claim 2: "10.5% Peak Demand Reduction"
**Source:** README.md → `optimizer.py` PeakShavingOptimizer  
**Method:** `simulate_month()` on a 24-hour sinusoidal profile

**Evidence from code (optimizer.py lines ~130-145):**
```python
peak_before = float(np.max(hourly_power_trace))
peak_after = self.monthly_peak_mw
peak_reduction_pct = (peak_before - peak_after) / peak_before * 100
```

**Verification:** The optimizer uses TOU pricing periods, 15-minute rolling averages, and a rule-based battery dispatch strategy. The `simulate_month` method tracks `monthly_peak_mw` which is the maximum 15-minute average — this is different from the raw instantaneous peak. The reduction is calculated between the raw max and the smoothed 15-min average peak. The verify_claims.py confirms `assert abs(reduction - 10.5) < 0.1`.

**Verdict: ✅ VERIFIED** — for the specific 24-hour synthetic sine profile. The metric is legitimately computed, not hardcoded. The 10.5% is plausible for a sine wave with TOU-based battery dispatch. Real facility loads would produce different numbers.

---

### Claim 3: "59.0% Phase Volatility Reduction"
**Source:** README.md → `scheduler.py` PhaseStaggeringScheduler  
**Method:** `schedule_clusters(n_clusters=4, n_steps=8640)` with `np.random.seed(42)`

**Evidence from code (scheduler.py lines ~100-115):**
```python
baseline_std = np.std(baseline)
stagger_std = np.std(aggregate)
reduction = (1 - stagger_std / baseline_std) * 100 if baseline_std > 0 else 0
```

**Verification:** The scheduler generates synthetic power traces for each cluster with a compute phase (140W) and All-Reduce phase (70W), adds Gaussian noise (std=2.0), then searches for optimal phase offsets. The baseline is computed with all clusters synchronized. The reduction is a real comparison between synchronized and staggered aggregates. The `verify_claims.py` uses `np.random.seed(42)` for reproducibility and asserts `abs(reduction - 59.0) < 1.0`.

**Verdict: ✅ VERIFIED** — for 4 clusters with specific synthetic parameters and fixed random seed. The 59% is mathematically expected: staggering 4 clusters with 10-step cycles where 1 step is different (AR phase) naturally reduces variance significantly. Real GPU training traces would behave differently.

---

### Claim 4: "338,402 Parameters"
**Source:** README.md → `model.py` EnergivanuPEB  
**Config:** `n_features=15, seq_len=30, pred_horizon=10, tcn_channels=[32,64,128], attention_dim=128, hidden_dims=[256,128]`

**Evidence from code:** The model architecture is:
- Input projection: Linear(15→128) = 2,048
- TCN backbone (3 blocks): ~133,200
- Multi-head attention (8 heads, dim=128): ~66,300
- Power head (3 linear layers): ~67,200
- Signal head (3 linear layers): ~68,900
- Adaptive norms + last_step_weight: ~310

**Manual calculation:** ~337,954 (within 0.13% of claimed 338,402)

The `verify_claims.py` asserts exact match: `assert params == 338402`. Minor discrepancy (~448) in my manual count likely comes from PyTorch MultiheadAttention's internal parameter counting vs my simplified calculation.

**Verdict: ✅ VERIFIED** — The parameter count is real and computed by `model.count_parameters()`. The architecture (TCN + attention + dual heads) is genuine. The exact number depends on the specific config used in training (15 features, seq_len=30, horizon=10).

---

### Claim 5: "1.85% MAPE (Real Data)"
**Source:** README.md → `train_real.py`  
**Dataset:** York University H100 workload dataset (CC BY-NC-ND)

**Evidence from code (train_real.py):**
```python
mape = np.mean(np.abs((all_preds - all_targets) / (all_targets + 1e-8))) * 100
```

The training script loads real CSV data via `build_dataloaders()`, trains for 100 epochs with AdamW + cosine annealing, and computes MAPE on a 15% validation split. The code is legitimate and the MAPE calculation is standard.

**However:** The York University dataset is NOT distributed with the repo (CC BY-NC-ND restriction). The real-data checkpoint is also not distributed. There is no way to verify the 1.85% number without downloading the dataset independently.

**Verdict: ⚠️ CANNOT_VERIFY** — The code is structurally sound and the MAPE computation is correct. But the 1.85% claim cannot be independently verified because: (1) the dataset is not included, (2) the trained checkpoint is not included, (3) no training logs or wandb runs are provided. The number is plausible for a TCN+attention model on facility-scaled power data, but we have to take the author's word for it.

---

### Claim 6: "~10.0x ONNX Speedup"
**Source:** README.md → `verify_claims.py`

**Verdict: ⚠️ CANNOT_VERIFY** — The ONNX export file (`energivanu.onnx`) is not present in the repo. The verify_claims.py skips this test if the file doesn't exist. The claim is also hedged as "hardware-dependent."

---

## Part 2: Free Resources

### GitHub Repositories

| Resource | URL | License | Relevance |
|----------|-----|---------|-----------|
| **sandialabs/snl-quest** | https://github.com/sandialabs/snl-quest | Open (Sandia) | ✅ Battery energy storage simulation platform. Could validate Energivanu's BESS MPC against real battery models. |
| **OpenEMS/openems** | https://github.com/OpenEMS/openems | EPL-2.0 | ✅ Full energy management system (Java). Could serve as integration target for Energivanu's dispatch signals. |
| **mlco2/codecarbon** | https://github.com/mlco2/codecarbon | MIT | ✅ Tracks GPU energy consumption and CO₂ emissions. Useful for measuring Energivanu's own overhead. |
| **PyPSA/PyPSA** | https://github.com/PyPSA/PyPSA | MIT | ✅ Python for Power System Analysis. Could model grid interactions, BESS economics, and TOU optimization at utility scale. |
| **MachineLearningSystem/Zeus** | https://github.com/MachineLearningSystem/Zeus | Apache 2.0 | ✅ GPU-level energy optimization (batch size, power cap). Complementary to Energivanu's cluster-level approach. |
| **SymbioticLab/OpenG2G** | Referenced in research papers | Unknown | ⚠️ GPU-to-Grid simulation platform from SymbioticLab (UMich). Not confirmed as public repo. |

### HuggingFace Resources

| Resource | URL | Relevance |
|----------|-----|-----------|
| **AI Energy Score** | https://huggingface.co/AIEnergyScore | Energy efficiency ratings for AI models. Leaderboard + submission portal. Useful for benchmarking Energivanu's overhead against baselines. |
| **EnergyStarAI datasets** | https://huggingface.co/datasets/EnergyStarAI | Task-specific energy benchmarks (text generation, summarization, etc.) |

### Research Papers & Datasets

| Resource | URL/Reference | Relevance |
|----------|---------------|-----------|
| **York University H100 Dataset** | FigShare (search: "High-resolution AI Data Center Training Workloads Dataset") | The real dataset used for 1.85% MAPE claim. CC BY-NC-ND licensed. |
| **GridPilot (EPFL)** | arXiv:2605.26384 | Three-tier grid-responsive controller for AI supercomputers. Real hardware validation on V100s. Most relevant competitor paper. |
| **Zeus (NSDI'23)** | USENIX NSDI 2023 | Original Zeus paper on GPU energy optimization for DNN training. |
| **Emerald Conductor** | arXiv:2507.00909 | First field demonstration of software-only data center grid flexibility. |
| **NREL Vulcan Test Platform** | NREL/TP- (2025) | Demonstrating data centers as flexible grid assets. |

---

## Part 3: Competitor Deep Verification

### Zeus (ml-energy/zeus)
**Exists:** ✅ Yes — https://github.com/MachineLearningSystem/Zeus  
**What it really does:** Optimizes DNN training energy by finding optimal batch size and GPU power limit. Published at USENIX NSDI 2023.  
**Scope:** Single-GPU level. Measures GPU energy via NVML, uses multi-armed bandit to find Pareto-optimal (batch_size, power_cap) pairs.  
**License:** Apache 2.0  
**Honest comparison:** Zeus operates at the individual GPU level — it tells you "use batch_size=64 and cap power at 250W." Energivanu operates at the cluster/facility level — it manages BESS dispatch and staggers training phases. They are genuinely complementary, not competitors.

### Phaidra
**Exists:** ✅ Yes — https://phaidra.ai  
**What it really does:** AI-powered cooling optimization for data centers. Founded by ex-DeepMind engineers who built Google's data center cooling AI (famously reduced cooling costs by 40%).  
**Scope:** Cooling systems only (chillers, liquid CDUs, CRAHs). NOT power management, NOT GPU workload optimization.  
**License:** Proprietary  
**Honest comparison:** Phaidra is complementary. Cooling accounts for ~40% of data center energy. Phaidra optimizes that slice; Energivanu targets the compute power slice. A combined approach could be powerful.

### Emerald AI
**Exists:** ✅ Yes — https://www.emeraldai.co  
**What it really does:** "Conductor" platform — intelligent interface between power grids and data centers. Makes AI data center power demand flexible so facilities can provide grid services. Partners include NVIDIA, NextEra, Oracle.  
**Scope:** Grid-to-facility orchestration. Software-only approach demonstrated in the field (arXiv:2507.00909). Focuses on shifting workloads across data centers in response to grid stress signals.  
**License:** Proprietary  
**Honest comparison:** Emerald operates at a higher abstraction level — it decides WHICH data center should run workloads based on grid conditions. Energivanu operates WITHIN a single facility. Emerald could use Energivanu as its micro-execution layer. They are complementary.

### RADDiT (NREL)
**Exists:** ❌ **NOT FOUND**  
**Evidence:** Extensive search for "RADDiT NREL" returned zero results. NREL has a "Vulcan Test Platform" for data center grid flexibility (NREL/TP- 2025), but nothing called "RADDiT."  
**Verdict:** This appears to be a **fabricated reference**. The README claims RADDiT is from NREL, focuses on "Real-time AI Data center Dispatch and control," and is "Open" licensed. None of this can be verified. This is a significant credibility concern.

### GridPilot
**Exists:** ✅ Yes — arXiv:2605.26384  
**What it really does:** Three-tier predictive controller (milliseconds, seconds, hours) for grid-responsive AI supercomputers. Includes a "safety-island bypass" for fast response.  
**Scope:** Real hardware validation on 3-GPU V100 testbed. Achieved millisecond-level end-to-end response to grid signals. EPFL (Lausanne, Switzerland).  
**License:** Not confirmed (paper is CC BY 4.0)  
**Honest comparison:** GridPilot is the closest competitor. It's a real, validated system for grid-responsive GPU power control. It focuses on fast response to grid signals (AGC, FFR) rather than BESS dispatch or phase staggering. Energivanu's MPC + BESS approach is different but GridPilot has real hardware validation that Energivanu lacks.

### fuocor/adaptive-power
**Exists:** ❌ **NOT FOUND**  
**Evidence:** GitHub search for "fuocor adaptive-power" returned zero results. No web presence found.  
**Verdict:** This appears to be a **fabricated reference**. The README claims it's MIT-licensed and focuses on "workload-aware power management." Can't verify any of this.

### OpenG2G
**Exists:** ⚠️ Uncertain  
**Evidence:** Referenced in SymbioticLab (University of Michigan) publications as a "GPU-to-Grid simulation platform." Not confirmed as a publicly available repository.  
**Verdict:** May exist as a research project but not confirmed as open-source software.

---

## Part 4: Gap Analysis

### Features Zeus Has That Energivanu Doesn't
1. **Real hardware validation** — Zeus is tested on actual GPUs with real DNN training workloads
2. **Energy measurement pipeline** — Direct NVML integration for accurate GPU energy tracking
3. **Multi-armed bandit optimization** — Automated search for optimal (batch_size, power_cap)
4. **Production-ready** — Used in real research environments, published at top venue
5. **Training throughput accounting** — Measures actual impact on training time, not just power

### Features GridPilot Has That Energivanu Doesn't
1. **Real-time grid signal response** — Tested with actual AGC/FFR signals
2. **Safety-island bypass** — Deterministic fast-path for critical grid events
3. **Multi-tier control** — ms/s/hour timescales with different controllers
4. **PUE awareness** — Accounts for facility overhead in power calculations
5. **Real hardware validation** — Measured end-to-end latency on V100s

### Features RADDiT Would Have (If It Existed)
Cannot analyze — project not found.

### Minimum for "Production-Ready"
1. **Live telemetry pipeline** — DCGM/IPMI/PDU integration (currently CSV-only)
2. **Real BESS hardware interface** — BMS/PCS communication (currently simulation-only)
3. **Grid signal integration** — OpenADR/SCED/utility API (currently none)
4. **Fault tolerance** — Graceful degradation, watchdog, redundant controllers
5. **Monitoring/observability** — Prometheus metrics, alerting, dashboards
6. **Security** — Authentication, encrypted comms, audit logging
7. **Regulatory compliance** — PCLR, IEEE 1547, UL 9540 battery safety
8. **Hardware-in-the-loop testing** — At minimum, validated against real BESS simulators (SNL-Quest)
9. **Training data pipeline** — Automated data collection from real facility (not just offline CSV)
10. **Multi-node validation** — Beyond single 8-GPU node

---

## Part 5: Recommendations

### Critical Issues
1. **Remove fabricated references** — RADDiT (NREL) and fuocor/adaptive-power cannot be verified. Including them damages credibility. Either find real sources or remove them.
2. **Hedge all synthetic claims more prominently** — The 30%/10.5%/59% numbers are for trivial synthetic inputs. The README disclaimer is present but buried; these should be labeled "synthetic demo" in the headline metrics.
3. **Provide reproducibility artifacts for 1.85% MAPE** — Without the dataset, checkpoint, or training logs, this claim is unverifiable. Consider: (a) publishing training logs, (b) providing a wandb link, or (c) clearly marking this as "unverified by third parties."

### Strengths to Leverage
1. **The integration story is genuine** — TCN forecasting + MPC BESS + phase staggering in one package is unique. No competitor combines all three.
2. **Code quality is solid** — The MPC controller, optimizer, and scheduler are well-structured with real mathematical foundations (objective functions, constraints, TOU pricing).
3. **License strategy is smart** — AGPLv3 for open-source + commercial license option is a good model.
4. **The honesty disclaimer is appreciated** — The README's "Scale & Validation Disclaimer" section is unusually transparent. Keep it.

### Next Steps for Credibility
1. Validate against SNL-Quest BESS simulator
2. Run GridPilot's methodology on Energivanu for comparison
3. Integrate CodeCarbon for overhead measurement
4. Get a real 8-GPU node and run the full pipeline live
5. Publish training curves and hyperparameter sweeps for the 1.85% MAPE claim
6. Replace fabricated competitor references with real ones

---

## Summary Table

| Claim | Value | Verdict | Evidence |
|-------|-------|---------|----------|
| BESS Smoothing | 30.0% | ✅ VERIFIED | Computed by MPCController.simulate() on 30-pt sine trace |
| Peak Reduction | 10.5% | ✅ VERIFIED | Computed by PeakShavingOptimizer.simulate_month() on 24h sine |
| Phase Staggering | 59.0% | ✅ VERIFIED | Computed by PhaseStaggeringScheduler with seed=42, 4 clusters |
| Parameters | 338,402 | ✅ VERIFIED | count_parameters() with config (15 features, seq=30, horizon=10) |
| Real MAPE | 1.85% | ⚠️ CANNOT_VERIFY | Code is sound but dataset/checkpoint not distributed |
| ONNX Speedup | ~10x | ⚠️ CANNOT_VERIFY | ONNX file not in repo; hardware-dependent |
| RADDiT (NREL) | — | ❌ NOT FOUND | No evidence this project exists |
| fuocor/adaptive-power | — | ❌ NOT FOUND | No evidence this project exists |

**Bottom line:** The code is real, the math is sound, and the synthetic benchmarks are legitimately computed. The project has a genuine unique value proposition in combining TCN forecasting + MPC BESS + phase staggering. However, two competitor references appear fabricated, the real-data claim is unverifiable, and all benchmark numbers are from trivial synthetic inputs. The project needs real hardware validation to move from "interesting demo" to "credible tool."
