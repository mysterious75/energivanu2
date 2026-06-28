# Energivanu Code Review Report

**Reviewer:** Agent-BETA (Code Reviewer)
**Date:** 2026-06-28
**Scope:** Full source code, tests, and architecture review
**Version Reviewed:** 0.1.0

---

## Executive Summary

The Energivanu codebase is a well-structured prototype for ML-based power prediction and battery optimization for AI data centers. The architecture is clean and the code is generally readable. However, there are **3 critical issues**, **14 major issues**, and numerous minor concerns that should be addressed before any production deployment.

**Issue Counts by Severity:**
| Severity | Count |
|----------|-------|
| 🔴 CRITICAL | 3 |
| 🟠 MAJOR | 14 |
| 🟡 MINOR | 11 |
| 💡 SUGGESTION | 8 |

---

## File-by-File Review

---

### 1. `__init__.py`

**Status:** ✅ Clean

- Exports are well-defined and match the public API.
- `__all__` correctly lists all public symbols.
- No unused imports.

| Severity | Issue |
|----------|-------|
| 🟡 MINOR | `__version__` is defined but not exported in `__all__`. Consider adding it. |

---

### 2. `model.py` — ML Architecture

**Status:** 🟠 Multiple major issues

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | 🔴 CRITICAL | **`torch.load` with `weights_only=False`** | `load_model()` uses `torch.load(..., weights_only=False)` which enables arbitrary code execution via pickle deserialization. An attacker who can supply a malicious checkpoint can execute arbitrary Python code. **Fix:** Use `weights_only=True` and store config separately, or use `safetensors`. |
| 2 | 🟠 MAJOR | **`_adaptive_normalize` semantics are confusing and likely wrong** | `BatchNorm1d(seq_len)` is created with `num_features=seq_len`. When applied to input `(B, seq_len, 7)`, it normalizes across the last dim (7 features) for each timestep, learning per-timestep statistics. The docstring claims "per-feature-type normalization" but the implementation does per-timestep normalization across a feature group. This is semantically inverted from what the name suggests. Consider using `nn.LayerNorm` or `nn.GroupNorm` for clearer intent. |
| 3 | 🟠 MAJOR | **Three separate BatchNorm modules share same `seq_len` dimension** | `power_norm`, `telemetry_norm`, and `temporal_norm` all use `BatchNorm1d(seq_len)`. The feature groups have different sizes (7, 7, 3) but the BatchNorm operates on the seq_len dimension. This means all three normalizers learn the same-shaped parameters (seq_len) and the grouping by feature type is effectively meaningless to the normalizer. |
| 4 | 🟡 MINOR | **`_init_weights` resets BatchNorm running statistics** | The method iterates `self.modules()` and reinitializes Conv/Linear weights, but does not explicitly reset BatchNorm. However, since `__init__` calls `_init_weights()` at the end, this only affects the initial state. If called again after training starts, it would destroy learned statistics. |
| 5 | 🟡 MINOR | **No model save method** | `load_model` exists but there's no corresponding `save_model` utility. |
| 6 | 💡 SUGGESTION | **Add `torch.no_grad()` context in `load_model`** | After loading, the model is set to `eval()` but gradients are still enabled. Wrap in `torch.no_grad()` for memory efficiency. |

---

### 3. `mpc.py` — MPC Controller

**Status:** 🟠 Several major issues

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | 🟠 MAJOR | **Brute-force optimization is computationally expensive** | `optimize()` evaluates: 5 proportional gains + 11 constant trajectories + 49 two-phase trajectories + swing heuristic = ~66 trajectory simulations per call. For real-time control at 5-second intervals, this may be too slow on CPU. Consider scipy.optimize or CMA-ES. |
| 2 | 🟠 MAJOR | **Frequency deviation formula is physically incorrect** | `df = self.f0 * dP / (2 * self.H)` computes frequency deviation where `dP` is in MW. The standard swing equation requires `dP` in per-unit (or divided by nominal power). With `dP` in MW and `f0=60`, this produces Hz values that are off by a factor of ~200-1000x. The formula should be `df = f0 * dP / (2 * H * P_nominal)` where P_nominal is in MW. |
| 3 | 🟠 MAJOR | **`simulate_with_staggering` uses `np.roll` with wrap-around** | `np.roll(cluster_a_power, stagger_steps)` wraps data from the end to the beginning, creating artificial smoothness at the boundary. This overstates the staggering benefit. Use zero-padding or truncation instead. |
| 4 | 🟡 MINOR | **Hard-coded swing cycle positions** | Swing strategy uses magic numbers `pos == 8`, `pos == 9`, `pos == 0` with `cyc = 10`. These should be configurable or derived from the actual workload pattern. |
| 5 | 🟡 MINOR | **`_forecast` uses simplistic linear extrapolation** | With only 20 data points and a linear fit, the forecast is noisy. Consider exponential smoothing or a simple AR model. |
| 6 | 💡 SUGGESTION | **Consider scipy.optimize.minimize for trajectory optimization** | Replace grid search with gradient-free optimizer (Nelder-Mead or Powell) for better results with fewer evaluations. |

---

### 4. `optimizer.py` — Peak Shaving

**Status:** 🟠 Logic issues

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | 🟠 MAJOR | **`simulate_month` passes hourly data but optimizer assumes 5-second steps** | `update_15min_average()` is called once per hour (not per 5-second step), so the 15-minute rolling window never fills properly. With hourly data, `window_steps = 900/5 = 180` but only 1 call per hour means the window always has 1 element after the first hour. The 15-minute demand charge logic is effectively broken. |
| 2 | 🟠 MAJOR | **Dual SOC state is confusing and error-prone** | `self.soc` is maintained as instance state, but `optimize()` takes `battery_soc` as a parameter and updates `self.soc` based on it. In `simulate_month`, `self.soc` is passed as `battery_soc`, which works but creates two sources of truth. If someone calls `optimize()` with a different `battery_soc` value, `self.soc` diverges silently. |
| 3 | 🟡 MINOR | **`estimate_annual_savings` uses rough heuristic** | Assumes `E_max * 0.5` MWh daily arbitrage regardless of actual TOU patterns or battery degradation. |
| 4 | 💡 SUGGESTION | **Add battery degradation model** | Peak shaving optimization should account for cycle life degradation to avoid pyrrhic savings. |

---

### 5. `scheduler.py` — Phase Staggering

**Status:** 🟠 Non-deterministic

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | 🟠 MAJOR | **Non-deterministic optimization due to random noise** | `generate_cluster_trace()` adds `np.random.normal(0, noise_std)` on every call. `find_optimal_offset()` calls this multiple times, so the same offset gets different costs each run. The "optimal" offset is random. **Fix:** Use a fixed seed or pre-generate traces. |
| 2 | 🟠 MAJOR | **`schedule_clusters` generates different noise than `find_optimal_offset`** | The optimal offset was found with one set of random traces, but `schedule_clusters` generates entirely new random traces. The offset may not be optimal for the new traces. |
| 3 | 🟡 MINOR | **`estimate_bess_burden_reduction` uses hard-coded lookup table** | `{2: 35.0, 3: 50.0, ...}` is inflexible and doesn't account for actual workload patterns. |
| 4 | 💡 SUGGESTION | **Use deterministic noise or seed** | Add a `seed` parameter to `generate_cluster_trace()` for reproducibility. |

---

### 6. `data.py` — Data Pipeline

**Status:** 🟠 Missing robustness

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | 🟠 MAJOR | **No error handling for missing columns** | `load_node_data()` assumes columns like `gpu0_power_W` through `gpu7_power_W` exist. If the CSV has a different GPU count or naming, it crashes with a KeyError. Add column validation. |
| 2 | 🟠 MAJOR | **Non-reproducible train/val split** | `build_dataloaders()` uses `np.random.permutation()` without a seed. Every run produces a different split, making experiment comparison impossible. |
| 3 | 🟡 MINOR | **`warnings.filterwarnings("ignore")` at module level** | Suppresses ALL warnings including potentially important ones (e.g., NaN detection, deprecation). Use specific warning categories. |
| 4 | 🟡 MINOR | **Unexplained magic constant** | `cpu_power_estimated_W = df["cpu_utilization_percent"] * 2.5` — the 2.5W per percent utilization is undocumented. |
| 5 | 🟡 MINOR | **Signal labeling uses fixed threshold** | `signals[power_change > 0.5] = 1` — the 0.5 MW threshold is hard-coded and may not be appropriate for all facility scales. |
| 6 | 💡 SUGGESTION | **Add data validation/quality checks** | Check for NaN, negative power values, unrealistic temperatures, etc. before training. |

---

### 7. `api.py` — REST API

**Status:** 🔴 Security and robustness issues

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | 🔴 CRITICAL | **No authentication or authorization** | All endpoints are publicly accessible, including `/optimize/battery` which controls battery dispatch. In a real deployment, this could cause physical damage to battery systems. **Fix:** Add API key auth at minimum, OAuth2/JWT for production. |
| 2 | 🟠 MAJOR | **No input validation** | `PredictRequest.power_trace` accepts empty lists, NaN, Inf, negative values, or arbitrarily large inputs. No length limits. A 10-million-element list could OOM the server. |
| 3 | 🟠 MAJOR | **Dead code in predict endpoint** | After `x = torch.tensor(trace).unsqueeze(0).unsqueeze(-1)`, the tensor is always 3D (shape `(1, seq_len, 1)`). The check `if x.dim() == 2` is dead code and never executes. Additionally, the model expects `(B, seq_len, n_features)` where `n_features=15-17`, but this endpoint only passes 1 feature. The prediction will be wrong or crash. |
| 4 | 🟠 MAJOR | **No error handling** | Any exception returns a bare 500 error. No try/except blocks, no structured error responses, no logging. |
| 5 | 🟠 MAJOR | **New MPCController per request** | `/optimize/battery` creates a fresh `MPCController()` on every request, losing all state (SOC, history, step count). This means the controller never learns from previous calls. |
| 6 | 🟡 MINOR | **Global `_model` not thread-safe** | If the API serves concurrent requests, the shared model state could cause issues (though `model.eval()` with `torch.no_grad()` is generally safe for inference). |
| 7 | 💡 SUGGESTION | **Add request/response logging** | For debugging and audit trails, especially for battery control endpoints. |
| 8 | 💡 SUGGESTION | **Add rate limiting** | Prevent abuse of compute-intensive prediction endpoints. |

---

### 8. `cli.py` — CLI Commands

**Status:** 🟡 Functional but incomplete

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | 🟡 MINOR | **`cmd_predict` doesn't load a trained model** | Creates an untrained `EnergivanuPEB()` and prints parameter count. Misleading for users expecting actual predictions. |
| 2 | 🟡 MINOR | **No error handling for missing dependencies** | `cmd_serve` catches `ImportError` for uvicorn but other commands don't handle missing torch/numpy. |
| 3 | 💡 SUGGESTION | **Add `--config` flag** | Allow users to pass custom config files instead of using defaults. |

---

### 9. `train_demo.py` — Demo Training

**Status:** 🟡 Functional with issues

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | 🟠 MAJOR | **Non-reproducible synthetic data** | `SyntheticDataset.__init__` generates random data without a seed. Each instantiation produces different data, making results non-reproducible across runs. |
| 2 | 🟡 MINOR | **No early stopping** | Trains for all 10 epochs regardless of validation loss trend. |
| 3 | 🟡 MINOR | **Checkpoint path inconsistency** | Uses `../../models/checkpoints` (relative to `src/energivanu/`), placing checkpoints at `<project_root>/models/checkpoints/`. This is fine but differs from `train_real.py`. |
| 4 | 💡 SUGGESTION | **Add TensorBoard/WandB logging** | For better training visualization. |

---

### 10. `train_real.py` — Real Data Training

**Status:** 🟠 Missing critical features

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | 🟠 MAJOR | **Checkpoint path inconsistency with train_demo** | Uses `../models/checkpoints` (relative to `src/energivanu/`), placing checkpoints at `src/models/checkpoints/`. This differs from `train_demo.py` which uses `../../models/checkpoints`. Users will be confused about where models are saved. |
| 2 | 🟠 MAJOR | **No early stopping** | Trains for all 100 epochs regardless of validation performance. With a learning rate of 1e-3 and no early stopping, the model may overfit significantly. |
| 3 | 🟡 MINOR | **No mixed precision training** | For large-scale training on H100 data, FP16/BF16 would significantly speed up training. |
| 4 | 🟡 MINOR | **No gradient accumulation** | With batch_size=64 and potentially large datasets, gradient accumulation would allow effective larger batch sizes. |
| 5 | 💡 SUGGESTION | **Add checkpoint resume capability** | If training is interrupted, there's no way to resume from the last checkpoint. |

---

## Test Review

---

### `test_model.py`

| # | Issue | Details |
|---|-------|---------|
| 1 | **Missing: `load_model` test** | No test for the checkpoint loading function, which has the critical `weights_only=False` security issue. |
| 2 | **Missing: backward pass test** | No test verifying gradients flow correctly through the model. |
| 3 | **Missing: different configurations** | All tests use `n_features=15, seq_len=30, pred_horizon=10`. No tests for other configurations. |
| 4 | **Missing: edge cases** | No tests for `seq_len=1`, `n_features=1`, `pred_horizon=1`. |
| 5 | **Quality: good shape coverage** | Tests cover batch sizes, single sample, and determinism. |

### `test_mpc.py`

| # | Issue | Details |
|---|-------|---------|
| 1 | **Missing: `simulate` metrics validation** | Tests that `simulate` returns metrics but doesn't validate the values are reasonable. |
| 2 | **Missing: `simulate_with_staggering`** | No test for the staggering comparison method. |
| 3 | **Missing: custom config** | Only tests default config. |
| 4 | **Missing: edge cases** | No test for empty history, single-step trace, or SOC at boundaries. |
| 5 | **Missing: frequency deviation validation** | The physically incorrect formula is never tested for correctness. |

### `test_data.py`

| # | Issue | Details |
|---|-------|---------|
| 1 | **Missing: `load_node_data`** | No test (requires actual CSV files, but could use a fixture). |
| 2 | **Missing: `RealH100Dataset`** | No test for the PyTorch Dataset class. |
| 3 | **Missing: `build_dataloaders`** | No integration test for the full pipeline. |
| 4 | **Missing: edge cases** | No test for empty dataframe, single row, or missing columns. |
| 5 | **Quality: good for what it covers** | The existing tests properly validate shapes and scaling. |

### `test_onnx.py`

| # | Issue | Details |
|---|-------|---------|
| 1 | **Good: comprehensive ONNX validation** | Tests shapes, numerical accuracy, batch sizes, speed, determinism, and metadata. |
| 2 | **Issue: depends on external files** | All tests skip if checkpoint/ONNX files are missing, which is likely in CI. |
| 3 | **Missing: ONNX export test** | No test for the export process itself (PyTorch → ONNX conversion). |

---

## Architecture Review

### Module Structure ✅

The module structure is clean and well-organized:

```
src/energivanu/
├── __init__.py      # Public API
├── model.py         # ML architecture
├── mpc.py           # Battery controller
├── optimizer.py     # Peak shaving
├── scheduler.py     # Phase staggering
├── data.py          # Data pipeline
├── api.py           # REST API
├── cli.py           # CLI
├── train_demo.py    # Demo training
└── train_real.py    # Real training
```

Separation of concerns is good. Each module has a clear responsibility.

### Dependency Management ✅

- `pyproject.toml` is well-configured with optional dependencies.
- Core deps (torch, numpy, pandas, scikit-learn) are reasonable.
- API deps (fastapi, uvicorn) are properly optional.

### API Design 🟠

- Endpoints follow REST conventions (GET/POST, resource-based URLs).
- **Missing:** API versioning (`/v1/predict`), proper error responses, pagination for batch operations.
- **Missing:** OpenAPI schema customization (descriptions, examples).
- **Issue:** The predict endpoint doesn't actually work correctly (passes 1 feature instead of 15-17).

### Scalability 🟠

- Single-process uvicorn server with no worker configuration.
- No caching layer for repeated predictions.
- No async support (all endpoints are synchronous).
- No connection pooling or state management for MPC controllers.

### Code Quality ✅

- Consistent style (follows ruff configuration).
- Good docstrings on public classes and functions.
- Type hints on most function signatures.
- AGPL-3.0 license headers on all files.

---

## Priority Fix List

### P0 — Fix Immediately (Critical)

1. **`model.py`: Replace `weights_only=False` with `weights_only=True`** in `load_model()`. Store model config in a separate JSON file or use `safetensors` format. This is a remote code execution vulnerability.

2. **`api.py`: Add authentication** to all endpoints, especially `/optimize/battery`. At minimum, require an API key header. For production, use OAuth2/JWT.

3. **`api.py`: Fix the predict endpoint** — it passes 1 feature instead of the expected 15-17. The tensor construction logic is broken and produces wrong results.

### P1 — Fix Before Any Deployment (Major)

4. **`mpc.py`: Fix frequency deviation formula** — divide by nominal power to get per-unit values.
5. **`scheduler.py`: Make optimization deterministic** — use fixed seeds or pre-generated traces.
6. **`optimizer.py`: Fix 15-minute window logic** for hourly data in `simulate_month`.
7. **`api.py`: Add input validation** — length limits, NaN checks, type validation.
8. **`api.py`: Add error handling** — try/except blocks with structured error responses.
9. **`api.py`: Persist MPC controller state** across requests.
10. **`data.py`: Add column validation** in `load_node_data()`.
11. **`data.py`: Set random seed** in `build_dataloaders()` for reproducibility.
12. **`model.py`: Fix `_adaptive_normalize` semantics** — use LayerNorm or GroupNorm for clearer intent.
13. **`train_real.py`: Add early stopping** to prevent overfitting.
14. **`train_real.py`: Fix checkpoint path** to be consistent with `train_demo.py`.
15. **`mpc.py`: Fix `simulate_with_staggering`** — don't use `np.roll` with wrap-around.
16. **`train_demo.py`: Set random seed** in `SyntheticDataset` for reproducibility.
17. **`data.py`: Remove module-level `warnings.filterwarnings("ignore")`**.

### P2 — Improve Before Production (Minor)

18. Add `save_model()` utility function.
19. Add mixed precision training support.
20. Add early stopping to `train_demo.py`.
21. Add comprehensive test coverage (see test gaps above).
22. Add API versioning and rate limiting.
23. Add request/response logging.
24. Document magic constants (CPU power estimate, signal thresholds).
25. Add async support to API endpoints.

---

## Conclusion

The Energivanu codebase is a solid prototype with clean architecture and good separation of concerns. The ML model design (TCN + Attention) is sound, and the battery optimization logic covers the right use cases.

However, **3 critical security/correctness issues** and **14 major issues** prevent this from being production-ready. The most urgent fixes are:

1. The `torch.load` security vulnerability (arbitrary code execution)
2. The missing API authentication (battery control exposed to the internet)
3. The broken predict endpoint (wrong feature count)

After addressing the P0 and P1 items, this codebase would be suitable for a controlled pilot deployment. The P2 items should be addressed before full production rollout.

**Overall Assessment:** Prototype-quality with good bones. Needs security hardening, correctness fixes, and test coverage before production use.
