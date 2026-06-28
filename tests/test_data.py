import numpy as np
import pandas as pd

from energivanu.data import create_sequences, scale_to_facility


def _make_fake_df(n_rows=500):
    df = pd.DataFrame()
    for i in range(8):
        df[f"gpu{i}_power_W"] = np.random.uniform(200, 550, n_rows)
        df[f"gpu{i}_temp_C"] = np.random.uniform(35, 65, n_rows)
        df[f"gpu{i}_utilization_percent"] = np.random.uniform(0, 100, n_rows)
        df[f"gpu{i}_mem_utilization"] = np.random.uniform(0, 100, n_rows)
    df["cpu_utilization_percent"] = np.random.uniform(0, 100, n_rows)
    df["timestamp"] = pd.date_range("2024-01-01", periods=n_rows, freq="20ms")

    gpu_power_cols = [f"gpu{i}_power_W" for i in range(8)]
    gpu_temp_cols = [f"gpu{i}_temp_C" for i in range(8)]
    gpu_util_cols = [f"gpu{i}_utilization_percent" for i in range(8)]
    gpu_mem_cols = [f"gpu{i}_mem_utilization" for i in range(8)]

    df["node_power_W"] = df[gpu_power_cols].sum(axis=1)
    df["gpu_avg_power"] = df[gpu_power_cols].mean(axis=1)
    df["gpu_max_power"] = df[gpu_power_cols].max(axis=1)
    df["gpu_avg_temp"] = df[gpu_temp_cols].mean(axis=1)
    df["gpu_max_temp"] = df[gpu_temp_cols].max(axis=1)
    df["gpu_avg_util"] = df[gpu_util_cols].mean(axis=1)
    df["gpu_avg_mem_util"] = df[gpu_mem_cols].mean(axis=1)

    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["power_roc"] = df["node_power_W"].diff().fillna(0)
    df["power_roc2"] = df["power_roc"].diff().fillna(0)
    window = 250
    df["power_roll_mean"] = df["node_power_W"].rolling(window, min_periods=1).mean()
    df["power_roll_std"] = df["node_power_W"].rolling(window, min_periods=1).std().fillna(0)
    df["is_allreduce"] = ((df["gpu_avg_util"] > 80) & (df["gpu_avg_mem_util"] < 30)).astype(float)
    return df


def test_create_sequences_shapes():
    df = _make_fake_df(500)
    X, Yp, Ys = create_sequences(df, seq_len=30, pred_horizon=10, stride=50)
    assert X.ndim == 3
    assert X.shape[1] == 30
    assert X.shape[2] == 15
    assert Yp.ndim == 2
    assert Yp.shape[1] == 10
    assert Ys.ndim == 1


def test_scale_to_facility():
    node_W = np.array([4000.0, 5000.0, 6000.0])
    facility_MW = scale_to_facility(node_W, num_gpus=200000, gpus_per_node=8)
    expected_nodes = 200000 / 8
    expected = node_W * expected_nodes / 1e6
    np.testing.assert_allclose(facility_MW, expected)


def test_create_sequences_power_range():
    df = _make_fake_df(1000)
    X, Yp, Ys = create_sequences(df, seq_len=30, pred_horizon=10, stride=50, num_gpus=200000)
    assert Yp.min() > 0
    assert Yp.max() < 1000
