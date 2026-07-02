"""Tests for the GPU telemetry collector."""
import time

import numpy as np

from energivanu.telemetry.nvidia_smi_collector import (
    NvidiaSmiCollector, GpuSample, AggregatedSample, parse_nvidia_smi_xml,
)


def test_gpu_sample_defaults():
    sample = GpuSample(gpu_id=0, power_w=250.0, temp_c=65.0,
                      util_pct=80.0, mem_util_pct=50.0,
                      sm_clock_mhz=1980.0, mem_clock_mhz=2619.0)
    assert sample.gpu_id == 0
    assert sample.timestamp != ""


def test_aggregated_sample():
    sample = AggregatedSample(
        timestamp="test",
        unix_ts=0.0,
        node_power_w=4000.0,
        gpu_avg_power_w=500.0,
        gpu_max_power_w=550.0,
        gpu_std_power_w=20.0,
        gpu_avg_temp_c=65.0,
        gpu_max_temp_c=70.0,
        gpu_avg_util_pct=80.0,
        gpu_avg_mem_util_pct=50.0,
    )
    assert sample.node_power_w == 4000.0


def test_collector_simulation_mode():
    collector = NvidiaSmiCollector(simulation_mode=True, simulation_num_gpus=8)
    assert collector.simulation_mode is True
    assert collector._sim_source is not None


def test_collector_start_stop():
    collector = NvidiaSmiCollector(
        simulation_mode=True, simulation_num_gpus=4,
        collection_interval_s=0.1,
        rolling_window_size=100,
    )
    collector.start()
    assert collector.is_running()
    time.sleep(0.3)
    collector.stop()
    assert not collector.is_running()


def test_collector_collects_samples():
    collector = NvidiaSmiCollector(
        simulation_mode=True, simulation_num_gpus=8,
        collection_interval_s=0.05,
        rolling_window_size=100,
    )
    collector.start()
    time.sleep(0.3)
    buffer = collector.get_buffer()
    collector.stop()
    assert len(buffer) > 0


def test_collector_feature_vector():
    collector = NvidiaSmiCollector(
        simulation_mode=True, simulation_num_gpus=8,
        collection_interval_s=0.05,
        rolling_window_size=100,
    )
    collector.start()
    time.sleep(0.5)
    features = collector.get_feature_vector()
    collector.stop()
    assert features is not None
    assert len(features) == 15
    assert features.dtype == np.float32


def test_collector_feature_sequence():
    collector = NvidiaSmiCollector(
        simulation_mode=True, simulation_num_gpus=8,
        collection_interval_s=0.05,
        rolling_window_size=100,
    )
    collector.start()
    time.sleep(1.0)
    seq = collector.get_feature_sequence(seq_len=10)
    collector.stop()
    if seq is not None:
        assert seq.shape[0] == 10
        assert seq.shape[1] == 15


def test_collector_get_stats():
    collector = NvidiaSmiCollector(simulation_mode=True)
    stats = collector.get_stats()
    assert stats["simulation_mode"] is True
    assert stats["buffer_size"] == 0


def test_collector_raw_buffer():
    collector = NvidiaSmiCollector(
        simulation_mode=True, simulation_num_gpus=4,
        collection_interval_s=0.05,
        rolling_window_size=50,
    )
    collector.start()
    time.sleep(0.3)
    raw = collector.get_raw_buffer()
    collector.stop()
    if len(raw) > 0:
        assert len(raw[0]) == 4


def test_parse_nvidia_smi_xml():
    xml = """<?xml version="1.0" ?>
    <nvidia_smi_log>
        <gpu>
            <minor_number>0</minor_number>
            <power_readings>
                <power_draw>250.5 W</power_draw>
            </power_readings>
            <temperature>
                <gpu_temp>65 C</gpu_temp>
            </temperature>
            <utilization>
                <gpu_util>80 %</gpu_util>
                <memory_util>50 %</memory_util>
            </utilization>
            <clocks>
                <graphics_clock>1980 MHz</graphics_clock>
                <mem_clock>2619 MHz</mem_clock>
            </clocks>
        </gpu>
    </nvidia_smi_log>"""
    samples = parse_nvidia_smi_xml(xml)
    assert len(samples) == 1
    assert samples[0].power_w == 250.5
    assert samples[0].temp_c == 65.0
    assert samples[0].util_pct == 80.0


def test_parse_nvidia_smi_xml_multi_gpu():
    xml = """<?xml version="1.0" ?>
    <nvidia_smi_log>
        <gpu>
            <minor_number>0</minor_number>
            <power_readings><power_draw>100 W</power_draw></power_readings>
            <temperature><gpu_temp>50 C</gpu_temp></temperature>
            <utilization><gpu_util>50 %</gpu_util><memory_util>30 %</memory_util></utilization>
            <clocks><graphics_clock>1000 MHz</graphics_clock><mem_clock>2000 MHz</mem_clock></clocks>
        </gpu>
        <gpu>
            <minor_number>1</minor_number>
            <power_readings><power_draw>200 W</power_draw></power_readings>
            <temperature><gpu_temp>60 C</gpu_temp></temperature>
            <utilization><gpu_util>70 %</gpu_util><memory_util>40 %</memory_util></utilization>
            <clocks><graphics_clock>1500 MHz</graphics_clock><mem_clock>2500 MHz</mem_clock></clocks>
        </gpu>
    </nvidia_smi_log>"""
    samples = parse_nvidia_smi_xml(xml)
    assert len(samples) == 2
    assert samples[0].gpu_id == 0
    assert samples[1].gpu_id == 1


def test_parse_nvidia_smi_xml_bad_input():
    import pytest
    with pytest.raises(ValueError, match="Failed to parse nvidia-smi XML"):
        parse_nvidia_smi_xml("<invalid>")


def test_collector_sim_source_cycle():
    from energivanu.telemetry.nvidia_smi_collector import _SimulatedGpuSource
    sim = _SimulatedGpuSource(num_gpus=8)
    samples = sim.collect()
    assert len(samples) == 8
    assert all(s.power_w > 0 for s in samples)


def test_collector_sim_source_allreduce_detection():
    from energivanu.telemetry.nvidia_smi_collector import _SimulatedGpuSource
    sim = _SimulatedGpuSource(num_gpus=1)
    powers = [sim.collect()[0].power_w for _ in range(20)]
    assert max(powers) > min(powers)
