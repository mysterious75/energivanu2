"""Tests for grid integration modules (OpenADR + ERCOT SCED)."""
from datetime import datetime, timezone, timedelta

from energivanu.grid.openadr_ven import (
    OpenADRVEN, GridEvent, GridSignalLevel,
)
from energivanu.grid.ercot_sced import (
    ERCOTSCEDClient, SCEDSignal, SCEDResponseType,
)


# =========================================================================
# OpenADR VEN Tests
# =========================================================================

def test_gridsignal_level_values():
    assert GridSignalLevel.NORMAL == 0
    assert GridSignalLevel.MODERATE == 1
    assert GridSignalLevel.HIGH == 2
    assert GridSignalLevel.CRITICAL == 3


def test_openddr_ven_default_init():
    ven = OpenADRVEN()
    assert ven.ven_id == "energivanu-001"
    assert ven.poll_interval_s == 30.0


def test_openddr_ven_simulate_event_normal():
    ven = OpenADRVEN()
    event = ven.simulate_event(level=GridSignalLevel.NORMAL)
    assert event.signal_level == GridSignalLevel.NORMAL
    assert event.action == "none"


def test_openddr_ven_simulate_event_moderate():
    ven = OpenADRVEN()
    event = ven.simulate_event(level=GridSignalLevel.MODERATE, duration_seconds=300.0)
    assert event.signal_level == GridSignalLevel.MODERATE
    assert event.action == "reduce_10pct"
    assert event.duration_seconds == 300.0


def test_openddr_ven_simulate_event_high():
    ven = OpenADRVEN()
    event = ven.simulate_event(level=GridSignalLevel.HIGH)
    assert event.signal_level == GridSignalLevel.HIGH
    assert event.action == "reduce_30pct"


def test_openddr_ven_simulate_event_critical():
    ven = OpenADRVEN()
    event = ven.simulate_event(level=GridSignalLevel.CRITICAL)
    assert event.signal_level == GridSignalLevel.CRITICAL
    assert event.action == "reduce_50pct_plus"


def test_openddr_ven_event_is_active():
    ven = OpenADRVEN()
    event = ven.simulate_event(level=GridSignalLevel.MODERATE, duration_seconds=60.0)
    assert event.is_active is True


def test_openddr_ven_process_event():
    ven = OpenADRVEN()
    event = GridEvent(
        event_id="test-001",
        signal_level=GridSignalLevel.HIGH,
        signal_value=30.0,
        start_time=datetime.now(timezone.utc) - timedelta(minutes=5),
        end_time=datetime.now(timezone.utc) + timedelta(minutes=5),
        priority=1,
        action="reduce_30pct",
    )
    result = ven.process_event(event)
    assert "event" in result
    assert "recommended_actions" in result
    assert result["event"]["signal_level"] == "HIGH"


def test_openddr_ven_get_current_signal_default():
    ven = OpenADRVEN()
    level = ven.get_current_signal_level()
    assert level == GridSignalLevel.NORMAL


def test_openddr_ven_get_current_signal_after_event():
    ven = OpenADRVEN()
    ven.simulate_event(level=GridSignalLevel.CRITICAL, duration_seconds=60.0)
    level = ven.get_current_signal_level()
    assert level == GridSignalLevel.CRITICAL


def test_openddr_ven_get_stats():
    ven = OpenADRVEN()
    ven.simulate_event(level=GridSignalLevel.MODERATE)
    stats = ven.get_stats()
    assert stats["ven_id"] == "energivanu-001"
    assert stats["total_events"] >= 1


def test_openddr_ven_callback_fired():
    ven = OpenADRVEN()
    called = []
    def callback(event):
        called.append(event)
    ven.on_event_callback = callback
    ven.simulate_event(level=GridSignalLevel.HIGH)
    assert len(called) == 1
    assert called[0].signal_level == GridSignalLevel.HIGH


# =========================================================================
# ERCOT SCED Tests
# =========================================================================

def test_sced_client_default_init():
    client = ERCOTSCEDClient()
    assert client.config.qse_id == "QSE001"
    assert client.config.resource_id == "DC_LOAD_001"


def test_sced_parse_normal_message():
    client = ERCOTSCEDClient()
    signal = client.parse_sced_message({
        "basePoint": 150.0,
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 200.0,
    })
    assert isinstance(signal, SCEDSignal)
    assert signal.base_point_mw == 150.0
    assert signal.response_type == SCEDResponseType.REDUCE


def test_sced_parse_within_deadband():
    client = ERCOTSCEDClient(max_power_mw=200.0)
    signal = client.parse_sced_message({
        "basePoint": 198.0,
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 200.0,
    })
    assert signal.response_type == SCEDResponseType.NORMAL


def test_sced_parse_emergency():
    client = ERCOTSCEDClient(min_power_mw=50.0)
    signal = client.parse_sced_message({
        "basePoint": 60.0,
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 200.0,
    })
    assert signal.response_type == SCEDResponseType.EMERGENCY_REDUCE


def test_sced_parse_shed_load():
    client = ERCOTSCEDClient(min_power_mw=50.0)
    signal = client.parse_sced_message({
        "basePoint": 50.0,
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 200.0,
    })
    assert signal.response_type == SCEDResponseType.SHED_LOAD


def test_sced_parse_increase():
    client = ERCOTSCEDClient(max_power_mw=200.0)
    signal = client.parse_sced_message({
        "basePoint": 250.0,
        "lowEmergencyLimit": 50.0,
        "highEmergencyLimit": 300.0,
    })
    assert signal.response_type == SCEDResponseType.INCREASE


def test_sced_generate_command_hold():
    from energivanu.grid.ercot_sced import PCLRConfig
    config = PCLRConfig(max_power_mw=200.0, deadband_mw=5.0)
    client = ERCOTSCEDClient(
        max_power_mw=config.max_power_mw,
        min_power_mw=config.min_power_mw,
        ramp_rate_mw_per_min=config.ramp_rate_mw_per_min,
        response_time_s=config.response_time_s,
    )
    client.config.deadband_mw = 5.0
    signal = client.parse_sced_message({
        "basePoint": 200.0,
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 300.0,
    })
    cmd = client.generate_command(signal, current_power_mw=200.0)
    assert cmd["action"] == "hold"
    assert cmd["reason"] == "within_deadband"


def test_sced_generate_command_reduce():
    client = ERCOTSCEDClient(max_power_mw=200.0)
    signal = client.parse_sced_message({
        "basePoint": 150.0,
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 200.0,
    })
    cmd = client.generate_command(signal, current_power_mw=200.0)
    assert cmd["action"] == "reduce"


def test_sced_generate_command_with_bess():
    from energivanu.mpc import MPCController
    mpc = MPCController()
    client = ERCOTSCEDClient(max_power_mw=200.0, mpc_controller=mpc)
    signal = client.parse_sced_message({
        "basePoint": 150.0,
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 200.0,
    })
    cmd = client.generate_command(signal, current_power_mw=200.0)
    assert cmd["mpc_command"] is not None
    assert cmd["mpc_command"]["action"] == "dispatch_bess"


def test_sced_check_compliance_pass():
    client = ERCOTSCEDClient()
    client.config.deadband_mw = 5.0
    signal = client.parse_sced_message({
        "basePoint": 150.0,
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 200.0,
    })
    report = client.check_compliance(signal, actual_power_mw=152.0, response_time_s=120.0)
    assert report["compliant"] is True


def test_sced_check_compliance_fail():
    client = ERCOTSCEDClient()
    client.config.deadband_mw = 5.0
    signal = client.parse_sced_message({
        "basePoint": 150.0,
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 200.0,
    })
    report = client.check_compliance(signal, actual_power_mw=200.0, response_time_s=120.0)
    assert report["compliant"] is False
    assert "violation_reasons" in report


def test_sced_simulate_sequence():
    client = ERCOTSCEDClient()
    signals = client.simulate_sced_sequence(duration_minutes=10.0, interval_s=60.0)
    assert len(signals) == 10
    assert all(isinstance(s, SCEDSignal) for s in signals)


def test_sced_get_stats():
    client = ERCOTSCEDClient()
    client.parse_sced_message({
        "basePoint": 150.0,
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 200.0,
    })
    stats = client.get_stats()
    assert stats["total_signals"] == 1
    assert stats["resource_id"] == "DC_LOAD_001"


def test_sced_signal_is_emergency():
    signal = SCEDSignal(
        base_point_mw=50.0,
        low_emergency_mw=100.0,
        high_emergency_mw=200.0,
        timestamp=datetime.now(timezone.utc),
        resource_id="test",
        qse_id="test",
        response_type=SCEDResponseType.EMERGENCY_REDUCE,
    )
    assert signal.is_emergency is True


def test_sced_signal_to_dict():
    signal = SCEDSignal(
        base_point_mw=150.0,
        low_emergency_mw=100.0,
        high_emergency_mw=200.0,
        timestamp=datetime.now(timezone.utc),
        resource_id="DC_001",
        qse_id="QSE001",
    )
    d = signal.to_dict()
    assert d["base_point_mw"] == 150.0
    assert d["resource_id"] == "DC_001"
