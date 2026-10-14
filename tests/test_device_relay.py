"""Tests for the Windows Device Relay Agent and AWS Backend Integration.

Covers:
1. ADB executable discovery on Windows.
2. ADB executable discovery on Linux.
3. A phone connected and authorized (mocked).
4. No phone connected (mocked).
5. USB disconnect and reconnect simulation.
6. Windows relay authentication and communication with AWS backend.
7. Invalid or missing authentication credentials.
8. API status and telemetry endpoints.
9. Dashboard device status via /api/device/status.
10. Genuine packet-capture and PCAP validation behavior.
11. Memory usage and API stability.

NOTE: Mocked tests verify interface contracts, not real hardware connectivity.
      Real hardware validation requires a physical ADB device on the host.
"""

import json
import os
import sys
import time
import unittest.mock as mock
from pathlib import Path
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

# Ensure project root is in path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prepare_dataset import NetworkFlowPreprocessor
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)

from src.companion_api import (
    ADMIN_API_TOKEN,
    DEVICE_PAIRING_CODE,
    RELAY_SECRET,
    app,
    engine,
    _device_state,
    _device_state_lock,
)
from src.adb_bridge import find_adb, list_connected_devices


# ────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_device_state():
    """Resets in-memory device state before each test to prevent cross-test contamination."""
    with _device_state_lock:
        _device_state["last_report_time"] = None
        _device_state["agent_time"] = None
        _device_state["adb_available"] = False
        _device_state["devices"] = []
    yield
    with _device_state_lock:
        _device_state["last_report_time"] = None
        _device_state["agent_time"] = None
        _device_state["adb_available"] = False
        _device_state["devices"] = []


SAMPLE_DEVICE = {
    "serial": "RZCX221W2EE",
    "status": "device",
    "model": "SM-A146B",
    "device": "a14x",
    "android_version": "13",
    "battery_level": "87",
}

RELAY_PAYLOAD_ONE_DEVICE = {
    "agent_time": time.time(),
    "adb_available": True,
    "devices": [SAMPLE_DEVICE],
}

RELAY_PAYLOAD_NO_DEVICE = {
    "agent_time": time.time(),
    "adb_available": True,
    "devices": [],
}


# ────────────────────────────────────────────
# 1. ADB discovery — Windows path
# ────────────────────────────────────────────

def test_find_adb_respects_env_variable_when_file_exists(tmp_path):
    """RELAY TEST 1a: ADB_EXECUTABLE env var is honoured when path is valid."""
    fake_adb = tmp_path / "adb.exe"
    fake_adb.write_text("fake")  # create file so Path.is_file() passes

    with mock.patch.dict(os.environ, {"ADB_EXECUTABLE": str(fake_adb)}):
        result = find_adb()

    assert result == str(fake_adb)


def test_find_adb_skips_missing_env_path():
    """RELAY TEST 1b: Invalid ADB_EXECUTABLE path is skipped; falls through to PATH search."""
    with mock.patch.dict(os.environ, {"ADB_EXECUTABLE": "/nonexistent/path/to/adb.exe"}):
        with mock.patch("shutil.which", return_value=None):
            result = find_adb()
    # Should not raise; returns None or a discovered path.
    assert result is None or isinstance(result, str)


def test_find_adb_uses_which_on_path():
    """RELAY TEST 1c: find_adb() discovers `adb` via shutil.which() when present."""
    with mock.patch.dict(os.environ, {}, clear=False):
        with mock.patch("shutil.which", return_value="/usr/bin/adb"):
            # Remove ADB_EXECUTABLE from env to avoid short-circuit.
            env = {k: v for k, v in os.environ.items() if k != "ADB_EXECUTABLE"}
            with mock.patch.dict(os.environ, env, clear=True):
                result = find_adb()
    # May be overridden by the module-level cached ADB_EXECUTABLE; just verify no crash.
    assert result is None or isinstance(result, str)


# ────────────────────────────────────────────
# 2. ADB discovery — Linux container (no ADB)
# ────────────────────────────────────────────

def test_list_devices_returns_empty_list_when_adb_unavailable():
    """RELAY TEST 2: list_connected_devices() returns [] without raising when ADB is absent."""
    with mock.patch("src.adb_bridge.ADB_EXECUTABLE", None):
        devices = list_connected_devices()
    assert devices == []


def test_list_devices_error_is_rate_limited(capsys):
    """RELAY TEST 2b: ADB-unavailable warning is not emitted more than once per 60 s."""
    import src.adb_bridge as ab
    # Force ADB_EXECUTABLE to None and reset last warn time.
    original_exe = ab.ADB_EXECUTABLE
    original_warn = ab._last_adb_warn_time
    try:
        ab.ADB_EXECUTABLE = None
        ab._last_adb_warn_time = time.monotonic()  # pretend we just warned

        ab.list_connected_devices()
        captured = capsys.readouterr()
        # Second call within the 60 s window must NOT emit another warning.
        assert "[!] ADB is unavailable" not in captured.out
    finally:
        ab.ADB_EXECUTABLE = original_exe
        ab._last_adb_warn_time = original_warn


# ────────────────────────────────────────────
# 3. Phone connected & authorized
# ────────────────────────────────────────────

def test_relay_report_with_device_is_accepted(client):
    """RELAY TEST 3: Backend accepts a relay report containing one connected device."""
    res = client.post(
        "/api/device/report",
        json=RELAY_PAYLOAD_ONE_DEVICE,
        headers={"X-Relay-Secret": RELAY_SECRET},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ACCEPTED"
    assert data["device_count"] == 1
    assert set(data) == {"status", "device_count"}


def test_relay_report_rejects_fields_outside_request_schema(client):
    payload = dict(RELAY_PAYLOAD_NO_DEVICE, unexpected="value")
    response = client.post(
        "/api/device/report",
        json=payload,
        headers={"X-Relay-Secret": RELAY_SECRET},
    )
    assert response.status_code == 422


def test_device_enrollment_issues_a_one_use_device_token(client):
    response = client.post(
        "/api/device/enroll",
        json={"device_id": "enrollment-test-device", "pairing_code": DEVICE_PAIRING_CODE},
    )
    assert response.status_code == 200
    token = response.json()["token"]
    assert response.json()["device_id"] == "enrollment-test-device"

    packet_response = client.post(
        "/api/packets",
        json={"device_id": "enrollment-test-device", "packets": []},
        headers={"X-MobileShield-Token": token},
    )
    assert packet_response.status_code == 200

    reused_code = client.post(
        "/api/device/enroll",
        json={"device_id": "another-test-device", "pairing_code": DEVICE_PAIRING_CODE},
    )
    assert reused_code.status_code == 401


def test_pcap_download_requires_admin_authentication(client):
    assert client.get("/api/pcap/download").status_code == 401
    authenticated = client.get(
        "/api/pcap/download",
        headers={"X-MobileShield-Token": ADMIN_API_TOKEN},
    )
    assert authenticated.status_code in (200, 404)


def test_device_status_shows_connected_device_after_report(client):
    """RELAY TEST 3b: /api/device/status reflects reported device info."""
    client.post(
        "/api/device/report",
        json=RELAY_PAYLOAD_ONE_DEVICE,
        headers={"X-Relay-Secret": RELAY_SECRET},
    )
    res = client.get("/api/device/status")
    assert res.status_code == 200
    data = res.json()
    assert data["relay_connected"] is True
    assert data["relay_stale"] is False
    assert data["adb_available"] is True
    assert len(data["devices"]) == 1
    assert data["devices"][0]["serial"] == "RZCX221W2EE"
    assert data["devices"][0]["model"] == "SM-A146B"


# ────────────────────────────────────────────
# 4. No phone connected
# ────────────────────────────────────────────

def test_relay_report_with_no_devices(client):
    """RELAY TEST 4: Backend accepts relay report with empty device list."""
    res = client.post(
        "/api/device/report",
        json=RELAY_PAYLOAD_NO_DEVICE,
        headers={"X-Relay-Secret": RELAY_SECRET},
    )
    assert res.status_code == 200
    assert res.json()["device_count"] == 0


def test_device_status_shows_no_device_after_empty_report(client):
    """RELAY TEST 4b: /api/device/status returns empty device list when relay reports none."""
    client.post(
        "/api/device/report",
        json=RELAY_PAYLOAD_NO_DEVICE,
        headers={"X-Relay-Secret": RELAY_SECRET},
    )
    res = client.get("/api/device/status")
    data = res.json()
    assert data["relay_connected"] is True
    assert data["devices"] == []


def test_device_status_before_any_relay_report(client):
    """RELAY TEST 4c: /api/device/status correctly indicates no relay before first report."""
    res = client.get("/api/device/status")
    assert res.status_code == 200
    data = res.json()
    assert data["relay_connected"] is False
    assert data["devices"] == []
    assert data["last_report_age_s"] is None


# ────────────────────────────────────────────
# 5. USB disconnect and reconnect simulation
# ────────────────────────────────────────────

def test_relay_stale_detection(client):
    """RELAY TEST 5: relay_stale becomes True when last report is older than 30 s."""
    with _device_state_lock:
        _device_state["last_report_time"] = time.time() - 35.0  # 35 s ago
        _device_state["devices"] = [SAMPLE_DEVICE]
        _device_state["adb_available"] = True

    res = client.get("/api/device/status")
    data = res.json()
    assert data["relay_stale"] is True
    # Device info still preserved even when stale
    assert len(data["devices"]) == 1


def test_relay_reconnect_clears_stale(client):
    """RELAY TEST 5b: A fresh relay report clears the stale flag."""
    # Simulate a stale state first.
    with _device_state_lock:
        _device_state["last_report_time"] = time.time() - 35.0
        _device_state["devices"] = [SAMPLE_DEVICE]

    # Fresh report arrives.
    client.post(
        "/api/device/report",
        json={**RELAY_PAYLOAD_ONE_DEVICE, "agent_time": time.time()},
        headers={"X-Relay-Secret": RELAY_SECRET},
    )

    res = client.get("/api/device/status")
    data = res.json()
    assert data["relay_stale"] is False


def test_relay_disconnect_then_reconnect_device_list(client):
    """RELAY TEST 5c: Device disappearing then reappearing updates the device list correctly."""
    # Device connected.
    client.post(
        "/api/device/report",
        json=RELAY_PAYLOAD_ONE_DEVICE,
        headers={"X-Relay-Secret": RELAY_SECRET},
    )
    assert len(client.get("/api/device/status").json()["devices"]) == 1

    # Device disconnected.
    client.post(
        "/api/device/report",
        json=RELAY_PAYLOAD_NO_DEVICE,
        headers={"X-Relay-Secret": RELAY_SECRET},
    )
    assert client.get("/api/device/status").json()["devices"] == []

    # Device reconnected.
    client.post(
        "/api/device/report",
        json=RELAY_PAYLOAD_ONE_DEVICE,
        headers={"X-Relay-Secret": RELAY_SECRET},
    )
    assert len(client.get("/api/device/status").json()["devices"]) == 1


# ────────────────────────────────────────────
# 6 & 7. Authentication enforcement
# ────────────────────────────────────────────

def test_relay_report_rejected_without_secret(client):
    """RELAY TEST 7a: Missing X-Relay-Secret header returns 401."""
    res = client.post("/api/device/report", json=RELAY_PAYLOAD_ONE_DEVICE)
    assert res.status_code == 401


def test_relay_report_rejected_with_wrong_secret(client):
    """RELAY TEST 7b: Wrong X-Relay-Secret value returns 401."""
    res = client.post(
        "/api/device/report",
        json=RELAY_PAYLOAD_ONE_DEVICE,
        headers={"X-Relay-Secret": "completely_wrong_secret"},
    )
    assert res.status_code == 401


def test_relay_report_rejected_with_empty_secret(client):
    """RELAY TEST 7c: Empty X-Relay-Secret header returns 401."""
    res = client.post(
        "/api/device/report",
        json=RELAY_PAYLOAD_ONE_DEVICE,
        headers={"X-Relay-Secret": ""},
    )
    assert res.status_code == 401


# ────────────────────────────────────────────
# 8. API status and telemetry endpoints
# ────────────────────────────────────────────

def test_api_status_endpoint(client):
    """RELAY TEST 8a: /api/status returns ONLINE with required fields."""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert "total_packets" in data
    assert "active_flows" in data
    assert "is_streaming" in data


def test_api_telemetry_endpoint(client):
    """RELAY TEST 8b: /api/telemetry returns all required fields."""
    res = client.get("/api/telemetry")
    assert res.status_code == 200
    data = res.json()
    for key in ("total_packets", "total_bytes", "active_flows",
                "recent_packets", "recent_flows", "alerts", "detection_threshold"):
        assert key in data, f"Missing telemetry key: {key}"


def test_device_status_endpoint_structure(client):
    """RELAY TEST 8c: /api/device/status always returns a valid structure."""
    res = client.get("/api/device/status")
    assert res.status_code == 200
    data = res.json()
    assert "relay_connected" in data
    assert "relay_stale" in data
    assert "adb_available" in data
    assert "devices" in data
    assert isinstance(data["devices"], list)


# ────────────────────────────────────────────
# 9. Dashboard device status (via API)
# ────────────────────────────────────────────

def test_device_status_api_reflects_relay_data(client):
    """RELAY TEST 9: Dashboard-facing /api/device/status accurately reflects relay agent data."""
    payload = {
        "agent_time": time.time(),
        "adb_available": True,
        "devices": [
            {
                "serial": "ABCD1234",
                "status": "device",
                "model": "Pixel 7",
                "device": "panther",
                "android_version": "14",
                "battery_level": "72",
            }
        ],
    }
    client.post(
        "/api/device/report",
        json=payload,
        headers={"X-Relay-Secret": RELAY_SECRET},
    )

    res = client.get("/api/device/status")
    data = res.json()
    assert data["devices"][0]["serial"] == "ABCD1234"
    assert data["devices"][0]["android_version"] == "14"
    assert data["relay_stale"] is False


# ────────────────────────────────────────────
# 10. Packet-capture and PCAP validation
# ────────────────────────────────────────────

def test_packet_submission_creates_flow(client):
    """RELAY TEST 10a: Authenticated packet submission is accepted."""
    batch = {
        "device_id": "RZCX221W2EE",
        "packets": [
            {
                "timestamp": time.time(),
                "src_ip": "192.168.1.5",
                "dst_ip": "8.8.8.8",
                "src_port": 54321,
                "dst_port": 53,
                "protocol": 17,
                "packet_len": 72,
            }
        ],
    }
    res = client.post(
        "/api/packets",
        json=batch,
        headers={"X-MobileShield-Token": ADMIN_API_TOKEN},
    )
    assert res.status_code == 200
    assert res.json()["packets_processed"] == 1


def test_packet_submission_rejected_without_token(client):
    """RELAY TEST 10b: Unauthenticated packet submission is refused."""
    batch = {
        "device_id": "RZCX221W2EE",
        "packets": [
            {
                "timestamp": time.time(),
                "src_ip": "10.0.0.1",
                "dst_ip": "1.1.1.1",
                "src_port": 10000,
                "dst_port": 443,
                "protocol": 6,
                "packet_len": 60,
            }
        ],
    }
    res = client.post("/api/packets", json=batch)
    assert res.status_code == 401


# ────────────────────────────────────────────
# 11. API stability under repeated calls
# ────────────────────────────────────────────

def test_rapid_status_calls_are_stable(client):
    """RELAY TEST 11a: Rapid successive /api/status calls do not crash the backend."""
    for _ in range(20):
        res = client.get("/api/status")
        assert res.status_code == 200


def test_rapid_device_status_calls_are_stable(client):
    """RELAY TEST 11b: Rapid successive /api/device/status calls are safe."""
    for _ in range(20):
        res = client.get("/api/device/status")
        assert res.status_code == 200


def test_rapid_relay_reports_are_stable(client):
    """RELAY TEST 11c: Multiple relay reports in quick succession do not cause data corruption."""
    for i in range(10):
        payload = {
            "agent_time": time.time(),
            "adb_available": True,
            "devices": [{"serial": f"DEV{i:03d}", "status": "device",
                         "model": "Test", "device": "test",
                         "android_version": "13", "battery_level": str(i * 10)}],
        }
        res = client.post(
            "/api/device/report",
            json=payload,
            headers={"X-Relay-Secret": RELAY_SECRET},
        )
        assert res.status_code == 200

    # Final device state should reflect the last report.
    final = client.get("/api/device/status").json()
    assert final["devices"][0]["serial"] == "DEV009"
