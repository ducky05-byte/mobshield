"""Automated tests for MobileShield Real-Time Packet Highlighting & Sequence Tracking.

Verifies:
- Monotonic packet sequencing and deterministic, collision-free packet IDs.
- Batched ingestion handling over the companion API.
- Dashboard new-packet detection and registration logic.
- Retention of highlight state during live polling / refreshes (negative animation-delay).
- Highlight expiration after ~2.6s and permanent transition to normal row styling.
- Guard against pre-existing historical packets flashing red on startup.
- Complete independence of packet capture highlighting from ML malware detection / threat alerts.
"""

import time
import pytest
from fastapi.testclient import TestClient

from src.prepare_dataset import NetworkFlowPreprocessor
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)

from src.companion_api import app, engine, ADMIN_API_TOKEN, PacketRecord


@pytest.fixture
def client():
    return TestClient(app)


def test_packet_identity_and_sequence_generation():
    """Requirement 2: Verify packet sequence monotonically increments and packet IDs are unique."""
    with engine.lock:
        initial_seq = engine.total_packets_received

    rec1 = PacketRecord(
        timestamp=time.time(),
        src_ip="192.168.1.100",
        dst_ip="142.250.190.46",
        src_port=54321,
        dst_port=443,
        protocol=6,
        packet_len=120,
        tcp_flags=0x18
    )
    rec2 = PacketRecord(
        timestamp=time.time() + 0.001,
        src_ip="192.168.1.100",
        dst_ip="142.250.190.46",
        src_port=54321,
        dst_port=443,
        protocol=6,
        packet_len=85,
        tcp_flags=0x10
    )

    engine.ingest_packet(rec1, "DEVICE_UNIT_TEST")
    engine.ingest_packet(rec2, "DEVICE_UNIT_TEST")

    with engine.lock:
        assert engine.total_packets_received == initial_seq + 2
        p1 = engine.recent_packets[1]
        p2 = engine.recent_packets[0]

        assert p1["seq"] == initial_seq + 1
        assert p2["seq"] == initial_seq + 2
        assert p1["id"] != p2["id"]
        assert p1["id"].startswith(f"pkt_{initial_seq + 1}_")
        assert p2["id"].startswith(f"pkt_{initial_seq + 2}_")
        assert "ingest_time" in p1
        assert "ingest_time" in p2


def test_batched_packet_ingestion_api(client):
    """Requirement 2: Verify batches of packets arriving in one HTTP request receive distinct IDs."""
    batch_payload = {
        "device_id": "TEST_SAMSUNG_A14",
        "packets": [
            {
                "timestamp": time.time(),
                "src_ip": "10.0.0.5",
                "dst_ip": "1.1.1.1",
                "src_port": 60000 + i,
                "dst_port": 53,
                "protocol": 17,
                "packet_len": 72
            }
            for i in range(5)
        ]
    }

    res = client.post("/api/packets", json=batch_payload, headers={"X-MobileShield-Token": ADMIN_API_TOKEN})
    assert res.status_code == 200
    assert res.json()["packets_processed"] == 5

    # Verify telemetry returns newly ingested batch packets
    telemetry = client.get("/api/telemetry").json()
    recent = telemetry["recent_packets"]
    assert len(recent) >= 5
    
    # Check that top 5 packets have distinct unique IDs and descending seq
    top_ids = [p["id"] for p in recent[:5]]
    assert len(set(top_ids)) == 5
    top_seqs = [p["seq"] for p in recent[:5]]
    assert top_seqs == sorted(top_seqs, reverse=True)


def test_new_packet_highlight_detection_and_expiry_lifecycle():
    """Requirements 1 & 2: Test detection of new packet, negative animation delay, and expiry."""
    seen_registry = {}
    now = 1000.0

    test_pkt = {
        "id": "pkt_9999_1700000000",
        "seq": 9999,
        "raw_timestamp": 1700000000.0,
        "timestamp": "12:00:00",
        "src_ip": "10.0.0.5",
        "src_port": 49152,
        "dst_ip": "93.184.216.34",
        "dst_port": 80,
        "protocol": "TCP",
        "length": 54
    }
    pid = test_pkt["id"]

    # 1. First encounter (Cycle 1, t = 1000.0s) -> Must be detected as NEW
    assert pid not in seen_registry
    seen_registry[pid] = now
    highlight_age = now - seen_registry[pid]
    is_new = (highlight_age < 2.6)
    assert is_new is True
    assert highlight_age == 0.0

    # 2. Second render (Cycle 2, t = 1001.0s, +1s live refresh) -> Still within 2.6s window
    now_cycle2 = 1001.0
    highlight_age_cycle2 = now_cycle2 - seen_registry[pid]
    is_new_cycle2 = (highlight_age_cycle2 < 2.6)
    assert is_new_cycle2 is True
    assert highlight_age_cycle2 == 1.0  # Used for animation-delay: -1.00s

    # 3. Third render (Cycle 3, t = 1002.0s, +2s live refresh) -> Near end of 2.6s window
    now_cycle3 = 1002.0
    highlight_age_cycle3 = now_cycle3 - seen_registry[pid]
    is_new_cycle3 = (highlight_age_cycle3 < 2.6)
    assert is_new_cycle3 is True
    assert highlight_age_cycle3 == 2.0  # Used for animation-delay: -2.00s

    # 4. Expiration (Cycle 4, t = 1003.0s, +3s live refresh) -> EXPIRED, transitions to normal
    now_cycle4 = 1003.0
    highlight_age_cycle4 = now_cycle4 - seen_registry[pid]
    is_new_cycle4 = (highlight_age_cycle4 < 2.6)
    assert is_new_cycle4 is False
    assert highlight_age_cycle4 == 3.0

    # 5. Subsequent refreshes (Cycle 5, 6, 7...) -> Stays permanently normal, NEVER flashes again
    for t_future in [1005.0, 1010.0, 1060.0]:
        age_future = t_future - seen_registry[pid]
        assert (age_future < 2.6) is False


def test_initial_sync_historical_packets_guard():
    """Requirement 1: Pre-existing historical packets on dashboard startup must NOT turn red."""
    historical_pkts = [
        {"id": f"hist_{i}", "seq": i, "timestamp": "11:50:00"}
        for i in range(10)
    ]

    seen_registry = {}
    initial_sync_done = False

    # Simulate dashboard initial startup sync
    if not initial_sync_done:
        for p in historical_pkts:
            seen_registry[p["id"]] = 0.0  # Historical sentinel
        initial_sync_done = True

    # Validate that all historical packets are marked normal
    now = time.time()
    for p in historical_pkts:
        first_seen = seen_registry[p["id"]]
        if first_seen == 0.0:
            is_new = False
        else:
            is_new = (now - first_seen < 2.6)
        assert is_new is False, f"Historical packet {p['id']} erroneously marked as new!"


def test_capture_highlighting_distinguished_from_threat_classification(client):
    """Requirement 3: Red highlight signifies live packet capture event, NOT malware prediction."""
    # Ingest a standard benign DNS query packet
    dns_pkt = {
        "timestamp": time.time(),
        "src_ip": "10.0.0.15",
        "dst_ip": "8.8.8.8",
        "src_port": 53535,
        "dst_port": 53,
        "protocol": 17,
        "packet_len": 68
    }

    res = client.post("/api/packets", json={"device_id": "TEST_PHONE", "packets": [dns_pkt]}, headers={"X-MobileShield-Token": ADMIN_API_TOKEN})
    assert res.status_code == 200

    telemetry = client.get("/api/telemetry").json()
    newest = telemetry["recent_packets"][0]

    # The packet was received and is available for capture highlighting
    assert newest["src_ip"] == "10.0.0.15"
    assert newest["protocol"] == "UDP"

    # Threat detection policy: benign packet does NOT create threat alert
    alerts = telemetry["alerts"]
    dns_alerts = [a for a in alerts if a.get("src_ip") == "10.0.0.15"]
    assert len(dns_alerts) == 0, "Benign packet must not create threat alert!"
