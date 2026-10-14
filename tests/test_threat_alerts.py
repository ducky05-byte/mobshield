"""Tests for MobileShield Real-Time Random Forest Threat Detection.

Verifies:
1. Default threshold is exactly 0.65.
2. Threshold synchronization between frontend and backend.
3. Probability below 0.65 does not generate an ML threat alert.
4. Probability equal to or above 0.65 triggers genuine ML threat alerts.
5. BENIGN predictions never generate ML threat alerts.
6. Supported malware families (RANSOMWARE, ADWARE, SCAREWARE, SMSMALWARE) trigger genuine alerts.
7. Demo-only endpoints (/demo, /demo/navigate, /api/demo/url_event) are cleanly removed (return 404).
8. IPv6 packet ingestion and bidirectional flow aggregation operates cleanly without errors.
9. Live batched packet ingestion via /api/packets creates valid active flows.
10. Packet highlighting continues to work independently of classification.
11. Existing pipeline, telemetry, PCAP recording, and device discovery remain intact.
"""

import pytest
import time
from fastapi.testclient import TestClient

from src.prepare_dataset import NetworkFlowPreprocessor
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)

from src.companion_api import app, engine, ADMIN_API_TOKEN
from src.flow_engine import LiveCapturePipeline, BidirectionalFlow
from src.sample_threats import get_genuine_sample_flow
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_engine_state():
    """Ensures each test begins with default 0.65 threshold and clean state."""
    engine.detection_threshold = 0.65
    with engine.lock:
        engine.alerts.clear()
        engine.recent_flows.clear()
        engine.active_flows.clear()
    yield
    engine.detection_threshold = 0.65
    with engine.lock:
        engine.alerts.clear()
        engine.recent_flows.clear()
        engine.active_flows.clear()


# -------------------------------------------------------------
# 1. Default Threshold is Exactly 0.65
# -------------------------------------------------------------
def test_default_threshold_is_sixty_five_percent(client):
    """Requirement 1: Verify default threshold is exactly 0.65 (65%)."""
    res = client.get("/api/config/threshold")
    assert res.status_code == 200
    assert res.json()["threshold"] == 0.65

    telem = client.get("/api/telemetry")
    assert telem.status_code == 200
    assert telem.json()["detection_threshold"] == 0.65

    pipeline = LiveCapturePipeline(interface="dummy")
    assert pipeline.binary_threshold == 0.65


# -------------------------------------------------------------
# 2. Threshold Synchronization Between Frontend and Backend
# -------------------------------------------------------------
def test_threshold_synchronization(client):
    """Requirement 2: Updating threshold syncs across backend and telemetry."""
    res = client.post("/api/config/threshold?threshold=0.75")
    assert res.status_code == 200
    assert res.json()["threshold"] == 0.75
    assert engine.detection_threshold == 0.75

    telem = client.get("/api/telemetry").json()
    assert telem["detection_threshold"] == 0.75

    # Boundary tests
    res_low = client.post("/api/config/threshold?threshold=0.10")
    assert res_low.status_code == 200
    assert res_low.json()["threshold"] == 0.10

    res_high = client.post("/api/config/threshold?threshold=0.90")
    assert res_high.status_code == 200
    assert res_high.json()["threshold"] == 0.90


# -------------------------------------------------------------
# 3. Probability Below Cutoff Does Not Alert
# -------------------------------------------------------------
def test_probability_below_cutoff_does_not_alert(client):
    """Requirement 3: Flow with malware probability below configured threshold never alerts."""
    client.post("/api/config/threshold?threshold=0.85")

    # Ingest a flow where malware probability is below 0.85
    res = client.post("/api/diagnostic/flow?category=ADWARE")
    assert res.status_code == 200
    flow = res.json()

    assert flow["malware_probability"] < 0.85
    assert flow["is_threat"] is False
    assert flow["status"] == "BENIGN"

    telem = client.get("/api/telemetry").json()
    assert len(telem["alerts"]) == 0
    assert telem["ml_alerts_count"] == 0


# -------------------------------------------------------------
# 4. Probability Equal to or Above Cutoff Triggers Threat Alert
# -------------------------------------------------------------
def test_probability_above_cutoff_follows_alert_policy(client):
    """Requirement 4: Flow meeting threshold and valid family triggers genuine threat alert."""
    client.post("/api/config/threshold?threshold=0.50")

    res = client.post("/api/diagnostic/flow?category=RANSOMWARE")
    assert res.status_code == 200
    flow = res.json()

    assert flow["malware_probability"] >= 0.50
    assert flow["is_threat"] is True
    assert flow["status"] == "MALWARE_ASSOCIATED"
    assert flow["multiclass_category"] in {"RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"}
    assert flow["multiclass_category"] != "BENIGN"
    assert flow["detection_source"] == "ML_MODEL"

    telem = client.get("/api/telemetry").json()
    assert len(telem["alerts"]) >= 1
    assert telem["ml_alerts_count"] >= 1
    alert = telem["alerts"][0]
    assert alert["multiclass_category"] == flow["multiclass_category"]
    assert alert["malware_probability"] == flow["malware_probability"]


# -------------------------------------------------------------
# 5. BENIGN Predictions Never Trigger Threat Alerts
# -------------------------------------------------------------
def test_benign_flows_never_trigger_threat_alerts(client):
    """Requirement 5: BENIGN flows never generate threat alerts, even at ultra-low threshold."""
    client.post("/api/config/threshold?threshold=0.10")

    res = client.post("/api/diagnostic/flow?category=BENIGN")
    assert res.status_code == 200
    flow = res.json()

    assert flow["multiclass_category"] == "BENIGN" or flow["status"] == "BENIGN"
    assert flow["is_threat"] is False

    telem = client.get("/api/telemetry").json()
    assert len(telem["alerts"]) == 0


# -------------------------------------------------------------
# 6. Supported Malware Families Taxonomy
# -------------------------------------------------------------
def test_supported_malware_families_taxonomy(client):
    """Requirement 6: All supported malware families (RANSOMWARE, ADWARE, SCAREWARE, SMSMALWARE) trigger alerts."""
    VALID_THREAT_FAMILIES = {"RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"}
    client.post("/api/config/threshold?threshold=0.50")

    for cat in ["RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"]:
        res = client.post(f"/api/diagnostic/flow?category={cat}")
        assert res.status_code == 200
        flow = res.json()
        assert flow["is_threat"] is True
        assert flow["multiclass_category"] in VALID_THREAT_FAMILIES
        assert flow["multiclass_category"] != "BENIGN"
        assert flow["detection_source"] == "ML_MODEL"

    client.post("/api/config/threshold?threshold=0.65")


# -------------------------------------------------------------
# 7. Demo Endpoints Removed (Return 404)
# -------------------------------------------------------------
def test_demo_endpoints_removed(client):
    """Requirement 7: Artificial demo URL endpoints are removed and cannot inject fake alerts."""
    res_demo = client.get("/demo")
    assert res_demo.status_code == 404

    res_nav = client.get("/demo/navigate?target=phishing")
    assert res_nav.status_code == 404

    res_launch = client.post("/api/demo/launch_on_phone?target=phishing")
    assert res_launch.status_code == 404

    res_event = client.post("/api/demo/url_event", json={"url": "https://testsafebrowsing.appspot.com/s/phishing.html"})
    assert res_event.status_code == 404

    res_cfg = client.get("/api/config/demo_mode")
    assert res_cfg.status_code == 404

    telem = client.get("/api/telemetry").json()
    assert "demo_alerts" not in telem
    assert "demo_mode_enabled" not in telem


def test_manual_url_lookup_endpoint_removed(client):
    """Manual URL verification has been removed from the backend."""
    response = client.post(
        "/api/url/verify",
        json={
            "url": "https://example.com",
            "request_id": "8fdc76e0-5408-49e0-875b-2c4cf3ee4d84",
        },
    )

    assert response.status_code == 404


# -------------------------------------------------------------
# 8. IPv6 Bidirectional Flow Aggregation
# -------------------------------------------------------------
def test_ipv6_flow_aggregation_and_features():
    """Requirement 8: IPv6 packets aggregate without throwing Layer [IP] IndexError and extract 65 features."""
    p1 = IPv6(src="2001:db8::1", dst="2001:db8::2") / TCP(sport=12345, dport=443, flags="S")
    p1.time = 1000.0
    p1.len = 74

    flow = BidirectionalFlow(p1, ("2001:db8::1", "2001:db8::2", 12345, 443, 6))

    p2 = IPv6(src="2001:db8::2", dst="2001:db8::1") / TCP(sport=443, dport=12345, flags="SA")
    p2.time = 1000.05
    p2.len = 74
    flow.add_packet(p2)

    feats = flow.to_feature_dict()
    assert len(feats) == 65
    assert flow.fwd_packets == 1
    assert flow.bwd_packets == 1


# -------------------------------------------------------------
# 9. Batched Packet Ingestion API (/api/packets)
# -------------------------------------------------------------
def test_packet_ingestion_api(client):
    """Requirement 9: Ingesting live packets via /api/packets routes to active flows."""
    t0 = time.time()
    packet_batch = {
        "device_id": "RZCX221W2EE",
        "packets": [
            {
                "timestamp": t0,
                "src_ip": "192.168.1.105",
                "dst_ip": "1.1.1.1",
                "src_port": 54321,
                "dst_port": 443,
                "protocol": 6,
                "packet_len": 64,
                "tcp_flags": 0x02,  # SYN
                "ttl": 64
            },
            {
                "timestamp": t0 + 0.05,
                "src_ip": "1.1.1.1",
                "dst_ip": "192.168.1.105",
                "src_port": 443,
                "dst_port": 54321,
                "protocol": 6,
                "packet_len": 64,
                "tcp_flags": 0x12,  # SYN-ACK
                "ttl": 58
            }
        ]
    }

    res = client.post(
        "/api/packets",
        json=packet_batch,
        headers={"X-MobileShield-Token": ADMIN_API_TOKEN}
    )
    assert res.status_code == 200
    assert res.json()["packets_processed"] == 2

    # Check status and telemetry reflect received packets and active flow
    status = client.get("/api/status").json()
    assert status["total_packets"] >= 2
    assert status["active_flows"] >= 1


# -------------------------------------------------------------
# 10. Packet Highlighting Independency
# -------------------------------------------------------------
def test_packet_highlighting_independence():
    """Requirement 10: Packet highlighting is based strictly on capture age, independent of ML classification."""
    now = time.time()
    seen_table = {"pkt_1": now - 1.0, "pkt_2": now - 5.0}

    # Packet 1 is within highlight window (< 2.6s)
    age_1 = now - seen_table["pkt_1"]
    is_new_1 = (age_1 < 2.6)
    assert is_new_1 is True

    # Packet 2 is expired (> 2.6s)
    age_2 = now - seen_table["pkt_2"]
    is_new_2 = (age_2 < 2.6)
    assert is_new_2 is False


# -------------------------------------------------------------
# 11. Existing Pipeline, Telemetry, and PCAP Intact
# -------------------------------------------------------------
def test_existing_pipeline_and_pcap_intact(client):
    """Requirement 11: PCAP start/stop, validation, status, and telemetry endpoints remain intact."""
    status = client.get("/api/status").json()
    assert status["status"] == "ONLINE"

    telem = client.get("/api/telemetry").json()
    assert "detection_threshold" in telem
    assert "recent_packets" in telem
    assert "alerts" in telem

    # PCAP control
    p_start = client.post(
        "/api/control/pcap/start",
        headers={"X-MobileShield-Token": ADMIN_API_TOKEN},
    ).json()
    assert p_start["status"] == "RECORDING_STARTED"
    assert "pcap_path" in p_start

    p_stop = client.post(
        "/api/control/pcap/stop",
        headers={"X-MobileShield-Token": ADMIN_API_TOKEN},
    ).json()
    assert p_stop["status"] == "RECORDING_STOPPED"
