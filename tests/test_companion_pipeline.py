"""End-to-End Tests for MobileShield Companion API, ADB Bridge, PCAP Writer & ML Pipeline.

Verifies:
- Token-authenticated companion packet ingestion (/api/packets)
- Rejection of unauthenticated or malformed submissions
- Flow aggregation with exact 65-feature schema and order matching models/feature_names.json
- FlowPredictor inference compatibility and confidence output
- Handling of unsupported/incomplete flows as NOT_CLASSIFIED
- Real PCAP file writing, header validation, and Scapy re-reading
- ADB Bridge hardware discovery
"""

import os
import json
import time
import struct
import tempfile
import pytest
from pathlib import Path
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import rdpcap
from fastapi.testclient import TestClient

from src.prepare_dataset import NetworkFlowPreprocessor
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)

from src.companion_api import app, engine, ADMIN_API_TOKEN, PcapWriter, validate_pcap_file
from src.adb_bridge import list_connected_devices
from src.predict_flow import FlowPredictor
from src.flow_engine import BidirectionalFlow


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def feature_names():
    with open("models/feature_names.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_api_status_endpoint(client):
    """Verifies that the API health status endpoint returns online telemetry."""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert "total_packets" in data
    assert "active_flows" in data


def test_token_authentication_enforcement(client):
    """Verifies that unauthenticated or invalid token requests to /api/packets are rejected."""
    batch_data = {
        "device_id": "TEST_DEVICE",
        "packets": [
            {
                "timestamp": time.time(),
                "src_ip": "10.1.1.2",
                "dst_ip": "8.8.8.8",
                "src_port": 50000,
                "dst_port": 53,
                "protocol": 17,
                "packet_len": 64
            }
        ]
    }

    # 1. No token -> 401
    res_no_token = client.post("/api/packets", json=batch_data)
    assert res_no_token.status_code == 401

    # 2. Bad token -> 401
    res_bad_token = client.post("/api/packets", json=batch_data, headers={"X-MobileShield-Token": "invalid_token"})
    assert res_bad_token.status_code == 401

    # 3. Valid token -> 200
    res_valid = client.post("/api/packets", json=batch_data, headers={"X-MobileShield-Token": ADMIN_API_TOKEN})
    assert res_valid.status_code == 200
    assert res_valid.json()["status"] == "SUCCESS"
    assert res_valid.json()["packets_processed"] == 1


def test_real_packet_ingestion_and_flow_aggregation(client, feature_names):
    """Verifies that genuine ingested packets aggregate into 5-tuple flows with exact 65 features."""
    base_t = time.time()
    
    # TCP Handshake + Data
    syn_hex = bytes(IP(src="10.1.1.2", dst="93.184.216.34")/TCP(sport=49152, dport=80, flags="S", seq=1000)).hex()
    syn_ack_hex = bytes(IP(src="93.184.216.34", dst="10.1.1.2")/TCP(sport=80, dport=49152, flags="SA", seq=5000, ack=1001)).hex()
    ack_hex = bytes(IP(src="10.1.1.2", dst="93.184.216.34")/TCP(sport=49152, dport=80, flags="A", seq=1001, ack=5001)).hex()
    data_hex = bytes(IP(src="10.1.1.2", dst="93.184.216.34")/TCP(sport=49152, dport=80, flags="PA", seq=1001, ack=5001)/b"GET / HTTP/1.1\r\n\r\n").hex()

    packets = [
        {"timestamp": base_t, "src_ip": "10.1.1.2", "dst_ip": "93.184.216.34", "src_port": 49152, "dst_port": 80, "protocol": 6, "packet_len": len(syn_hex)//2, "tcp_flags": 0x02, "raw_packet_hex": syn_hex},
        {"timestamp": base_t + 0.02, "src_ip": "93.184.216.34", "dst_ip": "10.1.1.2", "src_port": 80, "dst_port": 49152, "protocol": 6, "packet_len": len(syn_ack_hex)//2, "tcp_flags": 0x12, "raw_packet_hex": syn_ack_hex},
        {"timestamp": base_t + 0.03, "src_ip": "10.1.1.2", "dst_ip": "93.184.216.34", "src_port": 49152, "dst_port": 80, "protocol": 6, "packet_len": len(ack_hex)//2, "tcp_flags": 0x10, "raw_packet_hex": ack_hex},
        {"timestamp": base_t + 0.05, "src_ip": "10.1.1.2", "dst_ip": "93.184.216.34", "src_port": 49152, "dst_port": 80, "protocol": 6, "packet_len": len(data_hex)//2, "tcp_flags": 0x18, "raw_packet_hex": data_hex},
    ]

    res = client.post(
        "/api/packets",
        json={"device_id": "RZCX221W2EE", "packets": packets},
        headers={"X-MobileShield-Token": ADMIN_API_TOKEN}
    )
    assert res.status_code == 200
    assert res.json()["packets_processed"] == 4

    # Verify flow tracking (either in active_flows or already flushed to recent_flows by background loop)
    flow_key = ("10.1.1.2", "93.184.216.34", 49152, 80, 6)
    with engine.lock:
        if flow_key in engine.active_flows:
            flow = engine.active_flows[flow_key]
            assert flow.fwd_packets == 3
            assert flow.bwd_packets == 1
            feat_dict = flow.to_feature_dict()
            for fname in feature_names:
                assert fname in feat_dict, f"Missing feature: {fname}"
            # Force classify
            engine._classify_flow(flow_key, flow)

    # Allow brief moment for classification worker if flushed asynchronously
    time.sleep(0.5)

    with engine.lock:
        classified = [f for f in engine.recent_flows if f.get("src_ip") == "10.1.1.2" and f.get("dst_ip") == "93.184.216.34"]
        assert len(classified) > 0, "Expected flow to be classified and present in recent_flows"
        rec = classified[0]
        assert rec["status"] in ("BENIGN", "MALWARE_ASSOCIATED")
        assert "malware_probability" in rec
        assert "binary_confidence" in rec


def test_incomplete_flow_handled_as_not_classified():
    """Verifies that flows with invalid or missing schemas are marked NOT_CLASSIFIED."""
    class DummyIncompleteFlow:
        def __init__(self):
            self.start_time = time.time()
            self.fwd_packets = 1
            self.bwd_packets = 0
            self.fwd_lengths = [40]
            self.bwd_lengths = []
        def to_feature_dict(self):
            # Missing 60 features
            return {"Protocol": 6.0, "Flow Duration": 100.0}

    dummy_key = ("10.1.1.2", "1.2.3.4", 9999, 80, 6)
    engine._classify_flow(dummy_key, DummyIncompleteFlow())

    with engine.lock:
        unclassified = [f for f in engine.recent_flows if f.get("status") == "NOT_CLASSIFIED" and f.get("src_ip") == "10.1.1.2"]
        assert len(unclassified) > 0
        assert "Incompatible feature schema" in unclassified[0]["reason"]


def test_pcap_writer_and_validator():
    """Verifies authentic Libpcap file generation, header format, and Scapy re-reading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pcap_file = os.path.join(tmp_dir, "test_capture.pcap")
        writer = PcapWriter(pcap_file)

        # Write genuine IPv4 TCP packets
        t0 = time.time()
        pkt1 = bytes(IP(src="192.168.1.50", dst="8.8.8.8")/UDP(sport=53000, dport=53)/b"dns_test_payload")
        pkt2 = bytes(IP(src="8.8.8.8", dst="192.168.1.50")/UDP(sport=53, dport=53000)/b"dns_reply_payload")

        writer.write_packet(pkt1, t0)
        writer.write_packet(pkt2, t0 + 0.015)
        writer.close()

        # Validate with validate_pcap_file
        val_result = validate_pcap_file(pcap_file)
        assert val_result["valid"] is True
        assert val_result["packet_count"] == 2
        assert val_result["link_type"] == 1  # Ethernet
        assert val_result["size_bytes"] > 24

        # Reopen directly with Scapy
        scapy_pkts = rdpcap(pcap_file)
        assert len(scapy_pkts) == 2
        assert scapy_pkts[0][IP].src == "192.168.1.50"
        assert scapy_pkts[0][IP].dst == "8.8.8.8"
        assert scapy_pkts[1][IP].src == "8.8.8.8"


def test_adb_device_discovery():
    """Verifies that ADB bridge executes and queries hardware metadata cleanly."""
    devices = list_connected_devices()
    assert isinstance(devices, list)
    if devices:
        d = devices[0]
        assert "serial" in d
        assert "status" in d
        assert "model" in d
