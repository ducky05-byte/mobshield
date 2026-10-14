"""Focused unit and integration tests for Phase 2 Live Packet Capture & Flow Engine.

Covers:
- Bidirectional flow aggregation
- TCP and UDP packet processing
- Feature schema completeness (all 65 features)
- Saved-model loading & inference from live extracted flow features
- Start/stop lifecycle of LiveCapturePipeline
- Graceful handling of invalid inputs and empty captures
"""

import time
import json
import pytest
from pathlib import Path
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Packet, Raw

from src.prepare_dataset import NetworkFlowPreprocessor
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)

from src.flow_engine import BidirectionalFlow, LiveCapturePipeline
from src.predict_flow import FlowPredictor


@pytest.fixture
def feature_names():
    with open("models/feature_names.json", "r") as f:
        return json.load(f)


@pytest.fixture
def synthetic_tcp_packets():
    """Generates synthetic TCP 3-way handshake + data exchange packets."""
    base_time = 1700000000.0
    
    # 1. Client -> Server SYN
    p1 = IP(src="192.168.137.45", dst="142.250.190.46")/TCP(sport=50123, dport=443, flags="S", window=65535)
    p1.time = base_time

    # 2. Server -> Client SYN-ACK
    p2 = IP(src="142.250.190.46", dst="192.168.137.45")/TCP(sport=443, dport=50123, flags="SA", window=65535)
    p2.time = base_time + 0.020

    # 3. Client -> Server ACK
    p3 = IP(src="192.168.137.45", dst="142.250.190.46")/TCP(sport=50123, dport=443, flags="A", window=65535)
    p3.time = base_time + 0.025

    # 4. Client -> Server Data (TLS Client Hello, payload 512 bytes)
    p4 = IP(src="192.168.137.45", dst="142.250.190.46")/TCP(sport=50123, dport=443, flags="PA")/Raw(load=b"X"*512)
    p4.time = base_time + 0.035

    # 5. Server -> Client Data (TLS Server Hello, payload 1400 bytes)
    p5 = IP(src="142.250.190.46", dst="192.168.137.45")/TCP(sport=443, dport=50123, flags="PA")/Raw(load=b"Y"*1400)
    p5.time = base_time + 0.055

    return [p1, p2, p3, p4, p5]


@pytest.fixture
def synthetic_udp_packets():
    """Generates synthetic UDP DNS query and reply."""
    base_time = 1700000100.0
    
    # Client -> DNS Resolver
    p1 = IP(src="192.168.137.45", dst="8.8.8.8")/UDP(sport=53001, dport=53)/b"dns_query_packet"
    p1.time = base_time

    # DNS Resolver -> Client
    p2 = IP(src="8.8.8.8", dst="192.168.137.45")/UDP(sport=53, dport=53001)/b"dns_response_packet_longer"
    p2.time = base_time + 0.012

    return [p1, p2]


def test_tcp_flow_aggregation(synthetic_tcp_packets, feature_names):
    """Verifies that bidirectional TCP packets correctly aggregate forward and backward statistics."""
    p_init = synthetic_tcp_packets[0]
    key = ("192.168.137.45", "142.250.190.46", 50123, 443, 6)
    flow = BidirectionalFlow(p_init, key)

    for pkt in synthetic_tcp_packets[1:]:
        flow.add_packet(pkt)

    assert flow.fwd_packets == 3  # SYN, ACK, Data
    assert flow.bwd_packets == 2  # SYN-ACK, Data
    assert flow.syn_flags == 2
    assert flow.fin_flags == 0
    assert flow.psh_flags == 2
    assert flow.init_win_fwd == 65535
    assert flow.init_win_bwd == 65535

    feat_dict = flow.to_feature_dict()

    # Verify all 65 features are present
    for fname in feature_names:
        assert fname in feat_dict, f"Extracted feature dict missing feature: {fname}"

    # Verify non-trivial values
    assert feat_dict["Total Fwd Packets"] == 3.0
    assert feat_dict["Total Backward Packets"] == 2.0
    assert feat_dict["Flow Duration"] > 0


def test_udp_flow_aggregation(synthetic_udp_packets, feature_names):
    """Verifies that UDP packets aggregate correctly without TCP flag errors."""
    p_init = synthetic_udp_packets[0]
    key = ("192.168.137.45", "8.8.8.8", 53001, 53, 17)
    flow = BidirectionalFlow(p_init, key)

    flow.add_packet(synthetic_udp_packets[1])

    assert flow.fwd_packets == 1
    assert flow.bwd_packets == 1
    assert flow.syn_flags == 0
    assert flow.ack_flags == 0

    feat_dict = flow.to_feature_dict()
    for fname in feature_names:
        assert fname in feat_dict, f"Missing feature: {fname}"


def test_live_features_inference_compatibility(synthetic_tcp_packets):
    """Verifies that the live-extracted feature dict runs successfully through the saved model."""
    predictor = FlowPredictor()
    p_init = synthetic_tcp_packets[0]
    key = ("192.168.137.45", "142.250.190.46", 50123, 443, 6)
    flow = BidirectionalFlow(p_init, key)
    for pkt in synthetic_tcp_packets[1:]:
        flow.add_packet(pkt)

    feat_dict = flow.to_feature_dict()

    # Binary prediction
    bin_res = predictor.predict_flow(feat_dict, mode="binary")
    assert bin_res["label"] in ("BENIGN", "MALWARE_ASSOCIATED")
    assert 0.0 <= bin_res["confidence"] <= 1.0
    assert sum(bin_res["probabilities"].values()) == pytest.approx(1.0, 0.01)

    # Multiclass prediction
    multi_res = predictor.predict_flow(feat_dict, mode="multiclass")
    assert multi_res["label"] in ("BENIGN", "ADWARE", "RANSOMWARE", "SCAREWARE", "SMSMALWARE")
    assert 0.0 <= multi_res["confidence"] <= 1.0


def test_flow_expiration_logic(synthetic_tcp_packets):
    """Verifies flow expiration after idle or active timeout."""
    p_init = synthetic_tcp_packets[0]
    key = ("192.168.137.45", "142.250.190.46", 50123, 443, 6)
    flow = BidirectionalFlow(p_init, key)

    # Within limits
    assert not flow.is_expired(now=p_init.time + 1.0, active_timeout=15.0, idle_timeout=5.0)

    # Exceed idle timeout
    assert flow.is_expired(now=p_init.time + 6.0, active_timeout=15.0, idle_timeout=5.0)

    # Exceed active timeout
    assert flow.is_expired(now=p_init.time + 20.0, active_timeout=15.0, idle_timeout=5.0)


def test_pipeline_lifecycle_and_packet_injection(synthetic_tcp_packets):
    """Tests starting, injecting synthetic packets, flushing, and stopping the LiveCapturePipeline."""
    classified_records = []

    def on_flow(rec):
        classified_records.append(rec)

    # Using dummy loopback interface name for pipeline test
    pipeline = LiveCapturePipeline(
        interface="127.0.0.1",
        target_ip=None,
        active_timeout=10.0,
        idle_timeout=2.0,
        binary_threshold=0.65,
        on_flow_classified=on_flow
    )

    # Test processing packet directly via internal handler
    pipeline.is_running = True
    for pkt in synthetic_tcp_packets:
        pipeline._process_packet(pkt)

    telemetry = pipeline.get_telemetry()
    assert telemetry["packets_captured"] == 5
    assert telemetry["active_flows_tracking"] == 1

    # Force flow flush
    pipeline.stop()
    assert len(classified_records) == 1
    rec = classified_records[0]
    assert rec["src_ip"] == "192.168.137.45"
    assert rec["dst_ip"] == "142.250.190.46"
    assert rec["protocol"] == "TCP"
    assert rec["packets"] == 5
    assert rec["status"] in ("BENIGN", "MALWARE_ASSOCIATED")


def test_prediction_error_handling():
    """Ensures predictor gracefully raises on invalid prediction modes or non-mapping inputs."""
    predictor = FlowPredictor()
    
    # Invalid mode
    with pytest.raises(ValueError):
        predictor.predict_flow({}, mode="invalid_mode")

    # Invalid input type
    with pytest.raises(Exception):
        predictor.predict_flow(12345, mode="binary")
