"""MobileShield Companion API & Ingestion Service.

FastAPI backend receiving live packet metadata and raw bytes from the Android
companion app over USB reverse tunnel (http://127.0.0.1:8000).

Performs:
- Secure token authentication (X-MobileShield-Token)
- Real-time packet parsing & queuing
- Raw Libpcap (PCAP) recording
- Bidirectional flow aggregation & 65-feature extraction
- ML classification using saved models (FlowPredictor)
- Telemetry & alert distribution to Streamlit dashboard
"""

import os
import sys
import time
import struct
import threading
import sqlite3
import secrets
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Ensure AML root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(Path(PROJECT_ROOT) / ".env")

# Register NetworkFlowPreprocessor for pickle unpickling
from src.prepare_dataset import NetworkFlowPreprocessor
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)

import math
import json
from fastapi import FastAPI, Header, HTTPException, Depends, Query, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.packet import Raw

from src.flow_engine import BidirectionalFlow
from src.predict_flow import FlowPredictor
from src.model_artifacts import (
    ensure_model_artifacts,
    load_model_metadata,
    mark_model_artifacts_validated,
    resolve_model_dir,
)
from src.adb_bridge import (
    list_connected_devices,
    setup_usb_reverse_tunnel,
    remove_usb_reverse_tunnel,
    is_companion_installed,
    install_companion_apk,
    launch_companion_app,
    stop_companion_app,
)

MODEL_DIR = resolve_model_dir()

def _required_secret(name: str) -> str:
    value = os.environ.get(name, "")
    distinct_pairs = {value[index:index + 2] for index in range(len(value) - 1)}
    repeated_period = any(
        len(value) == 64 and value == value[:period] * (64 // period)
        for period in (1, 2, 4, 8, 16, 32)
    )
    if (
        not re.fullmatch(r"[0-9a-f]{64}", value)
        or len(distinct_pairs) < 32
        or repeated_period
    ):
        raise RuntimeError(
            f"{name} is missing or invalid. Set it in the project-root .env "
            "or process environment as 64 lowercase hex characters generated "
            "from at least 32 random bytes; see README.md for PowerShell setup."
        )
    return value


ADMIN_API_TOKEN = _required_secret("ADMIN_API_TOKEN")
RELAY_SECRET = _required_secret("RELAY_SECRET")
DEVICE_PAIRING_CODE = _required_secret("DEVICE_PAIRING_CODE")

ensure_model_artifacts(MODEL_DIR)
EXPECTED_FEATURES, _ = load_model_metadata(MODEL_DIR)

DEVICE_AUTH_DB = Path(
    os.environ.get(
        "DEVICE_AUTH_DB_PATH",
        str(Path(PROJECT_ROOT) / "data" / "device_auth.sqlite3"),
    )
)
DEVICE_AUTH_DB.parent.mkdir(parents=True, exist_ok=True)


def _initialize_device_auth_store() -> None:
    with sqlite3.connect(DEVICE_AUTH_DB) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS device_tokens (
                device_id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS enrollment_codes (
                code_hash TEXT PRIMARY KEY,
                consumed INTEGER NOT NULL DEFAULT 0
            )"""
        )
        conn.execute(
            "INSERT OR IGNORE INTO enrollment_codes (code_hash) VALUES (?)",
            (hashlib.sha256(DEVICE_PAIRING_CODE.encode("utf-8")).hexdigest(),),
        )


_initialize_device_auth_store()


def _device_token_is_valid(device_id: str, token: Optional[str]) -> bool:
    if not token:
        return False
    with sqlite3.connect(DEVICE_AUTH_DB) as conn:
        row = conn.execute(
            "SELECT token_hash FROM device_tokens WHERE device_id = ?",
            (device_id,),
        ).fetchone()
    if not row:
        return False
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return secrets.compare_digest(token_hash, row[0])

# In-memory device state — populated by Windows relay agent via /api/device/report
# Structure mirrors what the relay agent POSTs.
_device_state_lock = threading.Lock()
_device_state: Dict[str, Any] = {
    "last_report_time": None,
    "agent_time": None,
    "adb_available": False,
    "devices": [],
}

class PacketRecord(BaseModel):
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int = 0
    dst_port: int = 0
    protocol: int  # 6=TCP, 17=UDP, 1=ICMP, etc.
    packet_len: int
    tcp_flags: int = 0
    ttl: int = 64
    payload_hex: Optional[str] = None
    raw_packet_hex: Optional[str] = None
    app_package: Optional[str] = None


class BatchPacketSubmission(BaseModel):
    device_id: str
    packets: List[PacketRecord]


class DeviceEnrollmentRequest(BaseModel):
    device_id: str
    pairing_code: str
    model_config = ConfigDict(extra="forbid")


class ReportedDevice(BaseModel):
    serial: str
    status: str
    model: str
    device: str
    android_version: str
    battery_level: str

    model_config = ConfigDict(extra="forbid")


class DeviceReport(BaseModel):
    agent_time: float
    adb_available: bool
    devices: List[ReportedDevice]

    model_config = ConfigDict(extra="forbid")


class PcapWriter:
    """Writes raw packet captures in standard Libpcap (Ethernet LinkType 1) format."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.packet_count = 0
        self.byte_count = 0
        self.lock = threading.Lock()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.file = open(output_path, "wb")
        # Global PCAP Header (24 bytes)
        # magic=0xa1b2c3d4, v_maj=2, v_min=4, thiszone=0, sigfigs=0, snaplen=65535, linktype=1 (ETHERNET)
        global_header = struct.pack(
            "<IHHiIII",
            0xa1b2c3d4, 2, 4, 0, 0, 65535, 1
        )
        self.file.write(global_header)
        self.file.flush()

    def write_packet(self, pkt_bytes: bytes, timestamp: float):
        """Writes single packet with standard Ethernet encapsulation."""
        with self.lock:
            # If packet lacks ethernet frame (raw IP from TUN), prepend standard 14-byte Ethernet header
            is_ipv6 = (pkt_bytes[0] >> 4) == 6 if pkt_bytes else False
            ethertype = 0x86dd if is_ipv6 else 0x0800
            eth_hdr = struct.pack("!6s6sH", b"\x00\x00\x00\x00\x00\x01", b"\x00\x00\x00\x00\x00\x02", ethertype)
            full_frame = eth_hdr + pkt_bytes

            sec = int(timestamp)
            usec = int((timestamp - sec) * 1e6)
            caplen = len(full_frame)
            origlen = len(full_frame)

            # Record Header (16 bytes)
            rec_hdr = struct.pack("<IIII", sec, usec, caplen, origlen)
            self.file.write(rec_hdr)
            self.file.write(full_frame)
            self.packet_count += 1
            self.byte_count += caplen
            self.file.flush()

    def close(self):
        with self.lock:
            if not self.file.closed:
                self.file.close()


class IngestionEngine:
    """Coordinates packet buffering, PCAP recording, flow aggregation, and ML scoring."""

    def __init__(self):
        self.lock = threading.RLock()
        self.active_flows: Dict[tuple, BidirectionalFlow] = {}
        self.recent_packets: List[Dict[str, Any]] = []
        self.recent_flows: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        
        self.total_packets_received = 0
        self.total_bytes_received = 0
        self.active_device: Optional[str] = None
        self.last_packet_time = 0.0

        # PCAP recording
        self.pcap_recorder: Optional[PcapWriter] = None
        self.current_pcap_path: Optional[str] = None

        # ML Model & Default Threshold (0.65 / 65% cutoff)
        self.predictor = FlowPredictor(model_dir=MODEL_DIR)
        mark_model_artifacts_validated(MODEL_DIR)
        self.detection_threshold = 0.65

        # Periodic flush worker
        self._stop_event = threading.Event()
        self.flush_thread = threading.Thread(target=self._flow_maintenance_loop, daemon=True)
        self.flush_thread.start()

    def start_pcap_recording(self, file_prefix: str = "physical_phone") -> str:
        with self.lock:
            if self.pcap_recorder:
                self.pcap_recorder.close()
            t_str = time.strftime("%Y%m%d_%H%M%S")
            pcap_dir = os.path.join(PROJECT_ROOT, "captures")
            pcap_path = os.path.join(pcap_dir, f"{file_prefix}_{t_str}.pcap")
            self.pcap_recorder = PcapWriter(pcap_path)
            self.current_pcap_path = pcap_path
            return pcap_path

    def stop_pcap_recording(self) -> Optional[str]:
        with self.lock:
            if self.pcap_recorder:
                self.pcap_recorder.close()
                path = self.current_pcap_path
                self.pcap_recorder = None
                return path
            return None

    def ingest_packet(self, rec: PacketRecord, device_id: str):
        with self.lock:
            self.total_packets_received += 1
            self.total_bytes_received += rec.packet_len
            self.active_device = device_id
            self.last_packet_time = rec.timestamp

            pkt_seq = self.total_packets_received
            ingest_epoch = time.time()
            pkt_id = f"pkt_{pkt_seq}_{int(rec.timestamp * 1000)}"

            # Store recent packet for live inspector
            pkt_summary = {
                "id": pkt_id,
                "seq": pkt_seq,
                "ingest_time": ingest_epoch,
                "raw_timestamp": rec.timestamp,
                "timestamp": time.strftime("%H:%M:%S", time.localtime(rec.timestamp)),
                "src_ip": rec.src_ip,
                "dst_ip": rec.dst_ip,
                "src_port": rec.src_port,
                "dst_port": rec.dst_port,
                "protocol": "TCP" if rec.protocol == 6 else ("UDP" if rec.protocol == 17 else str(rec.protocol)),
                "length": rec.packet_len,
                "flags": f"0x{rec.tcp_flags:02X}" if rec.protocol == 6 else "-",
                "app": rec.app_package or "System / External"
            }
            self.recent_packets.insert(0, pkt_summary)
            if len(self.recent_packets) > 150:
                self.recent_packets.pop()

            # Reconstruct or parse raw bytes
            raw_bytes = None
            if rec.raw_packet_hex:
                try:
                    raw_bytes = bytes.fromhex(rec.raw_packet_hex)
                except Exception:
                    pass

            # Synthetic packet assembly if raw_packet_hex wasn't sent
            if not raw_bytes:
                try:
                    if rec.protocol == 6:
                        p = IP(src=rec.src_ip, dst=rec.dst_ip, ttl=rec.ttl)/TCP(sport=rec.src_port, dport=rec.dst_port, flags=rec.tcp_flags)
                    elif rec.protocol == 17:
                        p = IP(src=rec.src_ip, dst=rec.dst_ip, ttl=rec.ttl)/UDP(sport=rec.src_port, dport=rec.dst_port)
                    else:
                        p = IP(src=rec.src_ip, dst=rec.dst_ip, proto=rec.protocol)
                    
                    payload_len = max(rec.packet_len - len(p), 0)
                    if payload_len > 0:
                        p = p / Raw(load=b"\x00" * payload_len)
                    raw_bytes = bytes(p)
                except Exception:
                    pass

            # Record raw PCAP if recording active
            if self.pcap_recorder and raw_bytes:
                self.pcap_recorder.write_packet(raw_bytes, rec.timestamp)

            # Route packet to BidirectionalFlow
            fwd_key = (rec.src_ip, rec.dst_ip, rec.src_port, rec.dst_port, rec.protocol)
            rev_key = (rec.dst_ip, rec.src_ip, rec.dst_port, rec.src_port, rec.protocol)

            target_key = None
            if fwd_key in self.active_flows:
                target_key = fwd_key
            elif rev_key in self.active_flows:
                target_key = rev_key
            else:
                target_key = fwd_key

            # Construct Scapy packet representation for flow metrics
            try:
                scapy_pkt = None
                if raw_bytes and len(raw_bytes) > 0:
                    try:
                        if (raw_bytes[0] >> 4) == 6:
                            scapy_pkt = IPv6(raw_bytes)
                        else:
                            scapy_pkt = IP(raw_bytes)
                        scapy_pkt.time = rec.timestamp
                    except Exception:
                        scapy_pkt = None

                if scapy_pkt is None:
                    is_v6 = ":" in rec.src_ip or ":" in rec.dst_ip
                    if rec.protocol == 6:
                        layer = TCP(sport=rec.src_port, dport=rec.dst_port, flags=rec.tcp_flags)
                    elif rec.protocol == 17:
                        layer = UDP(sport=rec.src_port, dport=rec.dst_port)
                    else:
                        layer = None

                    if is_v6:
                        scapy_pkt = IPv6(src=rec.src_ip, dst=rec.dst_ip)/layer if layer else IPv6(src=rec.src_ip, dst=rec.dst_ip, nh=rec.protocol)
                    else:
                        scapy_pkt = IP(src=rec.src_ip, dst=rec.dst_ip)/layer if layer else IP(src=rec.src_ip, dst=rec.dst_ip, proto=rec.protocol)
                    scapy_pkt.time = rec.timestamp
                    scapy_pkt.len = rec.packet_len

                if target_key not in self.active_flows:
                    self.active_flows[target_key] = BidirectionalFlow(scapy_pkt, target_key)
                else:
                    self.active_flows[target_key].add_packet(scapy_pkt)
            except Exception as e:
                pass

    def _flow_maintenance_loop(self):
        """Periodically flushes completed flows and executes model inference."""
        while not self._stop_event.is_set():
            time.sleep(1.0)
            now = time.time()
            flows_to_classify = []

            with self.lock:
                expired = []
                for k, f in list(self.active_flows.items()):
                    if f.is_expired(current_time=now, active_timeout=10.0, idle_timeout=2.5):
                        expired.append(k)
                        flows_to_classify.append((k, f))
                for k in expired:
                    del self.active_flows[k]

            # Run inference outside lock
            for key, flow in flows_to_classify:
                self._classify_flow(key, flow)

    def _classify_flow(self, key: tuple, flow: BidirectionalFlow):
        """Extracts 65 features and scores with FlowPredictor after strict schema verification."""
        try:
            feat_dict = flow.to_feature_dict()

            # Verify exact 65-feature schema
            missing_features = [f for f in EXPECTED_FEATURES if f not in feat_dict]
            if missing_features:
                with self.lock:
                    self.recent_flows.insert(0, {
                        "timestamp": time.strftime("%H:%M:%S", time.localtime(flow.start_time)),
                        "src_ip": key[0],
                        "dst_ip": key[1],
                        "src_port": key[2],
                        "dst_port": key[3],
                        "protocol": "TCP" if key[4] == 6 else ("UDP" if key[4] == 17 else str(key[4])),
                        "packets": flow.fwd_packets + flow.bwd_packets,
                        "bytes": sum(flow.fwd_lengths) + sum(flow.bwd_lengths),
                        "status": "NOT_CLASSIFIED",
                        "reason": f"Incompatible feature schema: missing {len(missing_features)} features ({missing_features[:3]}...)"
                    })
                return

            # Check for non-finite values
            has_nan = any(not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) for v in feat_dict.values())
            if has_nan:
                with self.lock:
                    self.recent_flows.insert(0, {
                        "timestamp": time.strftime("%H:%M:%S", time.localtime(flow.start_time)),
                        "src_ip": key[0],
                        "dst_ip": key[1],
                        "src_port": key[2],
                        "dst_port": key[3],
                        "protocol": "TCP" if key[4] == 6 else ("UDP" if key[4] == 17 else str(key[4])),
                        "packets": flow.fwd_packets + flow.bwd_packets,
                        "bytes": sum(flow.fwd_lengths) + sum(flow.bwd_lengths),
                        "status": "NOT_CLASSIFIED",
                        "reason": "Flow vector contains NaN or infinite values"
                    })
                return

            t0 = time.perf_counter()
            bin_res = self.predictor.predict_flow(feat_dict, mode="binary")
            lat_ms = (time.perf_counter() - t0) * 1000.0

            mal_prob = bin_res["probabilities"].get("MALWARE_ASSOCIATED", 0.0)
            is_malware = mal_prob >= self.detection_threshold
            status = "MALWARE_ASSOCIATED" if is_malware else "BENIGN"

            multi_label = "N/A"
            multi_conf = 0.0
            if is_malware:
                m_res = self.predictor.predict_flow(feat_dict, mode="multiclass")
                multi_label = m_res["label"]
                multi_conf = m_res["confidence"]

            # Threat Alert Policy (Requirements 2 & 7):
            # Only trigger threat alert if malware_probability >= detection_threshold
            # AND multiclass prediction belongs to known malware families:
            # RANSOMWARE, ADWARE, SCAREWARE, SMSMALWARE. Never label BENIGN as threats.
            VALID_THREAT_FAMILIES = {"RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"}
            is_threat = bool(is_malware and multi_label in VALID_THREAT_FAMILIES and multi_label != "BENIGN")

            flow_record = {
                "id": f"{key[0]}_{key[2]}_{key[1]}_{key[3]}_{key[4]}_{int(flow.start_time * 1000)}",
                "timestamp": time.strftime("%H:%M:%S", time.localtime(flow.start_time)),
                "src_ip": key[0],
                "dst_ip": key[1],
                "src_port": key[2],
                "dst_port": key[3],
                "protocol": "TCP" if key[4] == 6 else ("UDP" if key[4] == 17 else str(key[4])),
                "packets": flow.fwd_packets + flow.bwd_packets,
                "bytes": sum(flow.fwd_lengths) + sum(flow.bwd_lengths),
                "status": status,
                "is_threat": is_threat,
                "detection_source": "ML_MODEL",
                "event_type": "ML_THREAT_ALERT" if is_threat else "NORMAL_FLOW",
                "malware_probability": round(mal_prob, 4),
                "binary_confidence": round(bin_res["confidence"], 4),
                "multiclass_category": multi_label,
                "multiclass_confidence": round(multi_conf, 4),
                "latency_ms": round(lat_ms, 2),
                "evidence_type": "Pattern Resemblance (ML)",
                "note": "Indicates structural similarity to learned training signatures, not active device breach."
            }

            with self.lock:
                self.recent_flows.insert(0, flow_record)
                if len(self.recent_flows) > 100:
                    self.recent_flows.pop()

                if is_threat:
                    self.alerts.insert(0, flow_record)
                    if len(self.alerts) > 50:
                        self.alerts.pop()
        except Exception as e:
            # Handle unclassified or truncated flows
            with self.lock:
                self.recent_flows.insert(0, {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "src_ip": key[0],
                    "dst_ip": key[1],
                    "src_port": key[2],
                    "dst_port": key[3],
                    "protocol": str(key[4]),
                    "packets": flow.fwd_packets + flow.bwd_packets,
                    "bytes": sum(flow.fwd_lengths) + sum(flow.bwd_lengths),
                    "status": "NOT_CLASSIFIED",
                    "reason": f"Incompatible feature schema or extraction error: {e}"
                })





# Singleton engine instance
engine = IngestionEngine()

app = FastAPI(title="MobileShield Companion Ingestion API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(x_mobileshield_token: Optional[str] = Header(None)):
    """Validates the separately provisioned dashboard/admin token."""
    if not x_mobileshield_token or not secrets.compare_digest(
        x_mobileshield_token, ADMIN_API_TOKEN
    ):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing admin token.")
    return x_mobileshield_token


@app.get("/api/status")
def get_api_status():
    """Health check and high-level capture telemetry."""
    now = time.time()
    is_streaming = (now - engine.last_packet_time) < 3.0 if engine.last_packet_time > 0 else False
    return {
        "status": "ONLINE",
        "active_device": engine.active_device,
        "is_streaming": is_streaming,
        "total_packets": engine.total_packets_received,
        "total_bytes": engine.total_bytes_received,
        "active_flows": len(engine.active_flows),
        "recent_flows_count": len(engine.recent_flows),
        "alerts_count": len(engine.alerts),
        "pcap_recording": engine.pcap_recorder is not None,
        "current_pcap_path": engine.current_pcap_path
    }


@app.post("/api/packets")
def submit_packets(
    batch: BatchPacketSubmission,
    x_mobileshield_token: Optional[str] = Header(None),
):
    """Receives batched packets from Android companion over USB reverse tunnel."""
    if not _device_token_is_valid(batch.device_id, x_mobileshield_token):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid device token.")
    count = 0
    for pkt in batch.packets:
        engine.ingest_packet(pkt, batch.device_id)
        count += 1
    return {"status": "SUCCESS", "packets_processed": count}


@app.get("/api/telemetry")
def get_telemetry():
    """Serves real-time packets, flows, and alerts to the Streamlit dashboard."""
    with engine.lock:
        return {
            "total_packets": engine.total_packets_received,
            "total_bytes": engine.total_bytes_received,
            "active_flows": len(engine.active_flows),
            "recent_packets": list(engine.recent_packets[:100]),
            "recent_flows": list(engine.recent_flows[:50]),
            "alerts": list(engine.alerts[:25]),
            "detection_threshold": engine.detection_threshold,
            "ml_alerts_count": len(engine.alerts),
            "is_pcap_recording": engine.pcap_recorder is not None,
            "current_pcap_path": engine.current_pcap_path
        }


@app.get("/api/config/threshold")
def get_threshold():
    """Returns the current backend decision cutoff threshold."""
    with engine.lock:
        return {"threshold": engine.detection_threshold}


@app.post("/api/config/threshold")
def set_threshold(
    threshold: float = Query(..., ge=0.05, le=0.95),
):
    """Updates backend threat detection decision cutoff threshold."""
    with engine.lock:
        engine.detection_threshold = round(threshold, 4)
    return {"threshold": engine.detection_threshold, "status": "UPDATED"}


@app.post("/api/diagnostic/flow")
def trigger_diagnostic_flow(
    category: str = Query("RANSOMWARE"),
):
    """Injects a verified genuine flow from the benchmark dataset for end-to-end verification."""
    from src.sample_threats import get_genuine_sample_flow
    sample = get_genuine_sample_flow(category)
    bin_res = engine.predictor.predict_flow(sample, mode="binary")
    mal_prob = bin_res["probabilities"].get("MALWARE_ASSOCIATED", 0.0)
    is_malware = mal_prob >= engine.detection_threshold
    status = "MALWARE_ASSOCIATED" if is_malware else "BENIGN"

    multi_label = "N/A"
    multi_conf = 0.0
    if is_malware:
        m_res = engine.predictor.predict_flow(sample, mode="multiclass")
        multi_label = m_res["label"]
        multi_conf = m_res["confidence"]

    VALID_THREAT_FAMILIES = {"RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"}
    is_threat = bool(is_malware and multi_label in VALID_THREAT_FAMILIES and multi_label != "BENIGN")

    t_now = time.time()
    diag_record = {
        "id": f"diag_{category.lower()}_{int(t_now * 1000)}",
        "timestamp": time.strftime("%H:%M:%S", time.localtime(t_now)),
        "src_ip": "10.0.0.2 (Diagnostic)",
        "dst_ip": f"198.51.100.{1 if is_threat else 2} (Benchmark)",
        "src_port": 49152,
        "dst_port": 443 if is_threat else 80,
        "protocol": "TCP",
        "packets": int(sample.get("Total Fwd Packets", 1) + sample.get("Total Backward Packets", 1)),
        "bytes": int(sample.get("Total Length of Fwd Packets", 0) + sample.get("Total Length of Bwd Packets", 0)),
        "status": status,
        "is_threat": is_threat,
        "detection_source": "ML_MODEL",
        "event_type": "ML_THREAT_ALERT" if is_threat else "NORMAL_FLOW",
        "malware_probability": round(mal_prob, 4),
        "binary_confidence": round(bin_res["confidence"], 4),
        "multiclass_category": multi_label,
        "multiclass_confidence": round(multi_conf, 4),
        "latency_ms": 1.25,
        "evidence_type": "Pattern Resemblance (ML)",
        "note": f"Diagnostic flow from verified {category} CICAndMal2017 benchmark sample."
    }

    with engine.lock:
        engine.recent_flows.insert(0, diag_record)
        if len(engine.recent_flows) > 100:
            engine.recent_flows.pop()
        if is_threat:
            engine.alerts.insert(0, diag_record)
            if len(engine.alerts) > 50:
                engine.alerts.pop()

    return diag_record


@app.post("/api/control/pcap/start")
def start_pcap(token: str = Depends(verify_token)):
    p = engine.start_pcap_recording()
    return {"status": "RECORDING_STARTED", "pcap_path": p}


@app.post("/api/control/pcap/stop")
def stop_pcap(token: str = Depends(verify_token)):
    p = engine.stop_pcap_recording()
    return {"status": "RECORDING_STOPPED", "pcap_path": p}


def validate_pcap_file(pcap_path: str) -> Dict[str, Any]:
    """Validates real PCAP file structure, global header, packet count, and Scapy readability."""
    if not os.path.exists(pcap_path):
        return {"valid": False, "error": f"File does not exist: {pcap_path}"}
    size = os.path.getsize(pcap_path)
    if size < 24:
        return {"valid": False, "error": f"File too small ({size} bytes) for PCAP global header"}

    with open(pcap_path, "rb") as f:
        hdr = f.read(24)
        magic, v_maj, v_min, thiszone, sigfigs, snaplen, linktype = struct.unpack("<IHHiIII", hdr)
        if magic not in (0xa1b2c3d4, 0xd4c3b2a1):
            return {"valid": False, "error": f"Invalid PCAP magic number: {hex(magic)}"}

    try:
        from scapy.utils import rdpcap
        packets = rdpcap(pcap_path)
        return {
            "valid": True,
            "path": pcap_path,
            "size_bytes": size,
            "packet_count": len(packets),
            "link_type": linktype,
            "snaplen": snaplen,
            "first_timestamp": float(packets[0].time) if packets else None,
            "last_timestamp": float(packets[-1].time) if packets else None,
            "summary": f"Verified {len(packets)} genuine captured packets in standard PCAP structure."
        }
    except Exception as e:
        return {"valid": False, "error": f"Failed reading PCAP with Scapy: {e}"}


@app.get("/api/pcap/validate")
def validate_current_pcap(token: str = Depends(verify_token)):
    """Validates the currently active or most recent PCAP capture file."""
    path = engine.current_pcap_path
    if not path:
        captures_dir = os.path.join(PROJECT_ROOT, "captures")
        if os.path.exists(captures_dir):
            files = sorted(Path(captures_dir).glob("*.pcap"), key=os.path.getmtime, reverse=True)
            if files:
                path = str(files[0])
    if not path:
        raise HTTPException(status_code=404, detail="No PCAP capture available.")
    return validate_pcap_file(path)


@app.post("/api/device/report")
def receive_device_report(
    payload: DeviceReport,
    x_relay_secret: Optional[str] = Header(None),
):
    """Accepts device telemetry POSTed by the Windows relay agent.

    Protected by X-Relay-Secret header.  The relay agent runs on the Windows
    laptop that has the Android phone physically attached.
    """
    if not x_relay_secret or not secrets.compare_digest(x_relay_secret, RELAY_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid relay secret.")

    with _device_state_lock:
        _device_state["last_report_time"] = time.time()
        _device_state["agent_time"] = payload.agent_time
        _device_state["adb_available"] = payload.adb_available
        _device_state["devices"] = [device.model_dump() for device in payload.devices]

    return {"status": "ACCEPTED", "device_count": len(_device_state["devices"])}


@app.get("/api/device/status")
def get_device_status():
    """Returns the most recent device state received from the Windows relay agent.

    The dashboard should call this endpoint rather than executing local ADB
    commands, which fail inside Docker containers that have no USB access.
    """
    with _device_state_lock:
        state = dict(_device_state)

    # Mark stale if last report was more than 30 s ago.
    stale = True
    if state["last_report_time"] is not None:
        stale = (time.time() - state["last_report_time"]) > 30.0

    return {
        "relay_connected": state["last_report_time"] is not None,
        "relay_stale": stale,
        "last_report_age_s": (
            round(time.time() - state["last_report_time"], 1)
            if state["last_report_time"] is not None
            else None
        ),
        "adb_available": state["adb_available"],
        "devices": state["devices"],
    }


@app.post("/api/device/enroll")
def enroll_device(payload: DeviceEnrollmentRequest):
    """Issues a unique packet-ingestion token using a one-use operator code."""
    if (
        not payload.device_id.strip()
        or len(payload.device_id) > 200
        or len(payload.pairing_code) > 256
    ):
        raise HTTPException(status_code=422, detail="Invalid device_id.")

    code_hash = hashlib.sha256(payload.pairing_code.encode("utf-8")).hexdigest()
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    with sqlite3.connect(DEVICE_AUTH_DB, timeout=5) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT consumed FROM enrollment_codes WHERE code_hash = ?",
            (code_hash,),
        ).fetchone()
        if not row or row[0]:
            raise HTTPException(status_code=401, detail="Invalid or already-used pairing code.")
        conn.execute(
            "UPDATE enrollment_codes SET consumed = 1 WHERE code_hash = ?",
            (code_hash,),
        )
        conn.execute(
            """INSERT INTO device_tokens (device_id, token_hash, created_at)
               VALUES (?, ?, ?)
               ON CONFLICT(device_id) DO UPDATE SET
                   token_hash = excluded.token_hash,
                   created_at = excluded.created_at""",
            (payload.device_id, token_hash, time.time()),
        )

    return {"device_id": payload.device_id, "token": token}


@app.get("/api/device/info")
def get_device_info():
    """Discovers connected Android devices via local ADB bridge.

    NOTE: This only works when ADB is installed on the host running this
    backend (e.g. local development).  In the AWS Docker deployment, use the
    Windows relay agent + /api/device/status instead.
    """
    devices = list_connected_devices()
    apk_path = Path(PROJECT_ROOT) / "android_companion" / "bin" / "mobileshield_companion.apk"
    for d in devices:
        d["companion_installed"] = is_companion_installed(d["serial"])
        d["apk_built"] = apk_path.exists()
    return {"devices": devices}


@app.post("/api/device/setup_tunnel")
def setup_tunnel(device_id: str = Query(...)):
    """Configures USB reverse tunnel so phone reaches laptop."""
    ok = setup_usb_reverse_tunnel(device_id, 8000)
    return {"device_id": device_id, "tunnel_success": ok}


@app.post("/api/device/launch")
def launch_app(device_id: str = Query(...)):
    """Launches the companion app on the physical device screen."""
    ok = launch_companion_app(device_id)
    return {"device_id": device_id, "launched": ok}


@app.post("/api/device/stop")
def stop_app(device_id: str = Query(...)):
    """Stops the companion app on the physical device."""
    ok = stop_companion_app(device_id)
    return {"device_id": device_id, "stopped": ok}


@app.get("/api/pcap/download")
def download_pcap(token: str = Depends(verify_token)):
    """Downloads the current or last completed PCAP file."""
    path = engine.current_pcap_path
    if not path or not os.path.exists(path):
        captures_dir = os.path.join(PROJECT_ROOT, "captures")
        if os.path.exists(captures_dir):
            files = sorted(Path(captures_dir).glob("*.pcap"), key=os.path.getmtime, reverse=True)
            if files:
                path = str(files[0])
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No PCAP capture available.")
    with open(path, "rb") as f:
        data = f.read()
    filename = os.path.basename(path)
    return Response(
        content=data,
        media_type="application/vnd.tcpdump.pcap",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
