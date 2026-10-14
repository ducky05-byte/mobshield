"""MobileShield — Live Android Network Intrusion Detection & Telemetry Dashboard.

Enterprise cybersecurity monitoring console and real-time inference interface.
Integrates:
- Physical ADB Android device discovery (Samsung Galaxy A14 5G)
- USB reverse-tunnel control (adb reverse tcp:8000 tcp:8000)
- Live packet inspector (real captured Layer 3/4 packets)
- Bidirectional flow aggregation & 65-feature schema alignment
- Machine learning inference (Binary & Multiclass Random Forest classifiers)
- Libpcap (PCAP) recording, validation, and export controls
- Gateway adapter diagnostics and error telemetry
"""

import os
import sys
import time
import json
import html
import urllib.request
import urllib.error
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv.parser import parse_stream

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_dashboard_dotenv = {}
try:
    with open(Path(PROJECT_ROOT) / ".env", "r", encoding="utf-8") as env_file:
        for binding in parse_stream(env_file):
            if binding.key in {"API_BASE_URL", "ADMIN_API_TOKEN"}:
                _dashboard_dotenv[binding.key] = binding.value
except FileNotFoundError:
    pass

# Register NetworkFlowPreprocessor for joblib/pickle deserialization
from src.prepare_dataset import NetworkFlowPreprocessor
import __main__
__main__.NetworkFlowPreprocessor = NetworkFlowPreprocessor

import streamlit as st
from src.adb_bridge import (
    list_connected_devices,
    setup_usb_reverse_tunnel,
    remove_usb_reverse_tunnel,
    is_companion_installed,
    install_companion_apk,
    launch_companion_app,
    stop_companion_app
)
from src.capture_diagnostics import list_available_interfaces
from src.flow_engine import LiveCapturePipeline
from src.predict_flow import FlowPredictor

import os

API_BASE_URL = (
    os.getenv("API_BASE_URL")
    or _dashboard_dotenv.get("API_BASE_URL")
    or "http://127.0.0.1:8000"
).rstrip("/")
ADMIN_API_TOKEN = (
    os.getenv("ADMIN_API_TOKEN")
    or _dashboard_dotenv.get("ADMIN_API_TOKEN")
    or ""
)
_admin_token_pairs = {
    ADMIN_API_TOKEN[index:index + 2] for index in range(len(ADMIN_API_TOKEN) - 1)
}
_admin_token_repeats = any(
    ADMIN_API_TOKEN == ADMIN_API_TOKEN[:period] * (64 // period)
    for period in (1, 2, 4, 8, 16, 32)
)
if (
    not re.fullmatch(r"[0-9a-f]{64}", ADMIN_API_TOKEN)
    or len(_admin_token_pairs) < 32
    or _admin_token_repeats
):
    raise RuntimeError(
        "ADMIN_API_TOKEN is missing or invalid. Set it in the project-root "
        ".env or process environment as 64 random lowercase hexadecimal "
        "characters; see README.md for PowerShell setup."
    )

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="MobileShield | Android Network Security Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- ENTERPRISE CSS DESIGN SYSTEM -----------------
st.markdown("""
<style>
    /* Dark cybersecurity console theme */
    .stApp {
        background-color: #0b0f17;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }
    
    /* Top title & subtitle styling */
    .ms-header-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
        padding: 0;
    }
    .ms-header-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 2px;
        margin-bottom: 14px;
        letter-spacing: 0.01em;
    }

    /* Status Pills */
    .status-bar-container {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        padding: 8px 14px;
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        margin-bottom: 16px;
        font-size: 0.82rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .pill-online {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
    }
    .pill-offline {
        background-color: #374151;
        color: #9ca3af;
        border: 1px solid #4b5563;
    }
    .pill-active {
        background-color: #0c4a6e;
        color: #7dd3fc;
        border: 1px solid #0284c7;
    }
    .pill-idle {
        background-color: #1f2937;
        color: #94a3b8;
        border: 1px solid #374151;
    }
    .pill-threat {
        background-color: #450a0a;
        color: #fca5a5;
        border: 1px solid #dc2626;
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 12px 16px;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.01em;
    }
    .metric-hint {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Informational Panels */
    .context-panel {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-left: 3px solid #0284c7;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #cbd5e1;
        line-height: 1.45;
        margin-bottom: 14px;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d121c;
        border-right: 1px solid #1e293b;
    }
    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    .sidebar-device-card {
        background-color: #131b2a;
        border: 1px solid #243047;
        border-radius: 5px;
        padding: 10px 12px;
        margin-bottom: 10px;
        font-size: 0.8rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
    

    /* Code / Monospace */
    code, pre {
        font-family: "JetBrains Mono", Consolas, Menlo, monospace !important;
        font-size: 0.82rem !important;
    }

    /* Live Packet Table & Real-Time Highlighting */
    .pkt-table-wrap {
        width: 100%;
        overflow-x: auto;
        border: 1px solid #1f2937;
        border-radius: 6px;
        background-color: #0b0f19;
        margin-top: 8px;
        margin-bottom: 12px;
        max-height: 520px;
        overflow-y: auto;
    }
    .pkt-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        text-align: left;
    }
    .pkt-table th {
        background-color: #111827;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        padding: 9px 12px;
        border-bottom: 1px solid #1f2937;
        position: sticky;
        top: 0;
        z-index: 2;
    }
    .pkt-table td {
        padding: 7px 12px;
        border-bottom: 1px solid #161f30;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
        font-family: "JetBrains Mono", Consolas, Menlo, monospace;
    }
    
    /* Normal Row */
    .pkt-row-normal {
        background-color: #0d121d;
        color: #cbd5e1;
        border-left: 3px solid transparent;
        transition: background-color 0.15s ease;
    }
    .pkt-row-normal:hover {
        background-color: #172133;
    }

    /* Real-Time Red Highlighting for Newly Captured Packets */
    @keyframes pktCaptureHighlightFade {
        0% {
            background-color: #7f1d1d;
            border-left-color: #ef4444;
            color: #fef2f2;
        }
        55% {
            background-color: #7f1d1d;
            border-left-color: #ef4444;
            color: #fef2f2;
        }
        100% {
            background-color: #0d121d;
            border-left-color: transparent;
            color: #cbd5e1;
        }
    }
    .pkt-row-new {
        animation: pktCaptureHighlightFade 2.6s cubic-bezier(0.25, 1, 0.5, 1) forwards;
        border-left: 3px solid #ef4444;
    }
    .pkt-row-new td {
        color: inherit !important;
    }
    .pkt-badge-new {
        display: inline-block;
        padding: 1px 5px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-radius: 3px;
        background-color: #ef4444;
        color: #ffffff;
        margin-right: 6px;
    }
    .pkt-proto-pill {
        display: inline-block;
        padding: 1px 6px;
        border-radius: 3px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .pkt-proto-tcp {
        background-color: #0c4a6e;
        color: #7dd3fc;
        border: 1px solid #0284c7;
    }
    .pkt-proto-udp {
        background-color: #3b0764;
        color: #d8b4fe;
        border: 1px solid #9333ea;
    }
    .pkt-proto-other {
        background-color: #1f2937;
        color: #94a3b8;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes persistent application session variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "data_mode" not in st.session_state:
        st.session_state.data_mode = "IDLE"
    if "predictor" not in st.session_state:
        st.session_state.predictor = FlowPredictor()
    if "diagnostic_history" not in st.session_state:
        st.session_state.diagnostic_history = []
    if "pcap_validation_result" not in st.session_state:
        st.session_state.pcap_validation_result = None
    if "last_synced_threshold" not in st.session_state:
        st.session_state.last_synced_threshold = 0.65
    if "rendered_packets_seen" not in st.session_state:
        st.session_state.rendered_packets_seen = {}
    if "initial_packet_sync_done" not in st.session_state:
        st.session_state.initial_packet_sync_done = False
    if "max_rendered_packet_seq" not in st.session_state:
        st.session_state.max_rendered_packet_seq = 0


init_session_state()

def query_api(endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Helper to communicate with local FastAPI backend on 127.0.0.1:8000."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        req = urllib.request.Request(url, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("X-MobileShield-Token", ADMIN_API_TOKEN)
        
        post_bytes = None
        if data is not None:
            post_bytes = json.dumps(data).encode("utf-8")
        
        with urllib.request.urlopen(req, data=post_bytes, timeout=0.5) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def query_api_file(endpoint: str) -> bytes:
    """Fetches a protected binary API response using the admin credential."""
    req = urllib.request.Request(f"{API_BASE_URL}{endpoint}", method="GET")
    req.add_header("X-MobileShield-Token", ADMIN_API_TOKEN)
    with urllib.request.urlopen(req, timeout=8.0) as resp:
        if resp.status != 200:
            raise urllib.error.HTTPError(
                req.full_url, resp.status, "PCAP download failed", resp.headers, None
            )
        return resp.read()


auto_refresh_enabled = st.session_state.get("auto_refresh", True)


def _refresh_setting_changed() -> None:
    st.rerun(scope="app")


@st.fragment(run_every=1 if auto_refresh_enabled else None)
def render_dashboard() -> None:
    """Refresh live dashboard content without rerunning the whole page."""


# ----------------- DATA INGESTION STATE -----------------
    api_status = query_api("/api/status")
    telemetry_data = query_api("/api/telemetry")
    is_api_online = telemetry_data is not None
    is_streaming = api_status.get("is_streaming", False) if api_status else False
    is_pcap_recording = api_status.get("pcap_recording", False) if api_status else False

    # ----------------- SIDEBAR CONTROLS -----------------
    st.sidebar.markdown('<div class="ms-header-title" style="font-size:1.25rem;">MobileShield</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="ms-header-sub">Network Security Console</div>', unsafe_allow_html=True)

    # 1. Device Connection
    st.sidebar.markdown('<div class="sidebar-section-title">Physical Device</div>', unsafe_allow_html=True)

    # Fetch device state from the relay endpoint (works in Docker/EC2; no local ADB needed).
    # Falls back to local ADB discovery when relay has never reported (local dev mode).
    device_status_data = query_api("/api/device/status")
    target_device = None
    devices = []

    if device_status_data is not None:
        devices = device_status_data.get("devices", [])
        relay_connected = device_status_data.get("relay_connected", False)
        relay_stale = device_status_data.get("relay_stale", True)
        relay_age = device_status_data.get("last_report_age_s")
    else:
        # API offline — try local ADB as last resort (works only on Windows dev machine).
        try:
            devices = list_connected_devices()
        except Exception:
            devices = []
        relay_connected = False
        relay_stale = True
        relay_age = None

    if devices:
        dev_labels = [f"{d['model']} ({d['serial']})" for d in devices]
        sel_idx = st.sidebar.selectbox("Select Target Device", options=range(len(devices)), format_func=lambda i: dev_labels[i], label_visibility="collapsed")
        target_device = devices[sel_idx]
    
        st.sidebar.markdown(f"""
        <div class="sidebar-device-card">
            <div style="font-weight:600; color:#f8fafc;">{target_device.get('model')}</div>
            <div style="color:#94a3b8; font-size:0.75rem;">Serial: {target_device.get('serial')}</div>
            <div style="color:#94a3b8; font-size:0.75rem;">OS: Android {target_device.get('android_version')} &bull; Status: {target_device.get('status')}</div>
        </div>
        """, unsafe_allow_html=True)
    
        col_adb1, col_adb2 = st.sidebar.columns(2)
        if col_adb1.button("Setup Tunnel", width="stretch"):
            ok = setup_usb_reverse_tunnel(target_device["serial"], 8000)
            if ok:
                st.sidebar.success("Tunnel active: port 8000")
            else:
                st.sidebar.error("Failed to establish tunnel.")
            
        if col_adb2.button("Launch App", width="stretch"):
            ok = launch_companion_app(target_device["serial"])
            if ok:
                st.sidebar.success("App opened on device.")
            else:
                st.sidebar.error("Failed to launch app.")
            
        apk_file = Path(PROJECT_ROOT) / "android_companion" / "bin" / "mobileshield_companion.apk"
        if st.sidebar.button("Install / Update APK", width="stretch"):
            if apk_file.exists():
                with st.sidebar.spinner("Installing APK..."):
                    installed = install_companion_apk(target_device["serial"], str(apk_file))
                    if installed:
                        st.sidebar.success("APK installed successfully.")
                    else:
                        st.sidebar.error("Installation failed.")
            else:
                st.sidebar.error("APK not found under bin directory.")
    else:
        st.sidebar.markdown("""
        <div class="sidebar-device-card" style="border-color:#475569;">
            <div style="color:#94a3b8;">No ADB device detected. Connect target device via USB debugging.</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Packet Capture Controls
    st.sidebar.markdown('<div class="sidebar-section-title">Packet Capture (PCAP)</div>', unsafe_allow_html=True)
    col_pcap1, col_pcap2 = st.sidebar.columns(2)

    if not is_pcap_recording:
        if col_pcap1.button("Start Recording", width="stretch", type="primary"):
            res = query_api("/api/control/pcap/start", method="POST")
            if res:
                st.rerun()
    else:
        if col_pcap1.button("Stop Recording", width="stretch"):
            res = query_api("/api/control/pcap/stop", method="POST")
            if res:
                st.rerun()

    if col_pcap2.button("Validate PCAP", width="stretch"):
        val_res = query_api("/api/pcap/validate")
        st.session_state.pcap_validation_result = val_res
        st.rerun()

    # 3. Model Calibration (Cutoff Threshold)
    st.sidebar.markdown('<div class="sidebar-section-title">Threat Detection Cutoff</div>', unsafe_allow_html=True)
    threshold = st.sidebar.slider(
        "Decision Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=0.65,
        step=0.05,
        label_visibility="collapsed",
        help="Probability cutoff for classifying flows as MALWARE_ASSOCIATED. Default is 0.65 (65%)."
    )
    st.sidebar.markdown(f"<div style='font-size:0.8rem; color:#94a3b8; margin-top:-4px; margin-bottom:8px;'>Current threshold: <b style='color:#ffffff;'>{threshold:.2f}</b> ({int(round(threshold * 100))}%)</div>", unsafe_allow_html=True)

    # Sync threshold with backend if changed
    if st.session_state.get("last_synced_threshold") != threshold:
        query_api(f"/api/config/threshold?threshold={threshold}", method="POST")
        st.session_state.last_synced_threshold = threshold

    # 6. Diagnostics & Benchmark Verification
    st.sidebar.markdown('<div class="sidebar-section-title">Diagnostics & Telemetry</div>', unsafe_allow_html=True)
    diag_cat = st.sidebar.selectbox(
        "Benchmark Flow Signature",
        options=["RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE", "BENIGN"],
        index=0
    )
    if st.sidebar.button(f"Inject {diag_cat} Flow", width="stretch"):
        if is_api_online:
            res = query_api(f"/api/diagnostic/flow?category={diag_cat}", method="POST")
            if res:
                st.session_state.data_mode = "LIVE"
                st.rerun()
        else:
            from src.sample_threats import get_genuine_sample_flow
            sample = get_genuine_sample_flow(diag_cat)
            pred = st.session_state.predictor
            t0 = time.perf_counter()
            bin_res = pred.predict_flow(sample, mode="binary")
            lat_ms = (time.perf_counter() - t0) * 1000.0
            mal_prob = bin_res["probabilities"].get("MALWARE_ASSOCIATED", 0.0)
            is_mal = mal_prob >= threshold
        
            m_res = pred.predict_flow(sample, mode="multiclass")
            multi_lbl = m_res["label"] if is_mal else "N/A"
            multi_conf = m_res["confidence"] if is_mal else 0.0
        
            VALID_FAMILIES = {"RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"}
            is_threat = bool(is_mal and multi_lbl in VALID_FAMILIES and multi_lbl != "BENIGN")
        
            t_now = time.time()
            diag_record = {
                "id": f"diag_{diag_cat.lower()}_{int(t_now * 1000)}",
                "timestamp": time.strftime("%H:%M:%S", time.localtime(t_now)),
                "src_ip": "10.0.0.2 (Diagnostic)",
                "dst_ip": f"198.51.100.{1 if is_threat else 2} (Benchmark)",
                "src_port": 49152,
                "dst_port": 443 if is_threat else 80,
                "protocol": "TCP",
                "packets": int(sample.get("Total Fwd Packets", 1) + sample.get("Total Backward Packets", 1)),
                "bytes": int(sample.get("Total Length of Fwd Packets", 0) + sample.get("Total Length of Bwd Packets", 0)),
                "status": "MALWARE_ASSOCIATED" if is_mal else "BENIGN",
                "is_threat": is_threat,
                "malware_probability": round(mal_prob, 4),
                "binary_confidence": round(bin_res["confidence"], 4),
                "multiclass_category": multi_lbl,
                "multiclass_confidence": round(multi_conf, 4),
                "latency_ms": round(lat_ms, 2),
                "mode": "DIAGNOSTIC_VERIFICATION",
                "evidence_type": "Pattern Resemblance (ML)",
                "note": f"Diagnostic flow from verified {diag_cat} CICAndMal2017 benchmark sample."
            }
            st.session_state.diagnostic_history.insert(0, diag_record)
            st.session_state.data_mode = "DIAGNOSTIC"
            st.rerun()

    if st.sidebar.button("Clear Telemetry", width="stretch"):
        st.session_state.diagnostic_history.clear()
        st.session_state.rendered_packets_seen.clear()
        st.session_state.initial_packet_sync_done = False
        st.session_state.max_rendered_packet_seq = 0
        st.session_state.pcap_validation_result = None
        st.rerun()


    # ----------------- TOP HEADER & STATUS BAR -----------------
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown('<div class="ms-header-title">MobileShield</div>', unsafe_allow_html=True)
        st.markdown('<div class="ms-header-sub">Android Network Security Monitor &bull; Layer 3/4 Intrusion Detection Console</div>', unsafe_allow_html=True)

    with col_h2:
        auto_refresh = st.checkbox(
            "Live Refresh (1s)",
            value=True,
            key="auto_refresh",
            on_change=_refresh_setting_changed,
        )

    # Build compact status bar
    backend_badge = '<span class="status-pill pill-online">Backend Online</span>' if is_api_online else '<span class="status-pill pill-offline">Backend Offline</span>'
    device_str = target_device.get("model", "None") if target_device else "None"
    if target_device and not relay_stale:
        device_badge = f'<span class="status-pill pill-active">Device: {device_str}</span>'
    elif target_device and relay_stale:
        device_badge = f'<span class="status-pill pill-idle">Device: {device_str} (stale)</span>'
    else:
        device_badge = '<span class="status-pill pill-idle">Device: Disconnected</span>'
    capture_badge = '<span class="status-pill pill-online">Capture: Streaming</span>' if is_streaming else '<span class="status-pill pill-idle">Capture: Standby</span>'
    pcap_badge = '<span class="status-pill pill-threat">PCAP: Recording</span>' if is_pcap_recording else '<span class="status-pill pill-idle">PCAP: Standby</span>'

    st.markdown(f"""
    <div class="status-bar-container">
        {backend_badge}
        {device_badge}
        {capture_badge}
        {pcap_badge}
        <span style="margin-left:auto; color:#64748b; font-size:0.75rem;">Endpoint: 127.0.0.1:8000 &bull; Tunnel: adb reverse tcp:8000</span>
    </div>
    """, unsafe_allow_html=True)

    # ----------------- COMPACT METRIC CARDS & ALERT INGESTION -----------------
    tot_packets = telemetry_data.get("total_packets", 0) if telemetry_data else 0
    tot_bytes = telemetry_data.get("total_bytes", 0) if telemetry_data else 0
    act_flows = telemetry_data.get("active_flows", 0) if telemetry_data else 0
    rec_flows = telemetry_data.get("recent_flows", []) if telemetry_data else []
    raw_alerts = telemetry_data.get("alerts", []) if telemetry_data else []

    if not is_api_online:
        if st.session_state.data_mode == "DIAGNOSTIC":
            rec_flows = st.session_state.diagnostic_history
            raw_alerts = [f for f in rec_flows if f.get("is_threat", False) or (f.get("status") == "MALWARE_ASSOCIATED" and f.get("multiclass_category") in {"RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"})]
            tot_packets = sum(f.get("packets", 0) for f in rec_flows)
            tot_bytes = sum(f.get("bytes", 0) for f in rec_flows)

    # Threat Alert Filtering & Deduplication (Policy: Probability >= threshold AND supported malware family):
    VALID_THREAT_FAMILIES = {"RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"}
    active_threat_alerts = []
    seen_alert_sigs = set()

    for a in raw_alerts:
        prob = float(a.get("malware_probability", 0.0))
        cat = str(a.get("multiclass_category", "Unknown")).upper()
        # ML Alert Policy: prob >= threshold AND supported family AND NOT BENIGN
        if prob >= threshold and cat in VALID_THREAT_FAMILIES and cat != "BENIGN":
            alert_sig = a.get("id") or f"{a.get('timestamp')}_{a.get('src_ip')}:{a.get('src_port')}_{a.get('dst_ip')}:{a.get('dst_port')}_{a.get('protocol')}_{cat}"
            if alert_sig not in seen_alert_sigs:
                seen_alert_sigs.add(alert_sig)
                a_copy = dict(a)
                a_copy["detection_source"] = a.get("detection_source", "ML_MODEL")
                active_threat_alerts.append(a_copy)

    vol_str = f"{tot_bytes / (1024*1024):.2f} MB" if tot_bytes > 1024*1024 else f"{tot_bytes / 1024:.1f} KB"

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Packets Captured</div>
            <div class="metric-value">{tot_packets:,}</div>
            <div class="metric-hint">Raw Layer 3/4 frames</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Traffic Volume</div>
            <div class="metric-value">{vol_str}</div>
            <div class="metric-hint">Ingested bytes</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Flows</div>
            <div class="metric-value">{act_flows}</div>
            <div class="metric-hint">In aggregation buffer</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ML Threat Alerts</div>
            <div class="metric-value">{len(active_threat_alerts)}</div>
            <div class="metric-hint">Model Cutoff &ge; {threshold:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ----------------- MAIN NAVIGATION TABS -----------------
    tab_packets, tab_flows, tab_alerts, tab_pcap, tab_diag = st.tabs([
        "Live Packets",
        "Flow Monitoring",
        "Threat Detection",
        "Packet Capture",
        "System Diagnostics"
    ])

    # ----------------- TAB 1: LIVE PACKET STREAM -----------------
    with tab_packets:
        recent_pkts = telemetry_data.get("recent_packets", []) if telemetry_data else []
    
        col_filter1, col_filter2, col_filter3 = st.columns([2, 3, 2])
        with col_filter1:
            proto_filter = st.selectbox("Protocol Filter", options=["All", "TCP", "UDP", "DNS", "ICMP"], index=0)
        with col_filter2:
            search_filter = st.text_input("Filter Address / Port", placeholder="Search IP or Port...")
        with col_filter3:
            max_rows = st.selectbox("Display Limit", options=[25, 50, 100], index=1)
        
        filtered_pkts = []
        for p in recent_pkts:
            proto_str = str(p.get("protocol", "")).upper()
            if proto_filter != "All" and proto_str != proto_filter:
                continue
            if search_filter:
                needle = search_filter.lower()
                src = f"{p.get('src_ip', '')}:{p.get('src_port', '')}".lower()
                dst = f"{p.get('dst_ip', '')}:{p.get('dst_port', '')}".lower()
                if needle not in src and needle not in dst:
                    continue
            filtered_pkts.append(p)
        
        if not filtered_pkts:
            st.markdown("""
            <div class="context-panel" style="border-left-color: #64748b;">
                <b>No packets matching criteria.</b><br/>
                Start capture in the MobileShield companion app on your connected device, browse web traffic, or trigger a diagnostic flow.
            </div>
            """, unsafe_allow_html=True)
        else:
            now = time.time()

            packet_sequences = [
                int(packet["seq"])
                for packet in recent_pkts
                if packet.get("seq") is not None
            ]
            current_max_seq = max(packet_sequences, default=0)

            if not st.session_state.initial_packet_sync_done:
                st.session_state.max_rendered_packet_seq = current_max_seq
                st.session_state.initial_packet_sync_done = True
            elif current_max_seq < st.session_state.max_rendered_packet_seq:
                st.session_state.max_rendered_packet_seq = current_max_seq
                st.session_state.rendered_packets_seen.clear()
            else:
                previous_max_seq = st.session_state.max_rendered_packet_seq
                for packet in recent_pkts:
                    sequence = packet.get("seq")
                    if sequence is not None and int(sequence) > previous_max_seq:
                        packet_id = packet.get("id") or (
                            f"pkt_{sequence}_{packet.get('raw_timestamp', packet.get('timestamp'))}"
                        )
                        st.session_state.rendered_packets_seen[packet_id] = now
                st.session_state.max_rendered_packet_seq = current_max_seq

            st.session_state.rendered_packets_seen = {
                packet_id: first_seen
                for packet_id, first_seen in st.session_state.rendered_packets_seen.items()
                if now - first_seen < 2.6
            }

            rows_html = []
            for p in filtered_pkts[:max_rows]:
                pid = p.get("id") or f"pkt_{p.get('seq', 0)}_{p.get('raw_timestamp', p.get('timestamp'))}"
                first_seen = st.session_state.rendered_packets_seen.get(pid)
                highlight_age = now - first_seen if first_seen is not None else 999.0
                is_new = highlight_age < 2.6

                if is_new:
                    row_cls = "pkt-row-new"
                    delay_sec = max(0.0, highlight_age)
                    anim_style = f"style=\"animation-delay: -{delay_sec:.2f}s;\""
                    badge_html = '<span class="pkt-badge-new">NEW</span>'
                else:
                    row_cls = "pkt-row-normal"
                    anim_style = ""
                    badge_html = ""

                proto_raw = str(p.get("protocol", "TCP")).upper()
                if proto_raw == "TCP":
                    proto_badge = f'<span class="pkt-proto-pill pkt-proto-tcp">{html.escape(proto_raw)}</span>'
                elif proto_raw in ("UDP", "DNS"):
                    proto_badge = f'<span class="pkt-proto-pill pkt-proto-udp">{html.escape(proto_raw)}</span>'
                else:
                    proto_badge = f'<span class="pkt-proto-pill pkt-proto-other">{html.escape(proto_raw)}</span>'

                seq_num = html.escape(str(p.get("seq", "-")))
                src_str = html.escape(f"{p.get('src_ip')}:{p.get('src_port')}")
                dst_str = html.escape(f"{p.get('dst_ip')}:{p.get('dst_port')}")
                ts_str = html.escape(str(p.get("timestamp", "--:--:--")))
                len_val = p.get("length", 0)
                len_str = f"{len_val:,} B" if isinstance(len_val, (int, float)) else html.escape(str(len_val))
                flags_str = html.escape(str(p.get("flags") if p.get("flags") else "-"))
                app_str = html.escape(str(p.get("app") if p.get("app") else "com.android.system"))

                rows_html.append(f"""<tr class="{row_cls}" {anim_style}>
                    <td style="color:#94a3b8; font-size:0.75rem;">{badge_html}#{seq_num}</td>
                    <td>{ts_str}</td>
                    <td>{src_str}</td>
                    <td>{dst_str}</td>
                    <td>{proto_badge}</td>
                    <td>{len_str}</td>
                    <td style="color:#94a3b8;">{flags_str}</td>
                    <td style="color:#94a3b8; max-width:200px; overflow:hidden; text-overflow:ellipsis;" title="{app_str}">{app_str}</td>
                </tr>""")

            table_markup = f"""<div class="pkt-table-wrap">
                <table class="pkt-table">
                    <thead>
                        <tr>
                            <th style="width: 85px;">Seq</th>
                            <th style="width: 90px;">Time</th>
                            <th>Source</th>
                            <th>Destination</th>
                            <th style="width: 80px;">Proto</th>
                            <th style="width: 95px;">Length</th>
                            <th style="width: 85px;">Flags</th>
                            <th>Application</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows_html)}
                    </tbody>
                </table>
            </div>"""
            st.markdown(table_markup, unsafe_allow_html=True)
            st.caption("Event Separation (Packet Event vs Threat Alert): Newly captured Layer 3/4 packets highlight ruby red (#7f1d1d) for ~2.6s to indicate immediate ingestion. This signifies live packet arrival only, completely independent of ML classification, and does NOT mean malware was detected.")

    # ----------------- TAB 2: FLOW MONITORING -----------------
    with tab_flows:
        st.markdown("""
        <div class="context-panel">
            <b>Bidirectional Flow Aggregation:</b> Packets are correlated into 5-tuple sessions (Source IP, Source Port, Destination IP, Destination Port, Protocol)
            and aligned to the 65-feature schema required by the Random Forest model. Unfinished or incompatible flows are designated <code>NOT_CLASSIFIED</code>.
        </div>
        """, unsafe_allow_html=True)

        if not rec_flows:
            st.markdown("""
            <div class="context-panel" style="border-left-color: #64748b;">
                <b>No completed network flows.</b> Flows finalize upon active timeout (15s) or inactivity timeout (5s).
            </div>
            """, unsafe_allow_html=True)
        else:
            flow_table = []
            for f in rec_flows[:50]:
                status = f.get("status", "BENIGN")
                prob = f.get("malware_probability", 0.0)
                prob_str = f"{prob * 100:.1f}%"
                conf = f.get("binary_confidence", max(prob, 1.0 - prob))

                if status == "MALWARE_ASSOCIATED":
                    status_display = f"MALWARE ({prob_str})"
                elif status == "NOT_CLASSIFIED":
                    status_display = "NOT_CLASSIFIED"
                elif 0.40 <= prob < threshold:
                    status_display = f"UNCERTAIN ({prob_str})"
                else:
                    status_display = f"BENIGN ({100.0 - prob * 100:.1f}%)"

                multi_cat = f.get("multiclass_category", "N/A")
                if multi_cat != "N/A" and multi_cat != "Unknown":
                    multi_cat = f"{multi_cat} ({f.get('multiclass_confidence', 0.0)*100:.1f}%)"

                flow_table.append({
                    "Timestamp": f.get("timestamp"),
                    "Source": f"{f.get('src_ip')}:{f.get('src_port')}",
                    "Destination": f"{f.get('dst_ip')}:{f.get('dst_port')}",
                    "Proto": f.get("protocol"),
                    "Packets": f.get("packets"),
                    "Volume (Bytes)": f"{f.get('bytes', 0):,}",
                    "Classification": status_display,
                    "Confidence": f"{conf*100:.1f}%",
                    "Family": multi_cat,
                    "Latency": f"{f.get('latency_ms', 0):.1f} ms"
                })
            st.dataframe(flow_table, width="stretch", hide_index=True)

    # ----------------- TAB 3: THREAT DETECTION -----------------
    with tab_alerts:
        st.markdown(f"""
        <div class="context-panel">
            <b>ML Classification Interpretation:</b> Independent machine learning inferences generated by the 65-feature flow classifier.
            Threat alerts require both binary malware probability &ge; {threshold:.2f} (65% default) and a predicted family in the supported malware taxonomy
            (RANSOMWARE, ADWARE, SCAREWARE, SMSMALWARE). BENIGN predictions never trigger alerts. This represents statistical pattern resemblance, not confirmed infection.
        </div>
        """, unsafe_allow_html=True)

        if not active_threat_alerts:
            st.markdown(f"""
            <div class="context-panel" style="border-left-color: #059669;">
                <b>No ML threat alerts recorded.</b> All inspected network flows conform to baseline benign behavior or fall below the decision cutoff threshold ({threshold:.2f}).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption(f"{len(active_threat_alerts)} deduplicated flow(s) flagged by ML model above cutoff ({threshold:.2f}):")
            alert_table = []
            for a in active_threat_alerts:
                alert_table.append({
                    "Timestamp": a.get("timestamp"),
                    "Source": f"{a.get('src_ip')}:{a.get('src_port')}",
                    "Destination": f"{a.get('dst_ip')}:{a.get('dst_port')}",
                    "Protocol": a.get("protocol"),
                    "Packets": a.get("packets"),
                    "Volume (Bytes)": f"{a.get('bytes', 0):,}",
                    "Malware Probability": f"{float(a.get('malware_probability', 0.0))*100:.1f}%",
                    "Multiclass Confidence": f"{float(a.get('multiclass_confidence', 0.0))*100:.1f}%",
                    "Predicted Family": a.get("multiclass_category", "Unknown"),
                    "Detection Source": a.get("detection_source", "ML_MODEL"),
                    "Assessment Note": a.get("note", "Structural pattern similarity")
                })
            st.dataframe(alert_table, width="stretch", hide_index=True)


    # ----------------- TAB 4: PACKET CAPTURE (PCAP) -----------------
    with tab_pcap:
        st.markdown('<div class="sidebar-section-title" style="margin-top:0;">Raw PCAP Capture Status</div>', unsafe_allow_html=True)
    
        col_pcap_info1, col_pcap_info2 = st.columns(2)
        with col_pcap_info1:
            st.markdown(f"""
            <div class="sidebar-device-card" style="font-size:0.85rem; line-height:1.6;">
                <div><b>Recorder State:</b> {'RECORDING' if is_pcap_recording else 'STANDBY'}</div>
                <div><b>Active PCAP File:</b> <code>{api_status.get('current_pcap_path', 'None') if api_status else 'None'}</code></div>
                <div><b>Framing:</b> Ethernet LinkType 1 (0x01) with standard 14-byte headers</div>
            </div>
            """, unsafe_allow_html=True)
        
            if api_status and api_status.get("current_pcap_path"):
                if st.button("Prepare authenticated PCAP download"):
                    try:
                        st.session_state.pcap_download_data = query_api_file("/api/pcap/download")
                    except (OSError, urllib.error.URLError) as exc:
                        st.session_state.pcap_download_data = None
                        st.error(f"Unable to download the PCAP: {exc}")
                if st.session_state.get("pcap_download_data"):
                    st.download_button(
                        "Download Current PCAP Capture File",
                        data=st.session_state.pcap_download_data,
                        file_name="mobileshield_capture.pcap",
                        mime="application/vnd.tcpdump.pcap",
                    )
            
        with col_pcap_info2:
            st.markdown('<div class="sidebar-section-title" style="margin-top:0;">Structure & Header Verification</div>', unsafe_allow_html=True)
            if st.session_state.pcap_validation_result:
                v_res = st.session_state.pcap_validation_result
                if v_res.get("valid"):
                    st.markdown(f"""
                    <div class="sidebar-device-card" style="border-color:#059669; font-size:0.85rem; line-height:1.6;">
                        <div style="color:#6ee7b7; font-weight:600;">PCAP Structure Verified Valid</div>
                        <div><b>Verified Packets:</b> {v_res.get('packet_count'):,}</div>
                        <div><b>File Size:</b> {v_res.get('size_bytes'):,} bytes</div>
                        <div><b>LinkType:</b> {v_res.get('link_type')} (Ethernet DLT_EN10MB)</div>
                        <div><b>Path:</b> <code>{v_res.get('path')}</code></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="sidebar-device-card" style="border-color:#dc2626; color:#fca5a5;">
                        <div><b>Validation Failed:</b> {v_res.get('error')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="sidebar-device-card" style="color:#94a3b8;">
                    Click <b>Validate PCAP</b> to parse the current capture file with Scapy and verify PCAP header integrity.
                </div>
                """, unsafe_allow_html=True)

    # ----------------- TAB 5: SYSTEM DIAGNOSTICS -----------------
    with tab_diag:
        diag_c1, diag_c2 = st.columns(2)
    
        with diag_c1:
            st.markdown('<div class="sidebar-section-title" style="margin-top:0;">Hardware & ADB Bridge</div>', unsafe_allow_html=True)
            if target_device:
                st.json({
                    "Serial": target_device.get("serial"),
                    "Model": target_device.get("model"),
                    "Android Version": target_device.get("android_version"),
                    "Connection State": target_device.get("status"),
                    "Device Codename": target_device.get("device"),
                    "Package Name": "com.mobileshield.companion"
                })
            else:
                st.markdown('<div class="context-panel" style="border-left-color:#64748b;">No physical ADB device connected.</div>', unsafe_allow_html=True)

        with diag_c2:
            st.markdown('<div class="sidebar-section-title" style="margin-top:0;">Local API & Bridge Endpoints</div>', unsafe_allow_html=True)
            st.json({
                "FastAPI Ingestion Endpoint": f"{API_BASE_URL}/api/packets",
                "Telemetry Endpoint": f"{API_BASE_URL}/api/telemetry",
                "ADB Reverse Command": "adb reverse tcp:8000 tcp:8000",
                "Authentication Token": "Configured (X-MobileShield-Token)",
                "Backend Status": "ONLINE" if is_api_online else "OFFLINE"
            })

        st.markdown('<div class="sidebar-section-title">Model Specifications</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-device-card" style="font-size:0.82rem; line-height:1.6;">
            <div><b>Binary Classifier:</b> Random Forest (100 estimators, max depth 20, class_weight='balanced')</div>
            <div><b>Multiclass Classifier:</b> Random Forest (100 estimators, max depth 20, 5 canonical classes)</div>
            <div><b>Feature Schema:</b> 65 non-identifying statistical flow features (10 identifiers stripped, 12 zero-variance features dropped)</div>
            <div><b>Training Benchmark:</b> CICAndMal2017 with leak-proof Group-Aware splitting by APK sample ID</div>
        </div>
        """, unsafe_allow_html=True)

render_dashboard()
