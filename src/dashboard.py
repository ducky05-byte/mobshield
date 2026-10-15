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

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://holder-negotiations-provision-handy.trycloudflare.com"
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


def _get_active_threat_alerts(
    telemetry_data: Optional[Dict[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    raw_alerts = telemetry_data.get("alerts", []) if telemetry_data else []
    if not telemetry_data and st.session_state.data_mode == "DIAGNOSTIC":
        raw_alerts = [
            flow
            for flow in st.session_state.diagnostic_history
            if flow.get("is_threat", False)
            or (
                flow.get("status") == "MALWARE_ASSOCIATED"
                and flow.get("multiclass_category")
                in {"RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"}
            )
        ]

    valid_families = {"RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"}
    active_alerts = []
    seen_signatures = set()
    for alert in raw_alerts:
        probability = float(alert.get("malware_probability", 0.0))
        category = str(alert.get("multiclass_category", "Unknown")).upper()
        if probability < threshold or category not in valid_families:
            continue

        signature = alert.get("id") or (
            f"{alert.get('timestamp')}_{alert.get('src_ip')}:{alert.get('src_port')}_"
            f"{alert.get('dst_ip')}:{alert.get('dst_port')}_"
            f"{alert.get('protocol')}_{category}"
        )
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            alert_copy = dict(alert)
            alert_copy["detection_source"] = alert.get("detection_source", "ML_MODEL")
            active_alerts.append(alert_copy)
    return active_alerts


@st.fragment(run_every=1 if auto_refresh_enabled else None)
def render_live_metrics(threshold: float) -> None:
    """Refresh capture status and summary metrics without rerunning other UI."""
    api_status = query_api("/api/status")
    telemetry_data = query_api("/api/telemetry")
    is_api_online = telemetry_data is not None
    is_streaming = api_status.get("is_streaming", False) if api_status else False
    is_pcap_recording = api_status.get("pcap_recording", False) if api_status else False
    target_device = st.session_state.get("dashboard_target_device")
    relay_stale = st.session_state.get("dashboard_relay_stale", True)

    backend_badge = (
        '<span class="status-pill pill-online">Backend Online</span>'
        if is_api_online
        else '<span class="status-pill pill-offline">Backend Offline</span>'
    )
    device_str = target_device.get("model", "None") if target_device else "None"
    if target_device and not relay_stale:
        device_badge = f'<span class="status-pill pill-active">Device: {device_str}</span>'
    elif target_device and relay_stale:
        device_badge = f'<span class="status-pill pill-idle">Device: {device_str} (stale)</span>'
    else:
        device_badge = '<span class="status-pill pill-idle">Device: Disconnected</span>'
    capture_badge = (
        '<span class="status-pill pill-online">Capture: Streaming</span>'
        if is_streaming
        else '<span class="status-pill pill-idle">Capture: Standby</span>'
    )
    pcap_badge = (
        '<span class="status-pill pill-threat">PCAP: Recording</span>'
        if is_pcap_recording
        else '<span class="status-pill pill-idle">PCAP: Standby</span>'
    )
    st.markdown(
        f"""
        <div class="status-bar-container">
            {backend_badge}
            {device_badge}
            {capture_badge}
            {pcap_badge}
            <span style="margin-left:auto; color:#64748b; font-size:0.75rem;">Endpoint: 127.0.0.1:8000 &bull; Tunnel: adb reverse tcp:8000</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_packets = telemetry_data.get("total_packets", 0) if telemetry_data else 0
    total_bytes = telemetry_data.get("total_bytes", 0) if telemetry_data else 0
    active_flows = telemetry_data.get("active_flows", 0) if telemetry_data else 0
    if not is_api_online and st.session_state.data_mode == "DIAGNOSTIC":
        flows = st.session_state.diagnostic_history
        total_packets = sum(flow.get("packets", 0) for flow in flows)
        total_bytes = sum(flow.get("bytes", 0) for flow in flows)
    active_alerts = _get_active_threat_alerts(telemetry_data, threshold)
    volume = (
        f"{total_bytes / (1024 * 1024):.2f} MB"
        if total_bytes > 1024 * 1024
        else f"{total_bytes / 1024:.1f} KB"
    )

    metric_columns = st.columns(4)
    metrics = (
        ("Packets Captured", f"{total_packets:,}", "Raw Layer 3/4 frames"),
        ("Traffic Volume", volume, "Ingested bytes"),
        ("Active Flows", str(active_flows), "In aggregation buffer"),
        ("ML Threat Alerts", str(len(active_alerts)), f"Model Cutoff >= {threshold:.2f}"),
    )
    for column, (label, value, hint) in zip(metric_columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-hint">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


@st.fragment(run_every=1 if auto_refresh_enabled else None)
def render_live_packets() -> None:
    """Refresh only the packet table while preserving its filter controls."""
    telemetry_data = query_api("/api/telemetry")
    recent_packets = telemetry_data.get("recent_packets", []) if telemetry_data else []

    filter_columns = st.columns([2, 3, 2])
    with filter_columns[0]:
        protocol_filter = st.selectbox(
            "Protocol Filter", options=["All", "TCP", "UDP", "DNS", "ICMP"], index=0
        )
    with filter_columns[1]:
        search_filter = st.text_input(
            "Filter Address / Port", placeholder="Search IP or Port..."
        )
    with filter_columns[2]:
        max_rows = st.selectbox("Display Limit", options=[25, 50, 100], index=1)

    filtered_packets = []
    for packet in recent_packets:
        protocol = str(packet.get("protocol", "")).upper()
        if protocol_filter != "All" and protocol != protocol_filter:
            continue
        if search_filter:
            needle = search_filter.lower()
            source = f"{packet.get('src_ip', '')}:{packet.get('src_port', '')}".lower()
            destination = f"{packet.get('dst_ip', '')}:{packet.get('dst_port', '')}".lower()
            if needle not in source and needle not in destination:
                continue
        filtered_packets.append(packet)

    if not filtered_packets:
        st.markdown(
            """
            <div class="context-panel" style="border-left-color: #64748b;">
                <b>No packets matching criteria.</b><br/>
                Start capture in the MobileShield companion app on your connected device, browse web traffic, or trigger a diagnostic flow.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    now = time.time()
    packet_sequences = [
        int(packet["seq"])
        for packet in recent_packets
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
        for packet in recent_packets:
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
    for packet in filtered_packets[:max_rows]:
        packet_id = packet.get("id") or (
            f"pkt_{packet.get('seq', 0)}_"
            f"{packet.get('raw_timestamp', packet.get('timestamp'))}"
        )
        first_seen = st.session_state.rendered_packets_seen.get(packet_id)
        highlight_age = now - first_seen if first_seen is not None else 999.0
        if highlight_age < 2.6:
            row_class = "pkt-row-new"
            animation_style = f"style=\"animation-delay: -{max(0.0, highlight_age):.2f}s;\""
            badge_html = '<span class="pkt-badge-new">NEW</span>'
        else:
            row_class = "pkt-row-normal"
            animation_style = ""
            badge_html = ""

        protocol = str(packet.get("protocol", "TCP")).upper()
        if protocol == "TCP":
            protocol_badge = (
                f'<span class="pkt-proto-pill pkt-proto-tcp">{html.escape(protocol)}</span>'
            )
        elif protocol in ("UDP", "DNS"):
            protocol_badge = (
                f'<span class="pkt-proto-pill pkt-proto-udp">{html.escape(protocol)}</span>'
            )
        else:
            protocol_badge = (
                f'<span class="pkt-proto-pill pkt-proto-other">{html.escape(protocol)}</span>'
            )

        sequence = html.escape(str(packet.get("seq", "-")))
        source = html.escape(f"{packet.get('src_ip')}:{packet.get('src_port')}")
        destination = html.escape(f"{packet.get('dst_ip')}:{packet.get('dst_port')}")
        timestamp = html.escape(str(packet.get("timestamp", "--:--:--")))
        length = packet.get("length", 0)
        length_display = (
            f"{length:,} B"
            if isinstance(length, (int, float))
            else html.escape(str(length))
        )
        flags = html.escape(str(packet.get("flags") or "-"))
        app = html.escape(str(packet.get("app") or "com.android.system"))
        rows_html.append(
            f"""<tr class="{row_class}" {animation_style}>
                <td style="color:#94a3b8; font-size:0.75rem;">{badge_html}#{sequence}</td>
                <td>{timestamp}</td>
                <td>{source}</td>
                <td>{destination}</td>
                <td>{protocol_badge}</td>
                <td>{length_display}</td>
                <td style="color:#94a3b8;">{flags}</td>
                <td style="color:#94a3b8; max-width:200px; overflow:hidden; text-overflow:ellipsis;" title="{app}">{app}</td>
            </tr>"""
        )

    st.markdown(
        f"""<div class="pkt-table-wrap">
            <table class="pkt-table">
                <thead><tr>
                    <th style="width: 85px;">Seq</th>
                    <th style="width: 90px;">Time</th>
                    <th>Source</th><th>Destination</th>
                    <th style="width: 80px;">Proto</th>
                    <th style="width: 95px;">Length</th>
                    <th style="width: 85px;">Flags</th><th>Application</th>
                </tr></thead>
                <tbody>{''.join(rows_html)}</tbody>
            </table>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption(
        "Newly captured packets highlight ruby red briefly to indicate packet "
        "arrival only; this is independent of ML threat detection."
    )


@st.fragment(run_every=1 if auto_refresh_enabled else None)
def render_live_flows(threshold: float) -> None:
    """Refresh the completed-flow table without rerunning the rest of the page."""
    telemetry_data = query_api("/api/telemetry")
    recent_flows = telemetry_data.get("recent_flows", []) if telemetry_data else []
    if not telemetry_data and st.session_state.data_mode == "DIAGNOSTIC":
        recent_flows = st.session_state.diagnostic_history

    st.markdown(
        """
        <div class="context-panel">
            <b>Bidirectional Flow Aggregation:</b> Packets are correlated into 5-tuple sessions (Source IP, Source Port, Destination IP, Destination Port, Protocol)
            and aligned to the 65-feature schema required by the Random Forest model. Unfinished or incompatible flows are designated <code>NOT_CLASSIFIED</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not recent_flows:
        st.markdown(
            """
            <div class="context-panel" style="border-left-color: #64748b;">
                <b>No completed network flows.</b> Flows finalize upon active timeout or inactivity timeout.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    flow_table = []
    for flow in recent_flows[:50]:
        status = flow.get("status", "BENIGN")
        probability = flow.get("malware_probability", 0.0)
        probability_display = f"{probability * 100:.1f}%"
        confidence = flow.get("binary_confidence", max(probability, 1.0 - probability))
        if status == "MALWARE_ASSOCIATED":
            status_display = f"MALWARE ({probability_display})"
        elif status == "NOT_CLASSIFIED":
            status_display = "NOT_CLASSIFIED"
        elif 0.40 <= probability < threshold:
            status_display = f"UNCERTAIN ({probability_display})"
        else:
            status_display = f"BENIGN ({100.0 - probability * 100:.1f}%)"

        category = flow.get("multiclass_category", "N/A")
        if category != "N/A" and category != "Unknown":
            category = f"{category} ({flow.get('multiclass_confidence', 0.0) * 100:.1f}%)"
        flow_table.append(
            {
                "Timestamp": flow.get("timestamp"),
                "Source": f"{flow.get('src_ip')}:{flow.get('src_port')}",
                "Destination": f"{flow.get('dst_ip')}:{flow.get('dst_port')}",
                "Proto": flow.get("protocol"),
                "Packets": flow.get("packets"),
                "Volume (Bytes)": f"{flow.get('bytes', 0):,}",
                "Classification": status_display,
                "Confidence": f"{confidence * 100:.1f}%",
                "Family": category,
                "Latency": f"{flow.get('latency_ms', 0):.1f} ms",
            }
        )
    st.dataframe(flow_table, width="stretch", hide_index=True)


@st.fragment(run_every=1 if auto_refresh_enabled else None)
def render_live_threat_alerts(threshold: float) -> None:
    """Refresh detailed threat results while keeping alert records unchanged."""
    telemetry_data = query_api("/api/telemetry")
    active_alerts = _get_active_threat_alerts(telemetry_data, threshold)
    st.markdown(
        f"""
        <div class="context-panel">
            <b>ML Classification Interpretation:</b> Independent machine learning inferences generated by the 65-feature flow classifier.
            Threat alerts require both binary malware probability &ge; {threshold:.2f} and a predicted family in the supported malware taxonomy
            (RANSOMWARE, ADWARE, SCAREWARE, SMSMALWARE). BENIGN predictions never trigger alerts. This represents statistical pattern resemblance, not confirmed infection.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not active_alerts:
        st.markdown(
            f"""
            <div class="context-panel" style="border-left-color: #059669;">
                <b>No ML threat alerts recorded.</b> All inspected network flows conform to baseline benign behavior or fall below the decision cutoff threshold ({threshold:.2f}).
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(
        f"{len(active_alerts)} deduplicated flow(s) flagged by ML model above "
        f"cutoff ({threshold:.2f}):"
    )
    alert_table = []
    for alert in active_alerts:
        alert_table.append(
            {
                "Timestamp": alert.get("timestamp"),
                "Source": f"{alert.get('src_ip')}:{alert.get('src_port')}",
                "Destination": f"{alert.get('dst_ip')}:{alert.get('dst_port')}",
                "Protocol": alert.get("protocol"),
                "Packets": alert.get("packets"),
                "Volume (Bytes)": f"{alert.get('bytes', 0):,}",
                "Malware Probability": (
                    f"{float(alert.get('malware_probability', 0.0)) * 100:.1f}%"
                ),
                "Multiclass Confidence": (
                    f"{float(alert.get('multiclass_confidence', 0.0)) * 100:.1f}%"
                ),
                "Predicted Family": alert.get("multiclass_category", "Unknown"),
                "Detection Source": alert.get("detection_source", "ML_MODEL"),
                "Assessment Note": alert.get("note", "Structural pattern similarity"),
            }
        )
    st.dataframe(alert_table, width="stretch", hide_index=True)


@st.fragment(run_every=1 if auto_refresh_enabled else None)
def render_live_pipeline_diagnostics() -> None:
    """Refresh inference counters without refreshing the hardware diagnostics."""
    telemetry_data = query_api("/api/telemetry")
    diagnostics = (
        telemetry_data.get("inference_diagnostics", {}) if telemetry_data else {}
    )
    st.markdown(
        '<div class="sidebar-section-title">Threat-Detection Pipeline</div>',
        unsafe_allow_html=True,
    )
    diagnostic_columns = st.columns(4)
    diagnostic_metrics = (
        ("Packets received", diagnostics.get("packets_received", 0)),
        ("Flows submitted", diagnostics.get("flows_submitted_for_inference", 0)),
        ("Predictions succeeded", diagnostics.get("predictions_succeeded", 0)),
        ("Predictions failed", diagnostics.get("predictions_failed", 0)),
        ("Rejected by threshold", diagnostics.get("alerts_rejected_threshold", 0)),
        ("Rejected by class", diagnostics.get("alerts_rejected_class_mapping", 0)),
        ("Alerts stored", diagnostics.get("alerts_stored", 0)),
        ("Feature errors", diagnostics.get("feature_extraction_failures", 0)),
    )
    for index, (label, value) in enumerate(diagnostic_metrics):
        with diagnostic_columns[index % len(diagnostic_columns)]:
            st.metric(label, value)

    last_error = diagnostics.get("last_error")
    if last_error:
        st.warning(
            "Last pipeline error: "
            f"{last_error.get('stage', 'unknown stage')} "
            f"({last_error.get('error_type', 'unknown error')})"
        )

    recent_predictions = diagnostics.get("recent_predictions", [])
    if recent_predictions:
        prediction_columns = (
            "timestamp",
            "binary_predicted_class",
            "malware_probability",
            "multiclass_predicted_class",
            "multiclass_confidence",
            "decision",
            "stage",
            "error_type",
        )
        st.dataframe(
            [
                {column: record.get(column) for column in prediction_columns}
                for record in recent_predictions
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No flow predictions have been recorded yet.")


def render_dashboard() -> None:
    """Render stable controls and isolate live data in small fragments."""


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
    selected_threshold = st.sidebar.slider(
        "Decision Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=0.65,
        step=0.05,
        label_visibility="collapsed",
        help="Probability cutoff for classifying flows as MALWARE_ASSOCIATED. Default is 0.65 (65%)."
    )
    threshold = selected_threshold
    if is_api_online:
        backend_threshold = (
            float(telemetry_data["detection_threshold"])
            if telemetry_data and telemetry_data.get("detection_threshold") is not None
            else None
        )
        if backend_threshold is None or backend_threshold != selected_threshold:
            threshold_result = query_api(
                f"/api/config/threshold?threshold={selected_threshold}",
                method="POST",
            )
            if (
                threshold_result
                and threshold_result.get("status") == "UPDATED"
                and threshold_result.get("threshold") is not None
            ):
                backend_threshold = float(threshold_result["threshold"])
            elif backend_threshold is not None:
                st.sidebar.warning(
                    "The backend cutoff was not updated; the dashboard is using "
                    f"the current backend value ({backend_threshold:.2f})."
                )
            else:
                backend_threshold = st.session_state.get("last_synced_threshold")
                st.sidebar.warning(
                    "The backend cutoff could not be verified; the dashboard is "
                    "using its last confirmed value."
                )
        if backend_threshold is not None:
            threshold = float(backend_threshold)
            st.session_state.last_synced_threshold = threshold

    cutoff_label = "Effective backend cutoff" if is_api_online else "Effective local cutoff"
    st.sidebar.markdown(
        f"<div style='font-size:0.8rem; color:#94a3b8; margin-top:-4px; margin-bottom:8px;'>"
        f"{cutoff_label}: <b style='color:#ffffff;'>{threshold:.2f}</b> "
        f"({int(round(threshold * 100))}%)</div>",
        unsafe_allow_html=True,
    )

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

    st.session_state.dashboard_target_device = target_device
    st.session_state.dashboard_relay_stale = relay_stale
    render_live_metrics(threshold)

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
        render_live_packets()

    # ----------------- TAB 2: FLOW MONITORING -----------------
    with tab_flows:
        render_live_flows(threshold)

    # ----------------- TAB 3: THREAT DETECTION -----------------
    with tab_alerts:
        render_live_threat_alerts(threshold)


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
        render_live_pipeline_diagnostics()

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
