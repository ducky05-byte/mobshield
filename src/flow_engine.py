"""Live Bidirectional Flow Aggregation & Feature Alignment Engine.

Assembles raw Layer 3/4 packets into bidirectional flows, tracks inter-arrival
times, packet lengths, TCP flags, and subflow statistics, and computes the exact
65 features required by the trained Phase 1 machine learning model.
"""

import math
import time
import threading
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd

try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
except ImportError:
    scapy = None


class BidirectionalFlow:
    """Represents a single bidirectional 5-tuple network communication flow."""

    def __init__(self, first_packet, flow_key: Tuple):
        self.flow_key = flow_key  # (src_ip, dst_ip, src_port, dst_port, protocol)
        self.initiator_ip = flow_key[0]
        self.initiator_port = flow_key[2]
        self.protocol = flow_key[4]

        # Timestamps
        self.start_time = float(first_packet.time)
        self.last_time = self.start_time
        self.last_fwd_time = self.start_time
        self.last_bwd_time = None

        # Packet and byte counters
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_lengths: List[int] = []
        self.bwd_lengths: List[int] = []

        # Inter-arrival times (in microseconds)
        self.flow_iats: List[float] = []
        self.fwd_iats: List[float] = []
        self.bwd_iats: List[float] = []

        # Headers and flags
        self.fwd_header_bytes = 0
        self.bwd_header_bytes = 0
        self.fwd_psh_flags = 0
        self.fin_flags = 0
        self.syn_flags = 0
        self.psh_flags = 0
        self.ack_flags = 0
        self.urg_flags = 0

        # TCP windows and segments
        self.init_win_fwd = 0
        self.init_win_bwd = 0
        self.act_data_pkt_fwd = 0
        self.min_seg_size_fwd = 32  # Default standard TCP/IP minimum

        # Idle and Active times
        self.idle_times: List[float] = []
        self.active_times: List[float] = []
        self.current_active_start = self.start_time
        self.is_closed = False

        # Add initial packet
        self.add_packet(first_packet)

    def add_packet(self, pkt):
        """Processes an incoming packet belonging to this bidirectional flow."""
        pkt_time = float(pkt.time)
        pkt_len = int(getattr(pkt, "len", len(pkt))) if hasattr(pkt, "len") and pkt.len else len(pkt)
        
        # Determine source IP and port reliably across IPv4 and IPv6
        if IP in pkt:
            pkt_src = pkt[IP].src
        elif scapy and IPv6 in pkt:
            pkt_src = pkt[IPv6].src
        else:
            pkt_src = getattr(pkt, "src", self.initiator_ip)

        sport = getattr(pkt, "sport", None)
        if sport is None:
            if TCP in pkt:
                sport = getattr(pkt[TCP], "sport", None)
            elif UDP in pkt:
                sport = getattr(pkt[UDP], "sport", None)

        is_fwd = (pkt_src == self.initiator_ip and (sport == self.initiator_port if (sport is not None and self.initiator_port is not None) else True))

        # Update flow IAT
        if self.last_time is not None:
            iat_sec = pkt_time - self.last_time
            if iat_sec >= 0:
                self.flow_iats.append(iat_sec * 1e6)  # microseconds
                if iat_sec > 2.5:  # 2.5-second idle threshold
                    self.idle_times.append(iat_sec * 1e6)
                    self.active_times.append((self.last_time - self.current_active_start) * 1e6)
                    self.current_active_start = pkt_time

        self.last_time = pkt_time

        # TCP specific fields
        tcp_flags = 0
        if IP in pkt:
            ihl = pkt[IP].ihl if getattr(pkt[IP], "ihl", None) is not None else 5
            ip_hdr_len = ihl * 4
        elif scapy and IPv6 in pkt:
            ip_hdr_len = 40
        else:
            ip_hdr_len = 20

        header_len = ip_hdr_len
        if TCP in pkt:
            tcp = pkt[TCP]
            tcp_flags = int(tcp.flags)
            dataofs = tcp.dataofs if getattr(tcp, "dataofs", None) is not None else 5
            header_len = ip_hdr_len + (dataofs * 4)
            payload_len = len(tcp.payload)

            if tcp_flags & 0x01: self.fin_flags += 1
            if tcp_flags & 0x02: self.syn_flags += 1
            if tcp_flags & 0x08: self.psh_flags += 1
            if tcp_flags & 0x10: self.ack_flags += 1
            if tcp_flags & 0x20: self.urg_flags += 1
            if tcp_flags & 0x04: self.is_closed = True  # RST closes flow
            if tcp_flags & 0x01: self.is_closed = True  # FIN closes flow

            if is_fwd:
                if self.fwd_packets == 0:
                    self.init_win_fwd = int(tcp.window) if getattr(tcp, "window", None) is not None else 0
                    self.min_seg_size_fwd = int(dataofs * 4)
                if tcp_flags & 0x08:
                    self.fwd_psh_flags += 1
                if payload_len > 0:
                    self.act_data_pkt_fwd += 1
            else:
                if self.bwd_packets == 0:
                    self.init_win_bwd = int(tcp.window) if getattr(tcp, "window", None) is not None else 0

        elif UDP in pkt:
            header_len = ip_hdr_len + 8

        # Direction-specific updates
        if is_fwd:
            self.fwd_packets += 1
            self.fwd_lengths.append(pkt_len)
            self.fwd_header_bytes += header_len
            if self.last_fwd_time is not None and self.fwd_packets > 1:
                fwd_iat = (pkt_time - self.last_fwd_time) * 1e6
                if fwd_iat >= 0:
                    self.fwd_iats.append(fwd_iat)
            self.last_fwd_time = pkt_time
        else:
            self.bwd_packets += 1
            self.bwd_lengths.append(pkt_len)
            self.bwd_header_bytes += header_len
            if self.last_bwd_time is not None:
                bwd_iat = (pkt_time - self.last_bwd_time) * 1e6
                if bwd_iat >= 0:
                    self.bwd_iats.append(bwd_iat)
            self.last_bwd_time = pkt_time

    def is_expired(self, current_time: Optional[float] = None, active_timeout: float = 10.0, idle_timeout: float = 2.5, now: Optional[float] = None) -> bool:
        """Determines if the flow should be flushed for real-time classification."""
        t = current_time if current_time is not None else (now if now is not None else time.time())
        if self.is_closed:
            return True
        idle_duration = t - self.last_time
        total_duration = t - self.start_time
        return idle_duration >= idle_timeout or total_duration >= active_timeout

    def to_feature_dict(self) -> Dict[str, float]:
        """Calculates exact 65 flow features aligned with CICFlowMeter schema."""
        dur_sec = max(self.last_time - self.start_time, 1e-6)
        dur_micro = dur_sec * 1e6

        tot_fwd_len = sum(self.fwd_lengths)
        tot_bwd_len = sum(self.bwd_lengths)
        tot_pkts = self.fwd_packets + self.bwd_packets
        all_lengths = self.fwd_lengths + self.bwd_lengths

        # Packet length statistics
        fwd_max = max(self.fwd_lengths) if self.fwd_lengths else 0.0
        fwd_min = min(self.fwd_lengths) if self.fwd_lengths else 0.0
        fwd_mean = float(np.mean(self.fwd_lengths)) if self.fwd_lengths else 0.0
        fwd_std = float(np.std(self.fwd_lengths)) if self.fwd_lengths else 0.0

        bwd_max = max(self.bwd_lengths) if self.bwd_lengths else 0.0
        bwd_min = min(self.bwd_lengths) if self.bwd_lengths else 0.0
        bwd_mean = float(np.mean(self.bwd_lengths)) if self.bwd_lengths else 0.0
        bwd_std = float(np.std(self.bwd_lengths)) if self.bwd_lengths else 0.0

        pkt_max = max(all_lengths) if all_lengths else 0.0
        pkt_min = min(all_lengths) if all_lengths else 0.0
        pkt_mean = float(np.mean(all_lengths)) if all_lengths else 0.0
        pkt_std = float(np.std(all_lengths)) if all_lengths else 0.0
        pkt_var = float(np.var(all_lengths)) if all_lengths else 0.0

        # Flow IAT statistics
        flow_iat_mean = float(np.mean(self.flow_iats)) if self.flow_iats else 0.0
        flow_iat_std = float(np.std(self.flow_iats)) if self.flow_iats else 0.0
        flow_iat_max = max(self.flow_iats) if self.flow_iats else 0.0
        flow_iat_min = min(self.flow_iats) if self.flow_iats else 0.0

        # Fwd IAT statistics
        fwd_iat_tot = sum(self.fwd_iats)
        fwd_iat_mean = float(np.mean(self.fwd_iats)) if self.fwd_iats else 0.0
        fwd_iat_std = float(np.std(self.fwd_iats)) if self.fwd_iats else 0.0
        fwd_iat_max = max(self.fwd_iats) if self.fwd_iats else 0.0
        fwd_iat_min = min(self.fwd_iats) if self.fwd_iats else 0.0

        # Bwd IAT statistics
        bwd_iat_tot = sum(self.bwd_iats)
        bwd_iat_mean = float(np.mean(self.bwd_iats)) if self.bwd_iats else 0.0
        bwd_iat_std = float(np.std(self.bwd_iats)) if self.bwd_iats else 0.0
        bwd_iat_max = max(self.bwd_iats) if self.bwd_iats else 0.0
        bwd_iat_min = min(self.bwd_iats) if self.bwd_iats else 0.0

        # Rates
        flow_bytes_s = (tot_fwd_len + tot_bwd_len) / dur_sec
        flow_pkts_s = tot_pkts / dur_sec
        fwd_pkts_s = self.fwd_packets / dur_sec
        bwd_pkts_s = self.bwd_packets / dur_sec

        down_up_ratio = float(self.bwd_packets / self.fwd_packets) if self.fwd_packets > 0 else 0.0
        avg_pkt_size = float((tot_fwd_len + tot_bwd_len) / tot_pkts) if tot_pkts > 0 else 0.0
        avg_fwd_seg_size = fwd_mean
        avg_bwd_seg_size = bwd_mean

        # Active & Idle times
        if not self.active_times:
            self.active_times.append(dur_micro)
        act_mean = float(np.mean(self.active_times))
        act_std = float(np.std(self.active_times))
        act_max = max(self.active_times)
        act_min = min(self.active_times)

        idle_mean = float(np.mean(self.idle_times)) if self.idle_times else 0.0
        idle_std = float(np.std(self.idle_times)) if self.idle_times else 0.0
        idle_max = max(self.idle_times) if self.idle_times else 0.0
        idle_min = min(self.idle_times) if self.idle_times else 0.0

        # Return exact 65 features
        return {
            "Protocol": float(self.protocol),
            "Flow Duration": float(dur_micro),
            "Total Fwd Packets": float(self.fwd_packets),
            "Total Backward Packets": float(self.bwd_packets),
            "Total Length of Fwd Packets": float(tot_fwd_len),
            "Total Length of Bwd Packets": float(tot_bwd_len),
            "Fwd Packet Length Max": float(fwd_max),
            "Fwd Packet Length Min": float(fwd_min),
            "Fwd Packet Length Mean": float(fwd_mean),
            "Fwd Packet Length Std": float(fwd_std),
            "Bwd Packet Length Max": float(bwd_max),
            "Bwd Packet Length Min": float(bwd_min),
            "Bwd Packet Length Mean": float(bwd_mean),
            "Bwd Packet Length Std": float(bwd_std),
            "Flow Bytes/s": float(flow_bytes_s),
            "Flow Packets/s": float(flow_pkts_s),
            "Flow IAT Mean": float(flow_iat_mean),
            "Flow IAT Std": float(flow_iat_std),
            "Flow IAT Max": float(flow_iat_max),
            "Flow IAT Min": float(flow_iat_min),
            "Fwd IAT Total": float(fwd_iat_tot),
            "Fwd IAT Mean": float(fwd_iat_mean),
            "Fwd IAT Std": float(fwd_iat_std),
            "Fwd IAT Max": float(fwd_iat_max),
            "Fwd IAT Min": float(fwd_iat_min),
            "Bwd IAT Total": float(bwd_iat_tot),
            "Bwd IAT Mean": float(bwd_iat_mean),
            "Bwd IAT Std": float(bwd_iat_std),
            "Bwd IAT Max": float(bwd_iat_max),
            "Bwd IAT Min": float(bwd_iat_min),
            "Fwd PSH Flags": float(self.fwd_psh_flags),
            "Fwd Header Length": float(self.fwd_header_bytes),
            "Bwd Header Length": float(self.bwd_header_bytes),
            "Fwd Packets/s": float(fwd_pkts_s),
            "Bwd Packets/s": float(bwd_pkts_s),
            "Min Packet Length": float(pkt_min),
            "Max Packet Length": float(pkt_max),
            "Packet Length Mean": float(pkt_mean),
            "Packet Length Std": float(pkt_std),
            "Packet Length Variance": float(pkt_var),
            "FIN Flag Count": float(self.fin_flags),
            "SYN Flag Count": float(self.syn_flags),
            "PSH Flag Count": float(self.psh_flags),
            "ACK Flag Count": float(self.ack_flags),
            "URG Flag Count": float(self.urg_flags),
            "Down/Up Ratio": float(down_up_ratio),
            "Average Packet Size": float(avg_pkt_size),
            "Avg Fwd Segment Size": float(avg_fwd_seg_size),
            "Avg Bwd Segment Size": float(avg_bwd_seg_size),
            "Subflow Fwd Packets": float(self.fwd_packets),
            "Subflow Fwd Bytes": float(tot_fwd_len),
            "Subflow Bwd Packets": float(self.bwd_packets),
            "Subflow Bwd Bytes": float(tot_bwd_len),
            "Init_Win_bytes_forward": float(self.init_win_fwd),
            "Init_Win_bytes_backward": float(self.init_win_bwd),
            "act_data_pkt_fwd": float(self.act_data_pkt_fwd),
            "min_seg_size_forward": float(self.min_seg_size_fwd),
            "Active Mean": float(act_mean),
            "Active Std": float(act_std),
            "Active Max": float(act_max),
            "Active Min": float(act_min),
            "Idle Mean": float(idle_mean),
            "Idle Std": float(idle_std),
            "Idle Max": float(idle_max),
            "Idle Min": float(idle_min)
        }


class LiveCapturePipeline:
    """Production live capture pipeline orchestrating sniffing, flow aggregation, and ML inference."""

    def __init__(
        self,
        interface: str,
        target_ip: Optional[str] = None,
        active_timeout: float = 15.0,
        idle_timeout: float = 5.0,
        binary_threshold: float = 0.65,
        on_flow_classified: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.interface = interface
        self.target_ip = target_ip
        self.active_timeout = active_timeout
        self.idle_timeout = idle_timeout
        self.binary_threshold = binary_threshold
        self.on_flow_classified = on_flow_classified

        self.is_running = False
        self.sniffer_thread: Optional[threading.Thread] = None
        self.flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # State tables (thread-safe)
        self.lock = threading.Lock()
        self.active_flows: Dict[Tuple, BidirectionalFlow] = {}
        
        # Telemetry stats
        self.stats = {
            "packets_captured": 0,
            "bytes_captured": 0,
            "flows_completed": 0,
            "benign_flows": 0,
            "malware_flows": 0,
            "uncertain_flows": 0,
            "alerts": []
        }

        # Lazy load predictor
        from src.predict_flow import FlowPredictor
        self.predictor = FlowPredictor()

    def _get_packet_key(self, pkt) -> Optional[Tuple]:
        """Extracts normalized bidirectional 5-tuple."""
        if IP in pkt:
            ip = pkt[IP]
            src_ip = ip.src
            dst_ip = ip.dst
            proto = ip.proto
        elif scapy and IPv6 in pkt:
            ip = pkt[IPv6]
            src_ip = ip.src
            dst_ip = ip.dst
            proto = ip.nh
        else:
            return None

        # Filter by target phone IP if configured
        if self.target_ip and self.target_ip not in (src_ip, dst_ip):
            return None

        src_port = 0
        dst_port = 0
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        else:
            return None

        # Check existing flow in either direction
        fwd_key = (src_ip, dst_ip, src_port, dst_port, proto)
        rev_key = (dst_ip, src_ip, dst_port, src_port, proto)

        with self.lock:
            if fwd_key in self.active_flows:
                return fwd_key
            if rev_key in self.active_flows:
                return rev_key

        return fwd_key

    def _process_packet(self, pkt):
        """Processes each sniffed packet."""
        if not self.is_running:
            return

        flow_key = self._get_packet_key(pkt)
        if not flow_key:
            return

        with self.lock:
            self.stats["packets_captured"] += 1
            self.stats["bytes_captured"] += len(pkt)

            if flow_key not in self.active_flows:
                self.active_flows[flow_key] = BidirectionalFlow(pkt, flow_key)
            else:
                self.active_flows[flow_key].add_packet(pkt)

    def _flush_expired_flows_loop(self):
        """Background daemon periodically checking timeouts and running ML inference."""
        while not self._stop_event.is_set():
            time.sleep(1.0)
            now = time.time()
            flows_to_classify = []

            with self.lock:
                expired_keys = []
                for key, flow in list(self.active_flows.items()):
                    if flow.is_expired(now, self.active_timeout, self.idle_timeout):
                        expired_keys.append(key)
                        flows_to_classify.append((key, flow))
                
                for k in expired_keys:
                    del self.active_flows[k]

            # Run inference outside lock
            for key, flow in flows_to_classify:
                self._classify_and_alert(key, flow)

    def _classify_and_alert(self, flow_key: Tuple, flow: BidirectionalFlow):
        """Generates feature dict, runs binary & multiclass inference, and logs alerts."""
        feat_dict = flow.to_feature_dict()
        t0 = time.perf_counter()
        
        # 1. Binary prediction
        bin_res = self.predictor.predict_flow(feat_dict, mode="binary")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Apply configurable threshold
        malware_prob = bin_res["probabilities"].get("MALWARE_ASSOCIATED", 0.0)
        is_malware = malware_prob >= self.binary_threshold
        status = "MALWARE_ASSOCIATED" if is_malware else "BENIGN"
        
        # 2. Multiclass prediction if suspicious
        multiclass_label = "N/A"
        multiclass_conf = 0.0
        if is_malware:
            multi_res = self.predictor.predict_flow(feat_dict, mode="multiclass")
            multiclass_label = multi_res["label"]
            multiclass_conf = multi_res["confidence"]

        # Threat Alert Policy (Requirements 2 & 7):
        VALID_THREAT_FAMILIES = {"RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"}
        is_threat = bool(is_malware and multiclass_label in VALID_THREAT_FAMILIES and multiclass_label != "BENIGN")

        flow_record = {
            "id": f"{flow_key[0]}_{flow_key[2]}_{flow_key[1]}_{flow_key[3]}_{flow_key[4]}_{int(flow.start_time * 1000)}",
            "timestamp": time.strftime("%H:%M:%S", time.localtime(flow.start_time)),
            "src_ip": flow_key[0],
            "dst_ip": flow_key[1],
            "src_port": flow_key[2],
            "dst_port": flow_key[3],
            "protocol": "TCP" if flow_key[4] == 6 else ("UDP" if flow_key[4] == 17 else str(flow_key[4])),
            "packets": flow.fwd_packets + flow.bwd_packets,
            "bytes": sum(flow.fwd_lengths) + sum(flow.bwd_lengths),
            "status": status,
            "is_threat": is_threat,
            "detection_source": "ML_MODEL",
            "event_type": "ML_THREAT_ALERT" if is_threat else "NORMAL_FLOW",
            "malware_probability": round(malware_prob, 4),
            "multiclass_category": multiclass_label,
            "multiclass_confidence": round(multiclass_conf, 4),
            "inference_latency_ms": round(latency_ms, 2)
        }

        with self.lock:
            self.stats["flows_completed"] += 1
            if is_threat:
                self.stats["malware_flows"] += 1
                # Alert deduplication: keep last 50 alerts
                self.stats["alerts"].insert(0, flow_record)
                if len(self.stats["alerts"]) > 50:
                    self.stats["alerts"].pop()
            else:
                self.stats["benign_flows"] += 1

        if self.on_flow_classified:
            try:
                self.on_flow_classified(flow_record)
            except Exception:
                pass

    def start(self):
        """Starts live sniffing and flush worker daemon."""
        if self.is_running:
            return

        self.is_running = True
        self._stop_event.clear()

        # Start flush thread
        self.flush_thread = threading.Thread(target=self._flush_expired_flows_loop, daemon=True)
        self.flush_thread.start()

        # Start Scapy sniffer in separate thread
        def sniffer_worker():
            try:
                scapy.sniff(
                    iface=self.interface,
                    prn=self._process_packet,
                    stop_filter=lambda p: self._stop_event.is_set(),
                    store=False
                )
            except Exception as e:
                print(f"[!] Sniffer thread encountered error: {e}")
                self.is_running = False

        self.sniffer_thread = threading.Thread(target=sniffer_worker, daemon=True)
        self.sniffer_thread.start()

    def stop(self):
        """Gracefully halts live capture and flushes remaining active flows."""
        if not self.is_running:
            return
        self.is_running = False
        self._stop_event.set()

        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=2.0)
        if self.sniffer_thread and self.sniffer_thread.is_alive():
            self.sniffer_thread.join(timeout=2.0)

        # Final flush of all remaining flows
        with self.lock:
            remaining = list(self.active_flows.items())
            self.active_flows.clear()

        for key, flow in remaining:
            self._classify_and_alert(key, flow)

    def get_telemetry(self) -> Dict[str, Any]:
        """Thread-safe snapshot of live metrics."""
        with self.lock:
            return {
                "is_running": self.is_running,
                "interface": self.interface,
                "target_ip": self.target_ip or "Any (All Devices)",
                "packets_captured": self.stats["packets_captured"],
                "bytes_captured": self.stats["bytes_captured"],
                "flows_completed": self.stats["flows_completed"],
                "active_flows_tracking": len(self.active_flows),
                "benign_flows": self.stats["benign_flows"],
                "malware_flows": self.stats["malware_flows"],
                "alerts_count": len(self.stats["alerts"]),
                "recent_alerts": list(self.stats["alerts"][:15])
            }
