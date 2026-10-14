"""Windows Packet Capture Diagnostic & Interface Enumerator.

Diagnoses Npcap/WinPcap status, lists network interfaces with IP/MAC details,
identifies Mobile Hotspot virtual adapters, and verifies live packet visibility.
"""

import sys
import time
import argparse
from typing import List, Dict, Any

try:
    import scapy.all as scapy
except ImportError:
    print("[!] ERROR: Scapy is not installed. Please run: pip install scapy")
    sys.exit(1)

# Windows-only interface discovery; Linux must not import winreg.
if sys.platform == "win32":
    try:
        from scapy.arch.windows import get_windows_if_list
    except ImportError:
        get_windows_if_list = None
else:
    get_windows_if_list = None


def check_npcap_status() -> bool:
    """Checks whether Npcap/WinPcap is loaded and functional."""
    has_pcap = getattr(scapy.conf, "use_pcap", False)
    pcap_provider = getattr(scapy.conf, "pcap_provider", "Unknown")
    return has_pcap or pcap_provider != "None"


def enumerate_interfaces() -> List[Dict[str, Any]]:
    """Return Scapy network interfaces on Windows or Linux."""
    results = []

    # Use Scapy's cross-platform interface collection.
    for iface_key, iface in scapy.conf.ifaces.items():
        name = str(getattr(iface, "name", "Unknown") or "Unknown")
        desc = str(getattr(iface, "description", "") or "")
        guid = str(getattr(iface, "guid", "") or "")
        mac = str(getattr(iface, "mac", "") or "")
        raw_ips = getattr(iface, "ips", [])
        extracted_ips: List[str] = []

        if isinstance(raw_ips, dict):
            for family in (4, 6):
                values = raw_ips.get(family, [])
                if isinstance(values, (list, tuple, set)):
                    for ip in values:
                        if isinstance(ip, str) and ip and ip not in extracted_ips:
                            extracted_ips.append(ip)
        elif isinstance(raw_ips, (list, tuple, set)):
            for ip in raw_ips:
                if isinstance(ip, str) and ip and ip not in extracted_ips:
                    extracted_ips.append(ip)

        ip_val = getattr(iface, "ip", None)
        if isinstance(ip_val, str) and ip_val not in ("", "0.0.0.0") and ip_val not in extracted_ips:
            extracted_ips.insert(0, ip_val)

        lower_desc = desc.lower()
        lower_name = name.lower()
        adapter_type = "Standard"
        if "wi-fi" in lower_name or "wireless" in lower_desc:
            adapter_type = "Wi-Fi"
        elif "hotspot" in lower_name or "virtual" in lower_desc or "loopback" in lower_desc:
            adapter_type = "Virtual / Hotspot Candidate" if "hotspot" in lower_name or "virtual" in lower_desc else "Loopback"
        elif "ethernet" in lower_desc or "ethernet" in lower_name:
            adapter_type = "Ethernet"

        primary_ip = extracted_ips[0] if extracted_ips else (ip_val if isinstance(ip_val, str) else "0.0.0.0")
        results.append({
            "name": name,
            "description": desc,
            "guid": guid,
            "mac": mac,
            "ips": extracted_ips,
            "ip": primary_ip,
            "is_up": bool(extracted_ips and primary_ip not in ("0.0.0.0", "127.0.0.1")),
            "npf_device": str(iface_key),
            "type": adapter_type,
            "scapy_name": str(iface_key),
        })

    return results


# Alias for backward and cross-module compatibility
list_available_interfaces = enumerate_interfaces


def print_interfaces():
    """Prints a formatted table of all interfaces for user selection."""
    ifaces = enumerate_interfaces()
    npcap_ok = check_npcap_status()

    print("\n" + "="*75)
    print("  MOBILESHIELD: WINDOWS PACKET CAPTURE DIAGNOSTIC")
    print("="*75)
    print(f"[*] Npcap / WinPcap Driver Status: {'ACTIVE / DETECTED' if npcap_ok else 'NOT DETECTED (Install Npcap in WinPcap mode)'}")
    print(f"[*] Total Interfaces Found: {len(ifaces)}\n")

    print(f"{'Idx':<4} | {'Adapter Name':<30} | {'Type':<20} | {'IP Address(es)'}")
    print("-" * 75)

    hotspot_candidates = []
    for idx, iface in enumerate(ifaces):
        # Safely normalize all items to strings before joining
        valid_ips = [str(ip) for ip in iface["ips"] if not isinstance(ip, (int, bool)) and ip]
        ip_str = ", ".join(valid_ips) if valid_ips else "No IP"
        print(f"{idx:<4} | {iface['name'][:30]:<30} | {iface['type']:<20} | {ip_str}")
        if iface["type"] == "Virtual / Hotspot Candidate" or "192.168.137." in ip_str:
            hotspot_candidates.append((idx, iface))

    print("\n" + "="*75)
    if hotspot_candidates:
        print("[*] DETECTED MOBILE HOTSPOT CANDIDATES:")
        for idx, iface in hotspot_candidates:
            print(f"    -> Index [{idx}]: {iface['name']} ({iface['description']}) IP: {iface['ips']}")
        print("    (Windows Mobile Hotspot typically uses 192.168.137.1 or 'Local Area Connection* X')")
    else:
        print("[*] Note: Windows Mobile Hotspot virtual adapter will appear once Mobile Hotspot is ENABLED in Windows Settings.")
    print("="*75 + "\n")


def test_live_capture(interface_identifier: str = None, duration_sec: int = 10):
    """Sniffs packets on specified interface and reports packet/protocol counters."""
    ifaces = enumerate_interfaces()
    selected_iface = None

    if interface_identifier is None:
        # Auto-detect hotspot or fallback to first Wi-Fi / Ethernet
        for iface in ifaces:
            if "192.168.137." in str(iface["ips"]) or "hotspot" in iface["name"].lower():
                selected_iface = iface
                break
        if selected_iface is None:
            for iface in ifaces:
                if iface["type"] in ("Wi-Fi", "Ethernet") and iface["ips"]:
                    selected_iface = iface
                    break
    else:
        # Lookup by index or name substring
        if interface_identifier.isdigit():
            idx = int(interface_identifier)
            if 0 <= idx < len(ifaces):
                selected_iface = ifaces[idx]
        else:
            for iface in ifaces:
                if interface_identifier.lower() in iface["name"].lower() or interface_identifier.lower() in iface["description"].lower():
                    selected_iface = iface
                    break

    if selected_iface is None:
        print(f"[!] Could not find matching interface for '{interface_identifier}'. Available:")
        print_interfaces()
        return

    target_name = selected_iface["name"]
    print(f"\n[*] Starting test capture on: [{target_name}] ({selected_iface['description']})")
    print(f"[*] Assigned IPs: {selected_iface['ips']}")
    print(f"[*] Listening for {duration_sec} seconds... (Generate traffic on your phone now!)\n")

    stats = {
        "packets": 0,
        "bytes": 0,
        "tcp": 0,
        "udp": 0,
        "dns": 0,
        "other": 0,
        "hosts": set()
    }

    t_start = time.time()

    def packet_handler(pkt):
        stats["packets"] += 1
        stats["bytes"] += len(pkt)
        if scapy.TCP in pkt:
            stats["tcp"] += 1
        elif scapy.UDP in pkt:
            stats["udp"] += 1
            if pkt[scapy.UDP].dport == 53 or pkt[scapy.UDP].sport == 53:
                stats["dns"] += 1
        else:
            stats["other"] += 1

        if scapy.IP in pkt:
            stats["hosts"].add(pkt[scapy.IP].src)
            stats["hosts"].add(pkt[scapy.IP].dst)

    try:
        scapy.sniff(
            iface=selected_iface["scapy_name"],
            prn=packet_handler,
            timeout=duration_sec,
            store=False
        )
    except PermissionError:
        print("[!] ERROR: Administrator privileges required for raw packet sniffing.")
        return
    except Exception as e:
        print(f"[!] Sniffing error: {e}")
        return

    elapsed = round(time.time() - t_start, 2)
    print("\n" + "="*50)
    print("  CAPTURE DIAGNOSTIC RESULTS")
    print("="*50)
    print(f"Duration:       {elapsed} seconds")
    print(f"Total Packets:  {stats['packets']}")
    print(f"Total Volume:   {stats['bytes'] / 1024:.2f} KB ({stats['bytes']} bytes)")
    print(f"TCP Packets:    {stats['tcp']}")
    print(f"UDP Packets:    {stats['udp']}")
    print(f"DNS Packets:    {stats['dns']}")
    print(f"Other Packets:  {stats['other']}")
    print(f"Unique IPs:     {len(stats['hosts'])} observed: {list(stats['hosts'])[:6]}")
    print("="*50)

    if stats["packets"] > 0:
        print("[+] SUCCESS: Packets captured successfully on this interface!")
    else:
        print("[!] WARNING: 0 packets captured. Check if phone is connected and generating traffic.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Windows Packet Capture Diagnostic")
    parser.add_argument("--list", action="store_true", help="List all available network adapters")
    parser.add_argument("--test", metavar="IFACE", help="Test live capture on interface index or name")
    parser.add_argument("--duration", type=int, default=10, help="Test capture duration in seconds")
    args = parser.parse_args()

    if args.test is not None:
        test_live_capture(args.test, duration_sec=args.duration)
    else:
        print_interfaces()
        if len(sys.argv) == 1:
            print("[*] To test live capture on an adapter, run: python src/capture_diagnostics.py --test <INDEX>")
