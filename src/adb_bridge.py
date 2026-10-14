
"""MobileShield ADB Bridge.

Discovers connected Android devices via ADB, queries hardware/OS metadata,
manages USB reverse-tunnels (adb reverse), and controls companion app lifecycle.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Rate-limit repeated ADB-unavailable log messages (one per 60 s max).
_last_adb_warn_time: float = 0.0
_ADB_WARN_INTERVAL: float = 60.0


def find_adb() -> Optional[str]:
    """Find the ADB executable on Windows or Linux."""

    # Prefer an explicitly configured executable.
    configured_path = os.environ.get("ADB_EXECUTABLE")
    if configured_path:
        if Path(configured_path).is_file():
            return configured_path
        print(f"[!] ADB_EXECUTABLE does not exist: {configured_path}")

    # Search PATH (works on Windows and Linux).
    adb_path = shutil.which("adb")
    if adb_path:
        return adb_path

    # Common platform-tools locations.
    if os.name == "nt":
        candidates = [
            r"C:\platform-tools\adb.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"),
        ]
    else:
        candidates = [
            "/usr/bin/adb",
            "/usr/local/bin/adb",
            "/opt/android-sdk/platform-tools/adb",
            os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
        ]

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate

    return None


ADB_EXECUTABLE = find_adb()


def run_adb(
    args: List[str],
    device_id: Optional[str] = None,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess:
    """Run an ADB command safely with a timeout."""

    if not ADB_EXECUTABLE:
        raise FileNotFoundError(
            "ADB executable not found. Install Android platform-tools or "
            "set the ADB_EXECUTABLE environment variable."
        )

    cmd = [ADB_EXECUTABLE]

    if device_id:
        cmd.extend(["-s", device_id])

    cmd.extend(args)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def list_connected_devices() -> List[Dict[str, Any]]:
    """Discover connected Android devices and read their metadata.

    Returns an empty list (without raising) when ADB is unavailable so that
    callers can degrade gracefully.  Repeated warnings are rate-limited to
    one message per 60 seconds to avoid flooding container logs.
    """

    global _last_adb_warn_time
    devices: List[Dict[str, Any]] = []

    if not ADB_EXECUTABLE:
        now = time.monotonic()
        if now - _last_adb_warn_time >= _ADB_WARN_INTERVAL:
            print(
                "[!] ADB is unavailable on this host. "
                "Install platform-tools or set ADB_EXECUTABLE. "
                "(This message repeats at most once per 60 s.)"
            )
            _last_adb_warn_time = now
        return devices

    try:
        res = run_adb(["devices", "-l"])

        if res.returncode != 0:
            print(f"[!] ADB device discovery failed: {res.stderr.strip()}")
            return devices

        for line in res.stdout.splitlines()[1:]:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            serial = parts[0]
            status = parts[1]

            metadata: Dict[str, str] = {}
            for part in parts[2:]:
                if ":" in part:
                    key, value = part.split(":", 1)
                    metadata[key] = value

            model = metadata.get("model", "Unknown")
            codename = metadata.get("device", "Unknown")
            android_version = "Unknown"
            battery_level = "Unknown"

            if status == "device":
                try:
                    version_res = run_adb(
                        ["shell", "getprop", "ro.build.version.release"],
                        device_id=serial,
                        timeout=5.0,
                    )
                    if version_res.returncode == 0:
                        android_version = version_res.stdout.strip() or "Unknown"
                except (OSError, subprocess.SubprocessError):
                    pass

                try:
                    model_res = run_adb(
                        ["shell", "getprop", "ro.product.model"],
                        device_id=serial,
                        timeout=5.0,
                    )
                    if model_res.returncode == 0 and model_res.stdout.strip():
                        model = model_res.stdout.strip()
                except (OSError, subprocess.SubprocessError):
                    pass

                try:
                    battery_res = run_adb(
                        ["shell", "dumpsys", "battery"],
                        device_id=serial,
                        timeout=5.0,
                    )
                    if battery_res.returncode == 0:
                        for battery_line in battery_res.stdout.splitlines():
                            battery_line = battery_line.strip()
                            if battery_line.startswith("level:"):
                                battery_level = battery_line.split(":", 1)[1].strip()
                                break
                except (OSError, subprocess.SubprocessError):
                    pass

            devices.append(
                {
                    "serial": serial,
                    "status": status,
                    "model": model,
                    "device": codename,
                    "android_version": android_version,
                    "battery_level": battery_level,
                }
            )

    except (OSError, subprocess.SubprocessError) as exc:
        now = time.monotonic()
        if now - _last_adb_warn_time >= _ADB_WARN_INTERVAL:
            print(f"[!] Error discovering ADB devices: {exc} "
                  "(This message repeats at most once per 60 s.)")
            _last_adb_warn_time = now

    return devices


def setup_usb_reverse_tunnel(
    device_id: str,
    local_port: int = 8000,
) -> bool:
    """Forward the phone's localhost port to the connected ADB host."""

    try:
        res = run_adb(
            ["reverse", f"tcp:{local_port}", f"tcp:{local_port}"],
            device_id=device_id,
        )
        if res.returncode != 0:
            print(f"[!] ADB reverse failed: {res.stderr.strip()}")
        return res.returncode == 0

    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[!] Failed to setup ADB reverse for {device_id}: {exc}")
        return False


def remove_usb_reverse_tunnel(
    device_id: str,
    local_port: int = 8000,
) -> bool:
    """Remove an ADB reverse tunnel."""

    try:
        res = run_adb(
            ["reverse", "--remove", f"tcp:{local_port}"],
            device_id=device_id,
        )
        return res.returncode == 0

    except (OSError, subprocess.SubprocessError):
        return False


def is_companion_installed(
    device_id: str,
    package_name: str = "com.mobileshield.companion",
) -> bool:
    """Check whether the companion app is installed."""

    try:
        res = run_adb(
            ["shell", "pm", "list", "packages", package_name],
            device_id=device_id,
        )
        return res.returncode == 0 and f"package:{package_name}" in res.stdout

    except (OSError, subprocess.SubprocessError):
        return False


def install_companion_apk(device_id: str, apk_path: str) -> bool:
    """Install or update the companion APK."""

    try:
        res = run_adb(
            ["install", "-r", apk_path],
            device_id=device_id,
            timeout=60.0,
        )

        if res.returncode != 0:
            print(f"[!] APK installation failed: {res.stderr.strip()}")

        return res.returncode == 0 and "Success" in res.stdout

    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[!] Failed to install APK on {device_id}: {exc}")
        return False


def launch_companion_app(
    device_id: str,
    package_name: str = "com.mobileshield.companion",
) -> bool:
    """Launch the companion app."""

    try:
        res = run_adb(
            [
                "shell",
                "monkey",
                "-p",
                package_name,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ],
            device_id=device_id,
        )
        return res.returncode == 0

    except (OSError, subprocess.SubprocessError):
        return False


def stop_companion_app(
    device_id: str,
    package_name: str = "com.mobileshield.companion",
) -> bool:
    """Force-stop the companion app."""

    try:
        res = run_adb(
            ["shell", "am", "force-stop", package_name],
            device_id=device_id,
        )
        return res.returncode == 0

    except (OSError, subprocess.SubprocessError):
        return False
