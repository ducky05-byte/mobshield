import os, sys, json, time, signal, urllib.request, urllib.error, threading
import ipaddress
import re
import ssl
from urllib.parse import urlsplit
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from src.adb_bridge import list_connected_devices, setup_usb_reverse_tunnel, ADB_EXECUTABLE

RELAY_TARGET_URL = os.environ.get('RELAY_TARGET_URL', 'http://127.0.0.1:8000').rstrip('/')
RELAY_SECRET = os.environ.get('RELAY_SECRET', '')
RELAY_INTERVAL_S = float(os.environ.get('RELAY_INTERVAL_S', '5'))
RELAY_SETUP_TUNNEL = os.environ.get('RELAY_SETUP_TUNNEL', '1') == '1'

_secret_pairs = {
    RELAY_SECRET[index:index + 2] for index in range(len(RELAY_SECRET) - 1)
}
_secret_repeats = any(
    RELAY_SECRET == RELAY_SECRET[:period] * (64 // period)
    for period in (1, 2, 4, 8, 16, 32)
)
if (
    not re.fullmatch(r'[0-9a-f]{64}', RELAY_SECRET)
    or len(_secret_pairs) < 32
    or _secret_repeats
):
    raise RuntimeError(
        'RELAY_SECRET is missing or invalid. Set it in the project-root .env '
        'or process environment as 64 random lowercase hexadecimal characters.'
    )

_target = urlsplit(RELAY_TARGET_URL)
_target_host = (_target.hostname or '').lower()
try:
    _is_loopback = _target_host == 'localhost' or ipaddress.ip_address(_target_host).is_loopback
except ValueError:
    _is_loopback = False
if (
    _target.scheme not in ('https', 'http')
    or not _target_host
    or _target.username
    or _target.password
    or _target.query
    or _target.fragment
    or (_target.scheme != 'https' and not (_target.scheme == 'http' and _is_loopback))
):
    raise RuntimeError(
        'RELAY_TARGET_URL must use HTTPS for non-loopback hosts. Plain HTTP '
        'is permitted only to localhost/127.0.0.1 for an encrypted tunnel.'
    )

class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_http_opener = urllib.request.build_opener(
    _NoRedirectHandler(),
    urllib.request.HTTPSHandler(context=ssl.create_default_context()),
)

_stop_event = threading.Event()
_tunnel_established = set()


def _post_device_report(payload):
    url = RELAY_TARGET_URL + '/api/device/report'
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=body, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-Relay-Secret', RELAY_SECRET)
    try:
        with _http_opener.open(req, timeout=8.0) as resp:
            return resp.status in (200, 204)
    except Exception as exc:
        print('[relay] Error posting report:', exc, flush=True)
        return False


def _poll_and_report():
    try:
        devices = list_connected_devices()
    except Exception as exc:
        print('[relay] ADB error:', exc, flush=True)
        devices = []
    if RELAY_SETUP_TUNNEL:
        for dev in devices:
            serial = dev.get('serial', '')
            if serial and dev.get('status') == 'device' and serial not in _tunnel_established:
                ok = setup_usb_reverse_tunnel(serial, 8000)
                if ok:
                    print('[relay] Tunnel established for', serial, flush=True)
                    _tunnel_established.add(serial)
    live = {d.get('serial') for d in devices}
    _tunnel_established.intersection_update(live)
    ok = _post_device_report({'agent_time': time.time(), 'adb_available': ADB_EXECUTABLE is not None, 'devices': devices})
    sw = 'OK' if ok else 'FAILED'
    print(f'[relay] Report sent ({sw}) -- {len(devices)} device(s)', flush=True)


def run_agent():
    adb_info = ADB_EXECUTABLE or '(not found)'
    tm = 'auto' if RELAY_SETUP_TUNNEL else 'disabled'
    print('=' * 60, flush=True)
    print('MobileShield Windows Device Relay Agent', flush=True)
    print(f'  Target  : {RELAY_TARGET_URL}', flush=True)
    print(f'  Interval: {RELAY_INTERVAL_S}s', flush=True)
    print(f'  ADB     : {adb_info}', flush=True)
    print(f'  Tunnel  : {tm}', flush=True)
    print('=' * 60, flush=True)
    if not ADB_EXECUTABLE:
        print('[relay] WARNING: ADB not found. Install platform-tools or set ADB_EXECUTABLE.', flush=True)
    while not _stop_event.is_set():
        _poll_and_report()
        _stop_event.wait(timeout=RELAY_INTERVAL_S)
    print('[relay] Agent stopped.', flush=True)


def _handle_signal(signum, frame):
    print(f'[relay] Signal {signum} -- stopping.', flush=True)
    _stop_event.set()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, _handle_signal)
    run_agent()