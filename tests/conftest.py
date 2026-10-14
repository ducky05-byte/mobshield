import hashlib
import os
import secrets
import sqlite3
import tempfile


_auth_directory = tempfile.mkdtemp(prefix="mobileshield-tests-")
os.environ["ADMIN_API_TOKEN"] = secrets.token_hex(32)
os.environ["RELAY_SECRET"] = secrets.token_hex(32)
os.environ["DEVICE_PAIRING_CODE"] = secrets.token_hex(32)
os.environ["DEVICE_AUTH_DB_PATH"] = os.path.join(
    _auth_directory, "device_auth.sqlite3"
)

from src.companion_api import ADMIN_API_TOKEN, DEVICE_AUTH_DB

with sqlite3.connect(DEVICE_AUTH_DB) as _connection:
    _token_hash = hashlib.sha256(ADMIN_API_TOKEN.encode("utf-8")).hexdigest()
    _connection.executemany(
        "INSERT OR REPLACE INTO device_tokens (device_id, token_hash, created_at) "
        "VALUES (?, ?, 0)",
        [
            (device_id, _token_hash)
            for device_id in (
                "RZCX221W2EE",
                "TEST_DEVICE",
                "TEST_SAMSUNG_A14",
                "TEST_PHONE",
            )
        ],
    )
