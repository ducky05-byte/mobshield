"""Download and validate the trained model bundle used by live inference."""

import hashlib
import hmac
import http.client
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_MODEL_ARTIFACTS = (
    "preprocessor.joblib",
    "binary/best_model.joblib",
    "multiclass/best_model.joblib",
    "feature_names.json",
    "label_mapping.json",
)
MAX_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024
MAX_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_TOTAL_ARTIFACT_BYTES = 1024 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60
VALIDATED_BUNDLE_MARKER = ".artifact-sha256"
PENDING_BUNDLE_MARKER = ".artifact-sha256.pending"
EXPECTED_LABEL_MAPPINGS = {
    "binary": {"BENIGN": 0, "MALWARE_ASSOCIATED": 1},
    "multiclass": {
        "BENIGN": 0,
        "ADWARE": 1,
        "RANSOMWARE": 2,
        "SCAREWARE": 3,
        "SMSMALWARE": 4,
    },
}


class ModelArtifactError(RuntimeError):
    """Raised when required model files cannot be safely prepared for inference."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        return None


def resolve_model_dir(model_dir: Optional[Path | str] = None) -> Path:
    """Resolve the configured model directory, defaulting to the local project."""
    configured_dir = model_dir or os.environ.get("MODEL_DIR")
    path = Path(configured_dir) if configured_dir else PROJECT_ROOT / "models"
    return path.expanduser().resolve()


def load_model_metadata(model_dir: Path | str) -> tuple[list[str], dict[str, Any]]:
    """Validate and load the fixed feature order and classifier label mappings."""
    model_dir = Path(model_dir)
    missing = [
        name
        for name in REQUIRED_MODEL_ARTIFACTS
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise ModelArtifactError(
            "Required model artifacts are missing from "
            f"{model_dir}: {', '.join(missing)}."
        )

    try:
        with (model_dir / "feature_names.json").open(
            "r", encoding="utf-8"
        ) as feature_file:
            feature_names = json.load(feature_file)
        with (model_dir / "label_mapping.json").open(
            "r", encoding="utf-8"
        ) as mapping_file:
            label_mappings = json.load(mapping_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(
            "Could not read model feature or label metadata; verify the "
            "artifact bundle."
        ) from exc

    if (
        not isinstance(feature_names, list)
        or len(feature_names) != 65
        or any(not isinstance(name, str) or not name for name in feature_names)
        or len(set(feature_names)) != 65
    ):
        raise ModelArtifactError(
            "models/feature_names.json must contain the ordered 65-feature "
            "inference schema."
        )

    if not isinstance(label_mappings, dict) or any(
        label_mappings.get(mode) != expected
        for mode, expected in EXPECTED_LABEL_MAPPINGS.items()
    ):
        raise ModelArtifactError(
            "models/label_mapping.json does not match the supported binary "
            "and multiclass model classes."
        )

    return feature_names, label_mappings


def _download_bundle(
    url: str,
    destination: Path,
    token: Optional[str],
) -> str:
    parsed_url = urllib.parse.urlsplit(url)
    if (
        parsed_url.scheme != "https"
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise ModelArtifactError(
            "MODEL_ARTIFACT_URL must be an HTTPS download URL without "
            "embedded credentials."
        )

    headers = {"Accept": "application/gzip, application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    digest = hashlib.sha256()
    total_bytes = 0

    try:
        opener = urllib.request.build_opener(_NoRedirectHandler())
        with opener.open(
            request, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response, destination.open("wb") as bundle_file:
            status = getattr(response, "status", 200)
            if status != 200:
                raise ModelArtifactError(
                    f"Model artifact storage returned HTTP {status}."
                )
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_BUNDLE_BYTES:
                    raise ModelArtifactError(
                        "Model artifact bundle exceeds the 2 GiB download limit."
                    )
                digest.update(chunk)
                bundle_file.write(chunk)
    except ModelArtifactError:
        raise
    except urllib.error.HTTPError as exc:
        raise ModelArtifactError(
            f"Model artifact storage returned HTTP {exc.code}."
        ) from None
    except (
        TimeoutError,
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
    ):
        raise ModelArtifactError(
            "Model artifact download failed or timed out."
        ) from None

    if total_bytes == 0:
        raise ModelArtifactError("Model artifact storage returned an empty bundle.")
    return digest.hexdigest()


def _extract_bundle(bundle_path: Path, staging_dir: Path) -> None:
    required_paths = {
        f"models/{relative_path}": relative_path
        for relative_path in REQUIRED_MODEL_ARTIFACTS
    }
    allowed_directories = {"models", "models/binary", "models/multiclass"}
    found: set[str] = set()
    total_artifact_bytes = 0

    try:
        with tarfile.open(bundle_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > 64:
                raise ModelArtifactError(
                    "Model artifact bundle contains too many archive entries."
                )

            for member in members:
                raw_path = PurePosixPath(member.name)
                path_parts = tuple(
                    part for part in raw_path.parts if part not in ("", ".")
                )
                if raw_path.is_absolute() or ".." in path_parts:
                    raise ModelArtifactError(
                        "Model artifact bundle contains an unsafe archive path."
                    )
                normalized_path = "/".join(path_parts)

                if normalized_path in allowed_directories and member.isdir():
                    continue
                relative_path = required_paths.get(normalized_path)
                if relative_path is None:
                    raise ModelArtifactError(
                        "Model artifact bundle contains an unexpected file; "
                        "package only the required inference artifacts."
                    )
                if not member.isfile() or member.size <= 0:
                    raise ModelArtifactError(
                        f"Model artifact '{relative_path}' is not a regular, "
                        "non-empty file."
                    )
                if member.size > MAX_MEMBER_BYTES:
                    raise ModelArtifactError(
                        f"Model artifact '{relative_path}' exceeds the 1 GiB limit."
                    )
                total_artifact_bytes += member.size
                if total_artifact_bytes > MAX_TOTAL_ARTIFACT_BYTES:
                    raise ModelArtifactError(
                        "Uncompressed model artifacts exceed the 1 GiB limit."
                    )
                if relative_path in found:
                    raise ModelArtifactError(
                        f"Model artifact '{relative_path}' appears more than once."
                    )

                source = archive.extractfile(member)
                if source is None:
                    raise ModelArtifactError(
                        f"Could not read model artifact '{relative_path}'."
                    )
                target = staging_dir / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                copied_bytes = 0
                with source, target.open("wb") as target_file:
                    while chunk := source.read(1024 * 1024):
                        copied_bytes += len(chunk)
                        if copied_bytes > member.size:
                            raise ModelArtifactError(
                                f"Model artifact '{relative_path}' exceeded its "
                                "declared archive size."
                            )
                        target_file.write(chunk)
                if copied_bytes != member.size:
                    raise ModelArtifactError(
                        f"Model artifact '{relative_path}' was incomplete."
                    )
                found.add(relative_path)
    except ModelArtifactError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ModelArtifactError(
            "Model artifact bundle is not a readable gzip-compressed tar archive."
        ) from exc

    missing = set(REQUIRED_MODEL_ARTIFACTS) - found
    if missing:
        raise ModelArtifactError(
            "Model artifact bundle is missing required files: "
            + ", ".join(sorted(missing))
            + "."
        )

    load_model_metadata(staging_dir)


def ensure_model_artifacts(
    model_dir: Optional[Path | str] = None,
    *,
    artifact_url: Optional[str] = None,
    artifact_sha256: Optional[str] = None,
    artifact_token: Optional[str] = None,
) -> Path:
    """Use existing local artifacts or atomically install a verified tar bundle."""
    destination = resolve_model_dir(model_dir)
    artifact_url = (
        os.environ.get("MODEL_ARTIFACT_URL")
        if artifact_url is None
        else artifact_url
    )
    artifact_sha256 = (
        os.environ.get("MODEL_ARTIFACT_SHA256")
        if artifact_sha256 is None
        else artifact_sha256
    )
    artifact_token = (
        os.environ.get("MODEL_ARTIFACT_TOKEN")
        if artifact_token is None
        else artifact_token
    )
    if bool(artifact_url) != bool(artifact_sha256):
        raise ModelArtifactError(
            "Configure both MODEL_ARTIFACT_URL and MODEL_ARTIFACT_SHA256."
        )
    if artifact_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", artifact_sha256):
        raise ModelArtifactError(
            "MODEL_ARTIFACT_SHA256 must be a 64-character SHA-256 hex digest."
        )

    missing = [
        name
        for name in REQUIRED_MODEL_ARTIFACTS
        if not (destination / name).is_file()
    ]
    if not missing:
        marker_path = destination / VALIDATED_BUNDLE_MARKER
        if not artifact_url or (
            marker_path.is_file()
            and marker_path.read_text(encoding="ascii").strip().lower()
            == artifact_sha256.lower()
        ):
            load_model_metadata(destination)
            return destination

    if not artifact_url or not artifact_sha256:
        raise ModelArtifactError(
            "Trained model files are unavailable. Configure MODEL_ARTIFACT_URL "
            "and MODEL_ARTIFACT_SHA256 in the deployment environment, and set "
            "MODEL_DIR to a persistent volume path."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".mobileshield-models-", dir=destination.parent
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            bundle_path = temporary_path / "model-artifacts.tar.gz"
            staging_dir = temporary_path / "staged-models"

            actual_digest = _download_bundle(
                artifact_url, bundle_path, artifact_token
            )
            if not hmac.compare_digest(
                actual_digest.lower(), artifact_sha256.lower()
            ):
                raise ModelArtifactError(
                    "Model artifact bundle SHA-256 does not match "
                    "MODEL_ARTIFACT_SHA256."
                )

            staging_dir.mkdir()
            _extract_bundle(bundle_path, staging_dir)
            (staging_dir / PENDING_BUNDLE_MARKER).write_text(
                actual_digest, encoding="ascii"
            )

            backup_dir = temporary_path / "previous-models"
            had_existing_directory = destination.exists()
            if had_existing_directory:
                os.replace(destination, backup_dir)
            try:
                os.replace(staging_dir, destination)
            except OSError:
                if had_existing_directory and backup_dir.exists():
                    os.replace(backup_dir, destination)
                raise
    except ModelArtifactError:
        raise
    except OSError as exc:
        raise ModelArtifactError(
            "Could not atomically install the verified model artifact bundle."
        ) from exc

    return destination


def mark_model_artifacts_validated(model_dir: Optional[Path | str] = None) -> None:
    """Cache an external bundle only after FlowPredictor loads and validates it."""
    destination = resolve_model_dir(model_dir)
    pending_marker = destination / PENDING_BUNDLE_MARKER
    if not pending_marker.is_file():
        return

    digest = pending_marker.read_text(encoding="ascii").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ModelArtifactError(
            "The staged model bundle validation marker is invalid."
        )

    marker_temporary = destination / f"{VALIDATED_BUNDLE_MARKER}.tmp"
    try:
        marker_temporary.write_text(digest, encoding="ascii")
        os.replace(marker_temporary, destination / VALIDATED_BUNDLE_MARKER)
        pending_marker.unlink()
    except OSError as exc:
        raise ModelArtifactError(
            "Could not save the validated model bundle marker."
        ) from exc
