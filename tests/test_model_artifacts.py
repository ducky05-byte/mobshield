"""Tests for safe external model bundle retrieval and integrity checking."""

import hashlib
import io
import json
import tarfile
from urllib.error import URLError

import pytest

from src import model_artifacts


FEATURE_NAMES = [f"flow_feature_{index}" for index in range(65)]
LABEL_MAPPINGS = model_artifacts.EXPECTED_LABEL_MAPPINGS
ARTIFACT_CONTENTS = {
    "preprocessor.joblib": b"serialized-preprocessor",
    "binary/best_model.joblib": b"serialized-binary-model",
    "multiclass/best_model.joblib": b"serialized-multiclass-model",
    "feature_names.json": json.dumps(FEATURE_NAMES).encode("utf-8"),
    "label_mapping.json": json.dumps(LABEL_MAPPINGS).encode("utf-8"),
}


def _make_bundle() -> bytes:
    bundle = io.BytesIO()
    with tarfile.open(fileobj=bundle, mode="w:gz") as archive:
        for relative_path, contents in ARTIFACT_CONTENTS.items():
            info = tarfile.TarInfo(f"models/{relative_path}")
            info.size = len(contents)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(contents))
    return bundle.getvalue()


def test_downloads_and_installs_verified_model_bundle(tmp_path, monkeypatch):
    bundle = _make_bundle()
    observed = {}

    def fake_urlopen(request, timeout):
        observed["authorization"] = request.get_header("Authorization")
        observed["timeout"] = timeout
        return io.BytesIO(bundle)

    monkeypatch.setattr(
        model_artifacts.urllib.request,
        "build_opener",
        lambda *_: _FakeOpener(fake_urlopen),
    )
    destination = tmp_path / "persistent" / "models"
    result = model_artifacts.ensure_model_artifacts(
        destination,
        artifact_url="https://storage.example.test/models.tar.gz",
        artifact_sha256=hashlib.sha256(bundle).hexdigest(),
        artifact_token="railway-storage-secret",
    )

    assert result == destination
    assert observed == {
        "authorization": "Bearer railway-storage-secret",
        "timeout": model_artifacts.DOWNLOAD_TIMEOUT_SECONDS,
    }
    for relative_path, contents in ARTIFACT_CONTENTS.items():
        assert (destination / relative_path).read_bytes() == contents
    assert (destination / model_artifacts.PENDING_BUNDLE_MARKER).is_file()
    assert "railway-storage-secret" not in str(destination)


def test_missing_download_configuration_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_ARTIFACT_URL", raising=False)
    monkeypatch.delenv("MODEL_ARTIFACT_SHA256", raising=False)

    with pytest.raises(model_artifacts.ModelArtifactError, match="MODEL_ARTIFACT_URL"):
        model_artifacts.ensure_model_artifacts(tmp_path / "models")


def test_failed_download_does_not_install_partial_artifacts(tmp_path, monkeypatch):
    def fail_download(request, timeout):
        raise URLError("network unavailable")

    monkeypatch.setattr(
        model_artifacts.urllib.request,
        "build_opener",
        lambda *_: _FakeOpener(fail_download),
    )
    destination = tmp_path / "persistent" / "models"

    with pytest.raises(
        model_artifacts.ModelArtifactError, match="download failed or timed out"
    ):
        model_artifacts.ensure_model_artifacts(
            destination,
            artifact_url="https://storage.example.test/models.tar.gz",
            artifact_sha256="a" * 64,
        )

    assert not destination.exists()
    assert not list(destination.parent.glob(".mobileshield-models-*"))


def test_checksum_mismatch_does_not_install_artifacts(tmp_path, monkeypatch):
    bundle = _make_bundle()
    monkeypatch.setattr(
        model_artifacts.urllib.request,
        "build_opener",
        lambda *_: _FakeOpener(lambda request, timeout: io.BytesIO(bundle)),
    )
    destination = tmp_path / "persistent" / "models"

    with pytest.raises(
        model_artifacts.ModelArtifactError, match="SHA-256 does not match"
    ):
        model_artifacts.ensure_model_artifacts(
            destination,
            artifact_url="https://storage.example.test/models.tar.gz",
            artifact_sha256="0" * 64,
        )

    assert not destination.exists()


def test_existing_local_artifacts_are_reused_without_download(tmp_path, monkeypatch):
    destination = tmp_path / "models"
    for relative_path, contents in ARTIFACT_CONTENTS.items():
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)

    def unexpected_download(request, timeout):
        pytest.fail("A complete local model directory should not be downloaded.")

    monkeypatch.setattr(
        model_artifacts.urllib.request,
        "build_opener",
        lambda *_: _FakeOpener(unexpected_download),
    )

    assert model_artifacts.ensure_model_artifacts(destination) == destination


def test_validated_external_bundle_is_reused_without_redownload(
    tmp_path, monkeypatch
):
    bundle = _make_bundle()
    download_count = 0

    def fake_urlopen(request, timeout):
        nonlocal download_count
        download_count += 1
        return io.BytesIO(bundle)

    monkeypatch.setattr(
        model_artifacts.urllib.request,
        "build_opener",
        lambda *_: _FakeOpener(fake_urlopen),
    )
    destination = tmp_path / "persistent" / "models"
    digest = hashlib.sha256(bundle).hexdigest()
    settings = {
        "artifact_url": "https://storage.example.test/models.tar.gz",
        "artifact_sha256": digest,
    }
    model_artifacts.ensure_model_artifacts(destination, **settings)
    model_artifacts.mark_model_artifacts_validated(destination)

    assert (
        (destination / model_artifacts.VALIDATED_BUNDLE_MARKER).read_text(
            encoding="ascii"
        )
        == digest
    )
    assert not (destination / model_artifacts.PENDING_BUNDLE_MARKER).exists()
    assert model_artifacts.ensure_model_artifacts(destination, **settings) == destination
    assert download_count == 1


def test_rejects_archives_with_unexpected_files(tmp_path, monkeypatch):
    bundle = io.BytesIO()
    with tarfile.open(fileobj=bundle, mode="w:gz") as archive:
        for relative_path, contents in ARTIFACT_CONTENTS.items():
            info = tarfile.TarInfo(f"models/{relative_path}")
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
        dataset = b"dataset-not-for-production"
        info = tarfile.TarInfo("models/training-data.csv")
        info.size = len(dataset)
        archive.addfile(info, io.BytesIO(dataset))
    bundle_bytes = bundle.getvalue()
    monkeypatch.setattr(
        model_artifacts.urllib.request,
        "build_opener",
        lambda *_: _FakeOpener(
            lambda request, timeout: io.BytesIO(bundle_bytes)
        ),
    )
    destination = tmp_path / "models"

    with pytest.raises(model_artifacts.ModelArtifactError, match="unexpected file"):
        model_artifacts.ensure_model_artifacts(
            destination,
            artifact_url="https://storage.example.test/models.tar.gz",
            artifact_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        )

    assert not destination.exists()


class _FakeOpener:
    def __init__(self, open_callback):
        self.open_callback = open_callback

    def open(self, request, timeout):
        return self.open_callback(request, timeout)
