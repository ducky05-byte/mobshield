"""Tests for leakage prevention, group split isolation, and identifier exclusion."""

import pytest
import json
from pathlib import Path
from src.prepare_dataset import (
    build_apk_sample_manifest,
    split_samples_group_aware,
    IDENTIFIER_COLS,
    NetworkFlowPreprocessor
)
import joblib
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)


def test_zero_group_overlap_between_splits():
    csv_dir = Path("CSVs") if Path("CSVs").exists() else Path("c:/Users/Varun/Desktop/AML/CSVs")
    manifest = build_apk_sample_manifest(csv_dir)
    splits = split_samples_group_aware(manifest, random_seed=42)

    train_ids = set(s["sample_id"] for s in splits["train"])
    val_ids = set(s["sample_id"] for s in splits["val"])
    test_ids = set(s["sample_id"] for s in splits["test"])
    holdout_ids = set(s["sample_id"] for s in splits["zero_day_holdout"])

    # Verify zero leakage across splits
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val must share 0 APK sample sessions"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test must share 0 APK sample sessions"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test must share 0 APK sample sessions"
    assert len(train_ids.intersection(holdout_ids)) == 0, "Train and Zero-Day Holdout must share 0 APK sessions"


def test_zero_identifier_leakage_in_saved_artifacts():
    prep_path = Path("models/preprocessor.joblib")
    assert prep_path.exists(), "Fitted preprocessor must exist"
    preprocessor: NetworkFlowPreprocessor = joblib.load(prep_path)

    for identifier in IDENTIFIER_COLS:
        assert identifier not in preprocessor.feature_names_out_, (
            f"Leakage hazard: {identifier} is present in final model features!"
        )

    with open("models/feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    for identifier in IDENTIFIER_COLS:
        assert identifier not in feature_names, (
            f"Leakage hazard: {identifier} is present in feature_names.json!"
        )
