"""Unit tests for preprocessing, identifier removal, duplicate columns, and label mapping."""

import pytest
import numpy as np
import pandas as pd
from src.prepare_dataset import (
    NetworkFlowPreprocessor,
    map_binary_label,
    map_multiclass_label,
    IDENTIFIER_COLS
)


def test_duplicate_column_handling():
    preprocessor = NetworkFlowPreprocessor()
    raw_cols = ["Flow ID", "Source IP", "Fwd Header Length", "Protocol", "Fwd Header Length"]
    cleaned_cols = preprocessor.sanitize_column_names(raw_cols)
    assert cleaned_cols == ["Flow ID", "Source IP", "Fwd Header Length", "Protocol", "Fwd Header Length.1"]
    assert len(cleaned_cols) == len(set(cleaned_cols)), "Duplicate column names must be unique after sanitation"


def test_identifier_removal():
    preprocessor = NetworkFlowPreprocessor()
    df = pd.DataFrame({
        "Flow ID": ["10.42.0.1-10.42.0.2-80-80-6"],
        "Source IP": ["10.42.0.1"],
        "Source Port": [80],
        "Destination IP": ["10.42.0.2"],
        "Destination Port": [80],
        "Timestamp": ["14/06/2017 04:22:52"],
        "Protocol": [6],
        "Flow Duration": [1000],
        "Total Fwd Packets": [10],
        "Label": ["BENIGN"]
    })
    preprocessor.fit(df)
    
    for identifier in IDENTIFIER_COLS:
        assert identifier not in preprocessor.feature_names_out_, f"{identifier} must be removed from model features"
    assert "Label" not in preprocessor.feature_names_out_, "Target label must never be an input feature"


def test_missing_and_infinite_values():
    preprocessor = NetworkFlowPreprocessor()
    df_train = pd.DataFrame({
        "Protocol": [6, 17, 6],
        "Flow Duration": [100.0, 200.0, 300.0],
        "Flow Bytes/s": [10.0, 20.0, 30.0]
    })
    preprocessor.fit(df_train)

    df_test = pd.DataFrame({
        "Protocol": [6, 17],
        "Flow Duration": [np.nan, 250.0],
        "Flow Bytes/s": [np.inf, -np.inf]
    })
    X_test = preprocessor.transform(df_test)
    assert not np.isnan(X_test).any(), "Transformed matrix must contain no NaN values"
    assert not np.isinf(X_test).any(), "Transformed matrix must contain no Infinite values"


def test_label_mappings():
    # Binary mappings
    assert map_binary_label("BENIGN") == 0
    assert map_binary_label("ADWARE_GOOLIGAN") == 1
    assert map_binary_label("RANSOMWARE_SVPENG") == 1
    assert map_binary_label("SCAREWARE_ANDROIDDEFENDER") == 1
    assert map_binary_label("SMSMALWARE_NANDROBOX") == 1
    assert map_binary_label("") is None
    assert map_binary_label("[BLANK]") is None

    # Multiclass mappings
    assert map_multiclass_label("BENIGN") == 0
    assert map_multiclass_label("ADWARE_FEIWO") == 1
    assert map_multiclass_label("RANSOMWARE_CHARGER") == 2
    assert map_multiclass_label("SCAREWARE_FAKEAV") == 3
    assert map_multiclass_label("SMSMALWARE_FAKEINST") == 4
    # Ambiguous generic label should be safely excluded (None)
    assert map_multiclass_label("MALWARE") is None
    assert map_multiclass_label("") is None
