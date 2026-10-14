"""Unit tests for model loading, prediction shapes, probability normalization, and determinism."""

import pytest
import numpy as np
import json
from pathlib import Path
from src.prepare_dataset import NetworkFlowPreprocessor
from src.predict_flow import FlowPredictor
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)


@pytest.fixture
def predictor():
    return FlowPredictor()


def test_model_loading(predictor):
    assert predictor.binary_model is not None, "Binary model artifact must load successfully"
    assert predictor.multiclass_model is not None, "Multiclass model artifact must load successfully"
    assert predictor.preprocessor is not None, "Preprocessor artifact must load successfully"
    assert predictor.feature_names == predictor.preprocessor.feature_names_out_


def test_binary_prediction(predictor):
    data = np.load("data/processed/processed_splits.npz", allow_pickle=True)
    sample_vec = data["X_test"][0]
    feat_names = data["feature_names"].tolist()
    sample_dict = {feat_names[i]: float(sample_vec[i]) for i in range(len(feat_names))}

    res = predictor.predict_flow(sample_dict, mode="binary")
    assert res["prediction"] in (0, 1)
    assert res["label"] in ("BENIGN", "MALWARE_ASSOCIATED")
    assert 0.0 <= res["confidence"] <= 1.0
    assert "BENIGN" in res["probabilities"]
    assert "MALWARE_ASSOCIATED" in res["probabilities"]
    
    # Probabilities sum to 1.0
    prob_sum = sum(res["probabilities"].values())
    assert pytest.approx(prob_sum, 0.01) == 1.0


def test_multiclass_prediction(predictor):
    data = np.load("data/processed/processed_splits.npz", allow_pickle=True)
    sample_vec = data["X_test"][0]
    feat_names = data["feature_names"].tolist()
    sample_dict = {feat_names[i]: float(sample_vec[i]) for i in range(len(feat_names))}

    res = predictor.predict_flow(sample_dict, mode="multiclass")
    assert res["prediction"] in range(5)
    assert res["label"] in ["BENIGN", "ADWARE", "RANSOMWARE", "SCAREWARE", "SMSMALWARE"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 5

    prob_sum = sum(res["probabilities"].values())
    assert pytest.approx(prob_sum, 0.01) == 1.0


def test_prediction_determinism(predictor):
    data = np.load("data/processed/processed_splits.npz", allow_pickle=True)
    sample_vec = data["X_test"][1]
    feat_names = data["feature_names"].tolist()
    sample_dict = {feat_names[i]: float(sample_vec[i]) for i in range(len(feat_names))}

    res1 = predictor.predict_flow(sample_dict, mode="binary")
    res2 = predictor.predict_flow(sample_dict, mode="binary")
    assert res1["prediction"] == res2["prediction"]
    assert res1["confidence"] == res2["confidence"]
