"""Unit tests for SVM model artifact loading, feature contract, and threshold behavior."""

import json
from pathlib import Path
import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.predict_flow import FlowPredictor
from src.prepare_dataset import NetworkFlowPreprocessor
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)


SVM_MODEL_PATH = Path("models/svm/best_model.joblib")
SVM_FEATURES_PATH = Path("models/svm/feature_names.json")
DATA_SPLITS_PATH = Path("data/processed/processed_splits.npz")


def test_svm_artifact_loading():
    """Verify serialized SVM pipeline loads and has expected stages."""
    assert SVM_MODEL_PATH.exists(), f"SVM model artifact missing at {SVM_MODEL_PATH}"
    pipeline = joblib.load(SVM_MODEL_PATH)
    assert isinstance(pipeline, Pipeline), "SVM model must be an sklearn Pipeline"
    
    # Must have scaler and nystroem/svm steps
    step_names = [name for name, _ in pipeline.steps]
    assert "scaler" in step_names, "Pipeline must include StandardScaler"
    assert "nystroem" in step_names, "Pipeline must include Nystroem RBF approximation"
    assert "svm" in step_names, "Pipeline must include SVM classifier step"


def test_svm_feature_contract():
    """Verify SVM pipeline expects exactly 65 features matching dataset order."""
    assert SVM_FEATURES_PATH.exists(), f"SVM feature names file missing at {SVM_FEATURES_PATH}"
    with open(SVM_FEATURES_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)
    assert len(features) == 65, f"Expected 65 features, got {len(features)}"

    # Load test split and verify shape matches
    data = np.load(DATA_SPLITS_PATH, allow_pickle=True)
    X_test = data["X_test"]
    assert X_test.shape[1] == 65

    pipeline = joblib.load(SVM_MODEL_PATH)
    preds = pipeline.predict(X_test[:5])
    assert len(preds) == 5
    assert all(p in (0, 1) for p in preds)


def test_svm_threshold_and_probabilities():
    """Verify probability output or decision scores adhere to numeric bounds."""
    pipeline = joblib.load(SVM_MODEL_PATH)
    data = np.load(DATA_SPLITS_PATH, allow_pickle=True)
    X_sample = data["X_test"][:20]

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_sample)
        assert probs.shape == (20, 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert np.allclose(np.sum(probs, axis=1), 1.0)

        # Applying threshold 0.65
        malicious_probs = probs[:, 1]
        threshold_preds = (malicious_probs >= 0.65).astype(int)
        assert all(p in (0, 1) for p in threshold_preds)
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_sample)
        assert len(scores) == 20
        assert not np.any(np.isnan(scores))


def test_benign_flows_not_all_flagged_malicious():
    """Verify that known benign flows are not all flagged as malicious."""
    pipeline = joblib.load(SVM_MODEL_PATH)
    data = np.load(DATA_SPLITS_PATH, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test_bin"]

    benign_indices = np.where(y_test == 0)[0][:100]
    benign_samples = X_test[benign_indices]

    preds = pipeline.predict(benign_samples)
    benign_count = np.sum(preds == 0)
    # At least some benign samples must be classified as benign
    assert benign_count > 0, "Model must not classify 100% of benign flows as malicious"


def test_backend_interface_compatibility():
    """Verify FlowPredictor continues to function properly with existing API contract."""
    predictor = FlowPredictor()
    assert predictor.binary_model is not None
    assert predictor.multiclass_model is not None

    data = np.load(DATA_SPLITS_PATH, allow_pickle=True)
    feat_names = data["feature_names"].tolist()
    sample_vec = data["X_test"][0]
    sample_dict = {feat_names[i]: float(sample_vec[i]) for i in range(len(feat_names))}

    res = predictor.predict_flow(sample_dict, mode="binary")
    assert "prediction" in res
    assert "label" in res
    assert "confidence" in res
    assert "probabilities" in res
    assert res["label"] in ("BENIGN", "MALWARE_ASSOCIATED")


def test_backend_and_companion_imports():
    """Verify critical server imports and modules compile without syntax errors."""
    import src.companion_api
    import src.flow_engine
    import src.predict_flow
    assert hasattr(src.companion_api, "app")
    assert hasattr(src.flow_engine, "LiveCapturePipeline")
    assert hasattr(src.flow_engine, "BidirectionalFlow")
