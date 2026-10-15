"""Real-Time Flow Prediction Engine.

Loads saved preprocessing pipeline and trained models from `models/`, accepts
unseen flow dictionaries or DataFrames, applies identical transformations,
and outputs classification predictions and confidence scores.
"""

import sys
import os
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from src.prepare_dataset import NetworkFlowPreprocessor
except ImportError:
    from prepare_dataset import NetworkFlowPreprocessor

from src.model_artifacts import load_model_metadata, resolve_model_dir

# Ensure joblib unpickling compatibility across modules
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)


class FlowPredictor:
    """Production inference engine for live mobile network flow classification."""

    def __init__(
        self,
        preprocessor_path: Optional[Union[str, Path]] = None,
        binary_model_path: Optional[Union[str, Path]] = None,
        multiclass_model_path: Optional[Union[str, Path]] = None,
        label_mapping_path: Optional[Union[str, Path]] = None,
        model_dir: Optional[Union[str, Path]] = None,
    ):
        model_dir = resolve_model_dir(model_dir)
        self.preprocessor_path = Path(
            preprocessor_path or model_dir / "preprocessor.joblib"
        )
        self.binary_model_path = Path(
            binary_model_path or model_dir / "binary" / "best_model.joblib"
        )
        self.multiclass_model_path = Path(
            multiclass_model_path or model_dir / "multiclass" / "best_model.joblib"
        )
        self.label_mapping_path = Path(
            label_mapping_path or model_dir / "label_mapping.json"
        )

        feature_names, mappings = load_model_metadata(model_dir)
        self.feature_names = feature_names
        try:
            self.preprocessor: NetworkFlowPreprocessor = joblib.load(
                self.preprocessor_path
            )
            self.binary_model = joblib.load(self.binary_model_path)
            self.multiclass_model = joblib.load(self.multiclass_model_path)
        except (
            OSError,
            EOFError,
            ValueError,
            ImportError,
            AttributeError,
            pickle.UnpicklingError,
        ) as exc:
            artifact_path = getattr(exc, "filename", None)
            detail = (
                f" ({artifact_path})"
                if artifact_path and Path(artifact_path).name
                else ""
            )
            raise RuntimeError(
                "Failed to load a required trained model artifact"
                f"{detail}; verify the bundle and compatible Python package versions."
            ) from exc

        preprocessor_features = getattr(
            self.preprocessor, "feature_names_out_", None
        )
        if (
            not getattr(self.preprocessor, "is_fitted", False)
            or (
                list(preprocessor_features)
                if preprocessor_features is not None
                else []
            )
            != feature_names
        ):
            raise RuntimeError(
                "The fitted preprocessor feature order does not match "
                "models/feature_names.json."
            )

        self.binary_classes = {v: k for k, v in mappings["binary"].items()}
        self.multiclass_classes = {
            v: k for k, v in mappings["multiclass"].items()
        }
        self._validate_model(self.binary_model, self.binary_classes, "binary")
        self._validate_model(
            self.multiclass_model, self.multiclass_classes, "multiclass"
        )

    @staticmethod
    def _validate_model(model: Any, classes: Dict[int, str], name: str) -> None:
        if not callable(getattr(model, "predict", None)):
            raise RuntimeError(f"The {name} model artifact has no predict method.")
        if not callable(getattr(model, "predict_proba", None)):
            raise RuntimeError(
                f"The {name} model artifact has no predict_proba method."
            )

        feature_count = getattr(model, "n_features_in_", None)
        if feature_count is not None and feature_count != 65:
            raise RuntimeError(
                f"The {name} model expects {feature_count} features; "
                "the production flow pipeline supplies 65."
            )

        model_classes = getattr(model, "classes_", None)
        if model_classes is not None and {
            int(class_id) for class_id in model_classes
        } != set(classes):
            raise RuntimeError(
                f"The {name} model classes do not match models/label_mapping.json."
            )

    def predict_flow(self, flow: Union[Dict[str, Any], pd.DataFrame], mode: str = "binary") -> Dict[str, Any]:
        """Classifies a network flow record.
        
        Args:
            flow: A dictionary of flow features or a pandas DataFrame.
            mode: "binary" or "multiclass".
            
        Returns:
            Dict containing predicted class index, label name, and class probability scores.
        """
        if isinstance(flow, dict):
            df = pd.DataFrame([flow])
        else:
            df = flow.copy()

        # Transform using fitted preprocessor
        X_scaled = self.preprocessor.transform(df)

        if mode == "binary":
            if self.binary_model is None:
                raise RuntimeError("Binary model artifact not loaded.")
            pred = self.binary_model.predict(X_scaled)[0]
            probs = self.binary_model.predict_proba(X_scaled)[0] if hasattr(self.binary_model, "predict_proba") else None
            
            label_name = self.binary_classes.get(int(pred), f"Class_{pred}")
            probabilities = {}
            if probs is not None:
                class_ids = getattr(self.binary_model, "classes_", None)
                if class_ids is None:
                    class_ids = range(len(probs))
                if len(class_ids) != len(probs):
                    raise RuntimeError(
                        "Binary model class order does not match its probability output."
                    )
                probabilities = {
                    self.binary_classes.get(int(class_id), f"Class_{class_id}"): float(probability)
                    for class_id, probability in zip(class_ids, probs)
                }
            conf = probabilities.get(label_name, 1.0)

            return {
                "mode": "binary",
                "prediction": int(pred),
                "label": label_name,
                "confidence": round(conf, 4),
                "probabilities": {label: round(probability, 4) for label, probability in probabilities.items()}
            }

        elif mode == "multiclass":
            if self.multiclass_model is None:
                raise RuntimeError("Multiclass model artifact not loaded.")
            pred = self.multiclass_model.predict(X_scaled)[0]
            probs = self.multiclass_model.predict_proba(X_scaled)[0] if hasattr(self.multiclass_model, "predict_proba") else None

            label_name = self.multiclass_classes.get(int(pred), f"Class_{pred}")
            probabilities = {}
            if probs is not None:
                class_ids = getattr(self.multiclass_model, "classes_", None)
                if class_ids is None:
                    class_ids = range(len(probs))
                if len(class_ids) != len(probs):
                    raise RuntimeError(
                        "Multiclass model class order does not match its probability output."
                    )
                probabilities = {
                    self.multiclass_classes.get(int(class_id), f"Class_{class_id}"): float(probability)
                    for class_id, probability in zip(class_ids, probs)
                }
            conf = probabilities.get(label_name, 1.0)

            return {
                "mode": "multiclass",
                "prediction": int(pred),
                "label": label_name,
                "confidence": round(conf, 4),
                "probabilities": {label: round(probability, 4) for label, probability in probabilities.items()}
            }
        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'binary' or 'multiclass'.")


def run_demo():
    """Runs a quick live verification test using a sample flow from test split."""
    print("[*] Initializing FlowPredictor...")
    predictor = FlowPredictor()

    data = np.load("data/processed/processed_splits.npz", allow_pickle=True)
    feat_names = data["feature_names"].tolist()
    sample_vec = data["X_test"][0]

    # Reconstruct dictionary
    sample_dict = {feat_names[i]: float(sample_vec[i]) for i in range(len(feat_names))}

    print("\n[*] Running Binary Inference on Test Sample:")
    bin_res = predictor.predict_flow(sample_dict, mode="binary")
    print(f"    Label:       {bin_res['label']}")
    print(f"    Confidence:  {bin_res['confidence']*100:.2f}%")
    print(f"    Probabilities: {bin_res['probabilities']}")

    print("\n[*] Running Multiclass Inference on Test Sample:")
    multi_res = predictor.predict_flow(sample_dict, mode="multiclass")
    print(f"    Label:       {multi_res['label']}")
    print(f"    Confidence:  {multi_res['confidence']*100:.2f}%")
    print(f"    Probabilities: {multi_res['probabilities']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Flow Prediction Engine")
    parser.add_argument("--demo", action="store_true", help="Run demonstration inference test")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        run_demo()
