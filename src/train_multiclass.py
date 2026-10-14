"""Experiment B: Multiclass Network Malware Traffic Classification.

Classifies flows into 5 canonical categories:
  0: BENIGN
  1: ADWARE
  2: RANSOMWARE
  3: SCAREWARE
  4: SMSMALWARE

Trains and compares:
1. Random Forest
2. Extra Trees
3. LightGBM
4. Simple Baseline (Logistic Regression)

Evaluates on both Group-Aware Test Set and Zero-Day Malware Family Holdout.
Saves best model and artifacts in `models/multiclass/`.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import lightgbm as lgb
import joblib

RANDOM_SEED = 42

CLASS_NAMES = ["BENIGN", "ADWARE", "RANSOMWARE", "SCAREWARE", "SMSMALWARE"]


def load_multiclass_data(npz_path: str = "data/processed/processed_splits.npz") -> Tuple[Any, ...]:
    """Loads preprocessed matrices and multiclass labels, filtering out unmapped ambiguous records."""
    data = np.load(npz_path, allow_pickle=True)
    X_train_raw = data["X_train"]
    y_train_raw = data["y_train_multi"]
    X_val_raw = data["X_val"]
    y_val_raw = data["y_val_multi"]
    X_test_raw = data["X_test"]
    y_test_raw = data["y_test_multi"]
    X_holdout_raw = data["X_holdout"]
    y_holdout_raw = data["y_holdout_multi"]
    feature_names = data["feature_names"].tolist()

    # Filter out None / -1 labels (e.g., generic ambiguous 'MALWARE' strings)
    def clean_split(X, y):
        mask = ~np.isnan(y.astype(float)) & (y != None) & (y != -1)
        # Convert mask to boolean array
        valid_indices = [i for i, val in enumerate(y) if val is not None and str(val) != "nan" and int(val) in range(5)]
        return X[valid_indices], np.array([int(y[i]) for i in valid_indices], dtype=int)

    X_train, y_train = clean_split(X_train_raw, y_train_raw)
    X_val, y_val = clean_split(X_val_raw, y_val_raw)
    X_test, y_test = clean_split(X_test_raw, y_test_raw)
    X_holdout, y_holdout = clean_split(X_holdout_raw, y_holdout_raw)

    return X_train, y_train, X_val, y_val, X_test, y_test, X_holdout, y_holdout, feature_names


def measure_inference_speed(model, X_sample: np.ndarray, num_runs: int = 500) -> Tuple[float, float]:
    """Measures single-sample latency (ms) and batch throughput (flows/sec)."""
    # Batch throughput
    t0 = time.perf_counter()
    _ = model.predict(X_sample)
    batch_time = time.perf_counter() - t0
    throughput = len(X_sample) / batch_time if batch_time > 0 else 0.0

    # Single-sample latency
    test_slice = X_sample[:num_runs]
    t0 = time.perf_counter()
    for i in range(len(test_slice)):
        _ = model.predict(test_slice[i:i+1])
    single_time = time.perf_counter() - t0
    latency_ms = (single_time / len(test_slice)) * 1000.0 if len(test_slice) > 0 else 0.0

    return round(latency_ms, 4), round(throughput, 1)


def evaluate_multiclass_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculates macro, weighted, per-class metrics, and Benign FPR."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    # Benign FPR: benign is class 0
    # FP for benign: non-benign predicted as benign: cm[1:, 0].sum()
    # TN for benign: non-benign predicted as non-benign: cm[1:, 1:].sum()
    # Or False Alarm Rate on Benign: Benign flows (class 0) falsely predicted as malware: cm[0, 1:].sum() / cm[0].sum()
    false_alarms_on_benign = cm[0, 1:].sum()
    total_benign = cm[0].sum()
    false_alarm_rate = false_alarms_on_benign / total_benign if total_benign > 0 else 0.0

    # Per-class metrics
    per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cname] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "support": int(cm[idx].sum())
        }

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "false_alarm_rate_benign": round(float(false_alarm_rate), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }
    return metrics


def train_and_evaluate_multiclass(output_dir: str = "models/multiclass") -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("[*] Loading processed multiclass dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, X_holdout, y_holdout, feature_names = load_multiclass_data()
    print(f"[+] Clean Train shape: {X_train.shape}, Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"[+] Clean Val shape:   {X_val.shape}, Class distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"[+] Clean Test shape:  {X_test.shape}, Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print(f"[+] Clean Holdout shape:{X_holdout.shape}, Class distribution: {dict(zip(*np.unique(y_holdout, return_counts=True)))}")

    models = {
        "Baseline_LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=20, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
    }

    comparison_results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\n[*] Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_duration = round(time.time() - t0, 2)
        fitted_models[name] = model

        # Val evaluation
        val_pred = model.predict(X_val)
        val_metrics = evaluate_multiclass_predictions(y_val, val_pred)

        # Test evaluation
        test_pred = model.predict(X_test)
        test_metrics = evaluate_multiclass_predictions(y_test, test_pred)

        # Holdout zero-day evaluation
        holdout_pred = model.predict(X_holdout)
        holdout_metrics = evaluate_multiclass_predictions(y_holdout, holdout_pred)

        # Latency & throughput
        latency_ms, throughput = measure_inference_speed(model, X_test[:1000])

        comparison_results[name] = {
            "train_duration_sec": train_duration,
            "latency_ms_per_flow": latency_ms,
            "throughput_flows_sec": throughput,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "zero_day_holdout_metrics": holdout_metrics
        }

        print(f"[+] {name} Multiclass Results:")
        print(f"    Val Macro-F1:   {val_metrics['macro_f1']} | Benign False Alarm: {val_metrics['false_alarm_rate_benign']*100:.2f}%")
        print(f"    Test Macro-F1:  {test_metrics['macro_f1']} | Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"    Holdout Macro-F1:{holdout_metrics['macro_f1']}")
        print(f"    Inference Latency: {latency_ms} ms/flow | Throughput: {throughput:,} flows/sec")

    # Select best model based on validation Macro-F1
    best_name = max(
        comparison_results.keys(),
        key=lambda k: comparison_results[k]["val_metrics"]["macro_f1"]
    )
    best_model = fitted_models[best_name]
    print(f"\n[*] Selected Best Multiclass Model: {best_name}")

    # Save artifacts
    best_model_path = out_path / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"[+] Saved best multiclass model binary to {best_model_path}")

    # Save feature importances if tree-based
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
        sorted_idx = np.argsort(fi)[::-1]
        feature_importances = {feature_names[i]: float(fi[i]) for i in sorted_idx}
        with open(out_path / "feature_importances.json", "w", encoding="utf-8") as f:
            json.dump(feature_importances, f, indent=2)

    metadata = {
        "experiment": "Experiment B: Multiclass Classification (BENIGN, ADWARE, RANSOMWARE, SCAREWARE, SMSMALWARE)",
        "best_model_name": best_name,
        "selected_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_count": len(feature_names),
        "classes": {str(i): c for i, c in enumerate(CLASS_NAMES)},
        "test_metrics": comparison_results[best_name]["test_metrics"],
        "zero_day_holdout_metrics": comparison_results[best_name]["zero_day_holdout_metrics"],
        "latency_ms": comparison_results[best_name]["latency_ms_per_flow"],
        "throughput_flows_sec": comparison_results[best_name]["throughput_flows_sec"]
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(out_path / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    with open(out_path / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    return comparison_results


if __name__ == "__main__":
    train_and_evaluate_multiclass()
