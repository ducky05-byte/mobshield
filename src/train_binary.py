"""Experiment A: Binary Network Malware Traffic Classification.

Trains and compares:
1. Random Forest
2. Extra Trees
3. LightGBM
4. Simple Baseline (Logistic Regression / Dummy)

Evaluates on both Group-Aware Test Set and Zero-Day Malware Family Holdout.
Saves best model and artifacts in `models/binary/`.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, classification_report
)
import lightgbm as lgb
import joblib

RANDOM_SEED = 42


def load_binary_data(npz_path: str = "data/processed/processed_splits.npz") -> Tuple[Any, ...]:
    """Loads preprocessed matrices and labels."""
    data = np.load(npz_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train_bin"].astype(int)
    X_val = data["X_val"]
    y_val = data["y_val_bin"].astype(int)
    X_test = data["X_test"]
    y_test = data["y_test_bin"].astype(int)
    X_holdout = data["X_holdout"]
    y_holdout = data["y_holdout_bin"].astype(int)
    feature_names = data["feature_names"].tolist()

    return X_train, y_train, X_val, y_val, X_test, y_test, X_holdout, y_holdout, feature_names


def measure_inference_speed(model, X_sample: np.ndarray, num_runs: int = 1000) -> Tuple[float, float]:
    """Measures single-sample and batch inference latency and throughput."""
    # Warm up
    _ = model.predict(X_sample[:10])
    
    # Batch throughput
    t0 = time.perf_counter()
    _ = model.predict(X_sample)
    batch_time = time.perf_counter() - t0
    throughput = len(X_sample) / batch_time if batch_time > 0 else 0.0

    # Single-sample latency (average of 1,000 single predictions)
    test_slice = X_sample[:num_runs]
    t0 = time.perf_counter()
    for i in range(len(test_slice)):
        _ = model.predict(test_slice[i:i+1])
    single_time = time.perf_counter() - t0
    latency_ms = (single_time / len(test_slice)) * 1000.0 if len(test_slice) > 0 else 0.0

    return round(latency_ms, 4), round(throughput, 1)


def evaluate_binary_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """Calculates full metrics suite including FPR on benign traffic."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    fpr_benign = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_malware = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "false_positive_rate_benign": round(float(fpr_benign), 4),
        "false_negative_rate_malware": round(float(fnr_malware), 4),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def train_and_evaluate_binary(output_dir: str = "models/binary") -> Dict[str, Any]:
    """Orchestrates model training, model selection, holdout evaluation, and artifact saving."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("[*] Loading processed binary dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, X_holdout, y_holdout, feature_names = load_binary_data()
    print(f"[+] Train shape: {X_train.shape}, Malicious share: {np.mean(y_train)*100:.1f}%")
    print(f"[+] Val shape:   {X_val.shape}, Malicious share: {np.mean(y_val)*100:.1f}%")
    print(f"[+] Test shape:  {X_test.shape}, Malicious share: {np.mean(y_test)*100:.1f}%")
    print(f"[+] Holdout shape:{X_holdout.shape}, Malicious share: {np.mean(y_holdout)*100:.1f}%")

    # Define candidate models with balanced class weights
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
        val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        val_metrics = evaluate_binary_predictions(y_val, val_pred, val_prob)

        # Test evaluation
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        test_metrics = evaluate_binary_predictions(y_test, test_pred, test_prob)

        # Holdout zero-day evaluation
        holdout_pred = model.predict(X_holdout)
        holdout_prob = model.predict_proba(X_holdout)[:, 1] if hasattr(model, "predict_proba") else None
        holdout_metrics = evaluate_binary_predictions(y_holdout, holdout_pred, holdout_prob)

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

        print(f"[+] {name} Results:")
        print(f"    Val F1:     {val_metrics['f1_score']} | FPR (Benign): {val_metrics['false_positive_rate_benign']*100:.2f}%")
        print(f"    Test F1:    {test_metrics['f1_score']} | FPR (Benign): {test_metrics['false_positive_rate_benign']*100:.2f}%")
        print(f"    Holdout F1: {holdout_metrics['f1_score']} | Zero-Day Recall: {holdout_metrics['recall']*100:.2f}%")
        print(f"    Inference Latency: {latency_ms} ms/flow | Throughput: {throughput:,} flows/sec")

    # Select best model based on Validation Macro-F1 and low FPR
    best_name = max(
        comparison_results.keys(),
        key=lambda k: comparison_results[k]["val_metrics"]["macro_f1"] - (comparison_results[k]["val_metrics"]["false_positive_rate_benign"] * 0.5)
    )
    best_model = fitted_models[best_name]
    print(f"\n[*] Selected Best Binary Model: {best_name}")

    # Save artifacts
    best_model_path = out_path / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"[+] Saved best model binary to {best_model_path}")

    # Save feature importances if tree-based
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
        sorted_idx = np.argsort(fi)[::-1]
        feature_importances = {feature_names[i]: float(fi[i]) for i in sorted_idx}
        with open(out_path / "feature_importances.json", "w", encoding="utf-8") as f:
            json.dump(feature_importances, f, indent=2)

    # Save configuration & metadata
    metadata = {
        "experiment": "Experiment A: Binary Classification (BENIGN vs MALWARE_ASSOCIATED)",
        "best_model_name": best_name,
        "selected_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_count": len(feature_names),
        "classes": {"0": "BENIGN", "1": "MALWARE_ASSOCIATED"},
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
    train_and_evaluate_binary()
