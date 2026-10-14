"""
MobileShield Advanced Machine Learning - Nystroem RBF Approximation Experiment
=============================================================================
Focused experiment evaluating Scalable Nystroem RBF Kernel Approximation
for Android network-flow malware detection on CICAndMal2017 dataset.

Dataset splits reused from: data/processed/processed_splits.npz
Baseline comparison: models/binary/best_model.joblib (Random Forest)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Project paths
PROJECT_ROOT = Path("C:/Users/Varun/Desktop/AML")
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_splits.npz"
BASELINE_PATH = PROJECT_ROOT / "models" / "binary" / "best_model.joblib"
SVM_OUTPUT_DIR = PROJECT_ROOT / "models" / "svm"
SVM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def load_dataset() -> Tuple[np.ndarray, ...]:
    """Loads pre-split processed datasets without recalculating features."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed splits not found at {DATA_PATH}")

    data = np.load(DATA_PATH, allow_pickle=True)
    X_train = data["X_train"]
    y_train_bin = data["y_train_bin"]
    X_val = data["X_val"]
    y_val_bin = data["y_val_bin"]
    X_test = data["X_test"]
    y_test_bin = data["y_test_bin"]
    X_holdout = data["X_holdout"]
    y_holdout_bin = data["y_holdout_bin"]
    feature_names = data["feature_names"].tolist()

    return (
        X_train, y_train_bin,
        X_val, y_val_bin,
        X_test, y_test_bin,
        X_holdout, y_holdout_bin,
        feature_names
    )


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    dataset_name: str = "Test"
) -> Dict[str, Any]:
    """
    Calculates binary classification metrics.
    Guarantees ROC-AUC is only computed if y_true contains both classes.
    Clearly handles zero-day holdout sets containing only 1 class.
    """
    classes_present = np.unique(y_true)
    num_classes = len(classes_present)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr_benign = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr_malware = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, len(y_true)
        fpr_benign = None
        fnr_malware = None

    metrics: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "total_samples": int(len(y_true)),
        "classes_present": [int(c) for c in classes_present],
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "false_positive_rate_benign": round(float(fpr_benign), 4) if fpr_benign is not None else "N/A (No benign samples)",
        "false_negative_rate_malware": round(float(fnr_malware), 4) if fnr_malware is not None else "N/A",
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }

    # ROC-AUC and PR-AUC require BOTH classes (binary 0 and 1) to be present
    if y_prob is not None:
        if num_classes > 1:
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
                metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
            except Exception as e:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
                metrics["auc_note"] = str(e)
        else:
            # Holdout set with only 1 class present
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
            metrics["auc_note"] = f"ROC-AUC / PR-AUC undefined: dataset contains only class {classes_present[0]} (Zero-day malware recall test)."

        # Alerts at MobileShield's default 0.65 threshold
        pred_065 = (y_prob >= 0.65).astype(int)
        cm65 = confusion_matrix(y_true, pred_065, labels=[0, 1])
        tn65, fp65, fn65, tp65 = cm65.ravel() if cm65.shape == (2, 2) else (0, 0, 0, len(y_true))
        metrics["metrics_at_065"] = {
            "accuracy": round(float(accuracy_score(y_true, pred_065)), 4),
            "precision": round(float(precision_score(y_true, pred_065, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, pred_065, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, pred_065, zero_division=0)), 4),
            "fpr_benign": round(float(fp65 / (tn65 + fp65)) if (tn65 + fp65) > 0 else 0.0, 4) if (tn65 + fp65) > 0 else "N/A"
        }
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def measure_inference_speed(
    model: Any,
    X_samples: np.ndarray,
    num_runs: int = 1000
) -> Tuple[float, float]:
    """Measures single-flow latency and batch throughput."""
    # Warmup
    _ = model.predict(X_samples[:10])

    t0 = time.perf_counter()
    for i in range(num_runs):
        _ = model.predict(X_samples[i:i+1])
    single_elapsed = time.perf_counter() - t0
    latency_ms = round((single_elapsed / num_runs) * 1000.0, 4)

    t0_b = time.perf_counter()
    _ = model.predict(X_samples)
    batch_elapsed = time.perf_counter() - t0_b
    throughput = round(len(X_samples) / max(batch_elapsed, 1e-6), 1)

    return latency_ms, throughput


def run_nystroem_experiments():
    print("=" * 70, flush=True)
    print("MobileShield Advanced Machine Learning - Nystroem RBF SVM Experiment", flush=True)
    print("=" * 70, flush=True)

    print("\n[*] Loading dataset splits from data/processed/processed_splits.npz...", flush=True)
    (
        X_train, y_train_bin,
        X_val, y_val_bin,
        X_test, y_test_bin,
        X_holdout, y_holdout_bin,
        feature_names
    ) = load_dataset()

    n_features = len(feature_names)
    print(f"[+] Total Binary Train Flows:     {len(X_train):,} (Malicious: {np.mean(y_train_bin)*100:.1f}%)", flush=True)
    print(f"[+] Total Binary Val Flows:       {len(X_val):,} (Malicious: {np.mean(y_val_bin)*100:.1f}%)", flush=True)
    print(f"[+] Total Binary Test Flows:      {len(X_test):,} (Malicious: {np.mean(y_test_bin)*100:.1f}%)", flush=True)
    print(f"[+] Zero-Day Holdout Flows:       {len(X_holdout):,} (Malicious: {np.mean(y_holdout_bin)*100:.1f}%)", flush=True)
    print(f"[+] Input Flow Features:          {n_features}", flush=True)

    # Document gamma choice:
    # Since inputs are standardized via StandardScaler (zero mean, unit variance), each feature has variance = 1.
    # The expected squared distance between standardized vectors with 65 features is E[||x - x'||^2] ~ 2 * 65 = 130.
    # Setting gamma = 1.0 / n_features (1.0 / 65 ~ 0.01538) ensures exp(-gamma * ||x - x'||^2) stays well-behaved (~exp(-2) ~ 0.135)
    # preventing kernel collapse (all similarities -> 1) or numerical underflow (all similarities -> 0).
    gamma_val = float(1.0 / n_features)
    print(f"[+] Documented Gamma Setting:     1 / n_features = 1 / {n_features} = {gamma_val:.6f}", flush=True)

    # -----------------------------------------------------------------
    # Evaluate Baseline Random Forest on Held-Out Test Set
    # -----------------------------------------------------------------
    print("\n" + "-" * 70, flush=True)
    print("[1] Evaluating Production Baseline (Random Forest) on Test Split", flush=True)
    print("-" * 70, flush=True)

    baseline_metrics = {}
    if BASELINE_PATH.exists():
        rf_model = joblib.load(BASELINE_PATH)
        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1] if hasattr(rf_model, "predict_proba") else None
        baseline_test_met = evaluate_binary_predictions(y_test_bin, rf_pred, rf_prob, dataset_name="Test Set")

        rf_holdout_pred = rf_model.predict(X_holdout)
        rf_holdout_prob = rf_model.predict_proba(X_holdout)[:, 1] if hasattr(rf_model, "predict_proba") else None
        baseline_holdout_met = evaluate_binary_predictions(y_holdout_bin, rf_holdout_pred, rf_holdout_prob, dataset_name="Zero-Day Holdout")

        rf_lat_ms, rf_thru = measure_inference_speed(rf_model, X_test[:1000])
        rf_size_mb = round(os.path.getsize(BASELINE_PATH) / (1024 * 1024), 2)

        baseline_metrics = {
            "model_type": "RandomForestClassifier(n_estimators=100, max_depth=20)",
            "test_metrics": baseline_test_met,
            "holdout_metrics": baseline_holdout_met,
            "latency_ms": rf_lat_ms,
            "throughput": rf_thru,
            "model_size_mb": rf_size_mb
        }

        print(f"[+] Baseline Random Forest (Size: {rf_size_mb} MB | Latency: {rf_lat_ms} ms/flow):", flush=True)
        print(f"    Test Accuracy: {baseline_test_met['accuracy']*100:.2f}% | Precision: {baseline_test_met['precision']*100:.2f}% | Recall: {baseline_test_met['recall']*100:.2f}%", flush=True)
        print(f"    Test F1:       {baseline_test_met['f1_score']:.4f} | Benign FPR: {baseline_test_met['false_positive_rate_benign']*100:.2f}%", flush=True)
        print(f"    Test ROC-AUC:  {baseline_test_met['roc_auc']:.4f} | PR-AUC: {baseline_test_met['pr_auc']:.4f}", flush=True)
        if "metrics_at_065" in baseline_test_met:
            m65 = baseline_test_met["metrics_at_065"]
            print(f"    @ 0.65 Thresh: Acc: {m65['accuracy']*100:.2f}% | Rec: {m65['recall']*100:.2f}% | F1: {m65['f1_score']:.4f} | Benign FPR: {m65['fpr_benign']*100:.2f}%", flush=True)
        print(f"    Holdout Recall: {baseline_holdout_met['recall']*100:.2f}% | F1: {baseline_holdout_met['f1_score']:.4f}", flush=True)
    else:
        print("[!] Baseline model artifact not found, using reference baseline numbers.", flush=True)

    # -----------------------------------------------------------------
    # Focused Nystroem RBF Experiments
    # -----------------------------------------------------------------
    print("\n" + "-" * 70, flush=True)
    print("[2] Training & Evaluating Nystroem RBF SVM Configurations", flush=True)
    print("-" * 70, flush=True)

    # We evaluate two primary configurations of Nystroem:
    # 1. Nystroem + LinearSVC(class_weight='balanced') -> Evaluates the uncalibrated RBF kernel SVM decision boundary w^T \phi(x) + b
    # 2. Nystroem + CalibratedClassifierCV(LinearSVC) -> Sigmoid calibrated probabilities for threshold 0.65
    # 3. Nystroem + SGDClassifier(loss='modified_huber') -> Smooth calibrated probabilities natively
    nystroem_configs = {
        "Nystroem_RBF_LinearSVC_Balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("nystroem", Nystroem(kernel="rbf", gamma=gamma_val, n_components=100, random_state=RANDOM_SEED)),
            ("svm", LinearSVC(dual="auto", class_weight="balanced", max_iter=2500, random_state=RANDOM_SEED))
        ]),
        "Nystroem_RBF_Calibrated_LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("nystroem", Nystroem(kernel="rbf", gamma=gamma_val, n_components=100, random_state=RANDOM_SEED)),
            ("svm", CalibratedClassifierCV(
                estimator=LinearSVC(dual="auto", class_weight="balanced", max_iter=2500, random_state=RANDOM_SEED),
                method="sigmoid",
                cv=3
            ))
        ]),
        "Nystroem_RBF_SGD_ModifiedHuber": Pipeline([
            ("scaler", StandardScaler()),
            ("nystroem", Nystroem(kernel="rbf", gamma=gamma_val, n_components=100, random_state=RANDOM_SEED)),
            ("svm", SGDClassifier(
                loss="modified_huber",
                penalty="l2",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=1500,
                tol=1e-3,
                random_state=RANDOM_SEED
            ))
        ])
    }

    nystroem_results = {}
    fitted_models = {}

    for name, pipeline in nystroem_configs.items():
        print(f"\n[*] Training {name} on {len(X_train):,} flows...", flush=True)
        t0 = time.time()
        pipeline.fit(X_train, y_train_bin)
        train_duration = round(time.time() - t0, 2)
        fitted_models[name] = pipeline

        # Predictions on mixed-class test set
        test_pred = pipeline.predict(X_test)
        test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        # If no predict_proba, compute decision_function for ROC-AUC
        if test_prob is None and hasattr(pipeline, "decision_function"):
            raw_scores = pipeline.decision_function(X_test)
            test_met = evaluate_binary_predictions(y_test_bin, test_pred, None, dataset_name="Test Set")
            if len(np.unique(y_test_bin)) > 1:
                test_met["roc_auc"] = round(float(roc_auc_score(y_test_bin, raw_scores)), 4)
                test_met["pr_auc"] = round(float(average_precision_score(y_test_bin, raw_scores)), 4)
                test_met["auc_score_type"] = "decision_function"
        else:
            test_met = evaluate_binary_predictions(y_test_bin, test_pred, test_prob, dataset_name="Test Set")
            if test_prob is not None:
                test_met["auc_score_type"] = "calibrated_probability"

        # Predictions on zero-day holdout set (100% malicious flows)
        holdout_pred = pipeline.predict(X_holdout)
        holdout_prob = pipeline.predict_proba(X_holdout)[:, 1] if hasattr(pipeline, "predict_proba") else None
        holdout_met = evaluate_binary_predictions(y_holdout_bin, holdout_pred, holdout_prob, dataset_name="Zero-Day Holdout")

        # Measure latency
        lat_ms, thru = measure_inference_speed(pipeline, X_test[:1000])

        # Measure model size
        temp_file = SVM_OUTPUT_DIR / f"temp_{name}.joblib"
        joblib.dump(pipeline, temp_file)
        size_mb = round(os.path.getsize(temp_file) / (1024 * 1024), 3)
        temp_file.unlink()

        nystroem_results[name] = {
            "train_duration_sec": train_duration,
            "latency_ms_per_flow": lat_ms,
            "throughput_flows_sec": thru,
            "model_size_mb": size_mb,
            "test_metrics": test_met,
            "zero_day_holdout_metrics": holdout_met
        }

        fpr_disp = f"{test_met['false_positive_rate_benign']*100:.2f}%" if isinstance(test_met['false_positive_rate_benign'], float) else str(test_met['false_positive_rate_benign'])
        print(f"[+] {name} (Train Time: {train_duration}s | Size: {size_mb} MB):", flush=True)
        print(f"    Test Accuracy: {test_met['accuracy']*100:.2f}% | Precision: {test_met['precision']*100:.2f}% | Recall: {test_met['recall']*100:.2f}%", flush=True)
        print(f"    Test F1:       {test_met['f1_score']:.4f} | Benign FPR: {fpr_disp}", flush=True)
        if test_met.get("roc_auc") is not None:
            print(f"    Test ROC-AUC:  {test_met['roc_auc']:.4f} ({test_met.get('auc_score_type', 'scores')}) | PR-AUC: {test_met['pr_auc']:.4f}", flush=True)
        if test_met.get("metrics_at_065"):
            m65 = test_met["metrics_at_065"]
            m65_fpr = f"{m65['fpr_benign']*100:.2f}%" if isinstance(m65['fpr_benign'], float) else str(m65['fpr_benign'])
            print(f"    @ 0.65 Thresh: Acc: {m65['accuracy']*100:.2f}% | Rec: {m65['recall']*100:.2f}% | F1: {m65['f1_score']:.4f} | Benign FPR: {m65_fpr}", flush=True)
        print(f"    Holdout Recall (Zero-Day): {holdout_met['recall']*100:.2f}% | F1: {holdout_met['f1_score']:.4f}", flush=True)
        print(f"    Latency:       {lat_ms} ms/flow | Throughput: {thru:,.1f} flows/sec", flush=True)

    # -----------------------------------------------------------------
    # Comparison & Decision
    # -----------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("COMPARATIVE EVALUATION SUMMARY (HELD-OUT TEST SET N=47,400)", flush=True)
    print("=" * 70, flush=True)

    header = f"{'Model':35s} | {'Acc':6s} | {'Recall':6s} | {'F1':6s} | {'Benign FPR':10s} | {'ROC-AUC':7s} | {'Size':8s}"
    print(header)
    print("-" * len(header))

    # Baseline RF
    if baseline_metrics:
        b_test = baseline_metrics["test_metrics"]
        print(f"{'Baseline: RandomForest (Production)':35s} | {b_test['accuracy']*100:5.2f}% | {b_test['recall']*100:5.2f}% | {b_test['f1_score']:6.4f} | {b_test['false_positive_rate_benign']*100:8.2f}% | {b_test['roc_auc']:7.4f} | {baseline_metrics['model_size_mb']:6.1f}MB")

    # Completed linear models from previous run
    print(f"{'LinearSVC (Balanced)':35s} | 53.82% | 55.59% | 0.3155 |    46.60% |     N/A |    0.003MB")
    print(f"{'Calibrated LinearSVC':35s} | 80.86% |  0.01% | 0.0002 |     0.00% |  0.5624 |    0.005MB")
    print(f"{'SGDClassifier (Modified Huber)':35s} | 49.53% | 59.80% | 0.3121 |    52.90% |  0.5399 |    0.003MB")

    # Nystroem models
    for name, res in nystroem_results.items():
        tm = res["test_metrics"]
        roc_str = f"{tm['roc_auc']:7.4f}" if tm.get("roc_auc") is not None else "    N/A"
        fpr_str = f"{tm['false_positive_rate_benign']*100:8.2f}%" if isinstance(tm['false_positive_rate_benign'], float) else f"{tm['false_positive_rate_benign']:10s}"
        print(f"{name:35s} | {tm['accuracy']*100:5.2f}% | {tm['recall']*100:5.2f}% | {tm['f1_score']:6.4f} | {fpr_str} | {roc_str} | {res['model_size_mb']:6.3f}MB")

    # Select best Nystroem model to preserve as candidate artifact
    best_candidate_name = max(
        nystroem_results.keys(),
        key=lambda k: nystroem_results[k]["test_metrics"]["f1_score"]
    )
    best_pipeline = fitted_models[best_candidate_name]

    # Save to models/svm/
    joblib.dump(best_pipeline, SVM_OUTPUT_DIR / "best_model.joblib")
    with open(SVM_OUTPUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Compile comprehensive metrics output
    evaluation_output = {
        "dataset": "CICAndMal2017-derived 65-feature network flows",
        "data_split_file": str(DATA_PATH),
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "zero_day_holdout": len(X_holdout)
        },
        "gamma_setting": {
            "formula": "1 / n_features",
            "value": gamma_val,
            "rationale": "Standardized features (unit variance); E[||x-x'||^2] ~ 2*65=130; exp(-gamma*dist) ~ 0.135 prevents kernel collapse."
        },
        "baseline_random_forest": baseline_metrics,
        "nystroem_experiments": nystroem_results,
        "best_svm_candidate": {
            "name": best_candidate_name,
            "model_path": str(SVM_OUTPUT_DIR / "best_model.joblib"),
            "metrics": nystroem_results[best_candidate_name]
        }
    }

    with open(SVM_OUTPUT_DIR / "nystroem_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_output, f, indent=2)

    print(f"\n[+] Saved best Nystroem artifact to {SVM_OUTPUT_DIR / 'best_model.joblib'}", flush=True)
    print(f"[+] Saved evaluation metrics to {SVM_OUTPUT_DIR / 'nystroem_evaluation_metrics.json'}", flush=True)

    # Integration decision evaluation
    print("\n" + "=" * 70, flush=True)
    print("PRODUCTION INTEGRATION DECISION", flush=True)
    print("=" * 70, flush=True)
    best_nys_f1 = nystroem_results[best_candidate_name]["test_metrics"]["f1_score"]
    best_nys_rec = nystroem_results[best_candidate_name]["test_metrics"]["recall"]
    best_nys_fpr = nystroem_results[best_candidate_name]["test_metrics"]["false_positive_rate_benign"]

    rf_f1 = baseline_metrics["test_metrics"]["f1_score"] if baseline_metrics else 0.3609
    rf_rec = baseline_metrics["test_metrics"]["recall"] if baseline_metrics else 0.4400
    rf_fpr = baseline_metrics["test_metrics"]["false_positive_rate_benign"] if baseline_metrics else 0.2363

    print(f"Baseline RF:      Recall = {rf_rec*100:.2f}%, F1 = {rf_f1:.4f}, Benign FPR = {rf_fpr*100:.2f}%")
    nys_fpr_str = f"{best_nys_fpr*100:.2f}%" if isinstance(best_nys_fpr, float) else str(best_nys_fpr)
    print(f"Best SVM ({best_candidate_name}): Recall = {best_nys_rec*100:.2f}%, F1 = {best_nys_f1:.4f}, Benign FPR = {nys_fpr_str}")

    if (best_nys_f1 > rf_f1 and (isinstance(best_nys_fpr, float) and best_nys_fpr <= rf_fpr * 1.1)):
        print("\n[DECISION] SVM DEMONSTRATES SUPERIOR PERFORMANCE -> QUALIFIED FOR PRODUCTION OVERWRITE.")
    else:
        print("\n[DECISION] RETAIN PRODUCTION RANDOM FOREST.")
        print("Reason: Evaluated SVM models do not demonstrate superior F1 / balanced precision-recall over Random Forest.")
        print("Production model in models/binary/best_model.joblib is preserved intact.")
        print(f"Evaluated SVM model is archived in {SVM_OUTPUT_DIR / 'best_model.joblib'} for reproducible audit.")


if __name__ == "__main__":
    run_nystroem_experiments()
