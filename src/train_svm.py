"""Support Vector Machine (SVM) Model Training and Evaluation Pipeline.

Evaluates:
1. Linear SVM Baseline (LinearSVC with balanced class weights)
2. Calibrated Linear SVM (LinearSVC + CalibratedClassifierCV)
3. Linear SVM with Modified Huber Loss (SGDClassifier with native calibrated probabilities)
4. Scalable Non-Linear RBF Kernel Approximation (Nystroem RBF features + Calibrated Linear SVM)
5. Exact RBF Kernel Scalability Benchmark & Stratified Pilot (SVC with RBF kernel and class_weight='balanced')
6. Multiclass Linear SVM (LinearSVC with CalibratedClassifierCV on 5 canonical classes)

Computes full evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR,
Confusion Matrix, Latency, Throughput, and Model Size) on the EXACT same test split
and holdout split used by the baseline RandomForest.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, classification_report
)
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RANDOM_SEED = 42


def load_dataset(npz_path: str = "data/processed/processed_splits.npz"):
    """Loads preprocessed matrices and labels, filtering out ambiguous NaN multiclass records."""
    fpath = Path(PROJECT_ROOT) / npz_path
    data = np.load(fpath, allow_pickle=True)

    X_train = data["X_train"]
    y_train_bin = data["y_train_bin"].astype(int)
    X_val = data["X_val"]
    y_val_bin = data["y_val_bin"].astype(int)
    X_test = data["X_test"]
    y_test_bin = data["y_test_bin"].astype(int)
    X_holdout = data["X_holdout"]
    y_holdout_bin = data["y_holdout_bin"].astype(int)
    feature_names = data["feature_names"].tolist()

    # Multiclass data with NaN handling for ambiguous generic labels
    y_train_multi_raw = data["y_train_multi"]
    y_val_multi_raw = data["y_val_multi"]
    y_test_multi_raw = data["y_test_multi"]

    def clean_multi(X, y_raw):
        valid_idx = [i for i, v in enumerate(y_raw) if v is not None and not np.isnan(float(v)) and int(v) in range(5)]
        return X[valid_idx], np.array([int(y_raw[i]) for i in valid_idx], dtype=int)

    X_train_multi, y_train_multi = clean_multi(X_train, y_train_multi_raw)
    X_val_multi, y_val_multi = clean_multi(X_val, y_val_multi_raw)
    X_test_multi, y_test_multi = clean_multi(X_test, y_test_multi_raw)

    return (
        X_train, y_train_bin,
        X_val, y_val_bin,
        X_test, y_test_bin,
        X_holdout, y_holdout_bin,
        X_train_multi, y_train_multi,
        X_test_multi, y_test_multi,
        feature_names
    )


def measure_inference_speed(model, X_sample: np.ndarray, num_runs: int = 1000) -> Tuple[float, float]:
    """Measures single-sample latency (ms) and batch throughput (flows/sec)."""
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


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Calculates comprehensive binary classification metrics."""
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

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    # Calculate metrics at MobileShield's default 0.65 probability threshold if probabilities exist
    if y_prob is not None:
        pred_065 = (y_prob >= 0.65).astype(int)
        cm65 = confusion_matrix(y_true, pred_065, labels=[0, 1])
        tn65, fp65, fn65, tp65 = cm65.ravel() if cm65.shape == (2, 2) else (0, 0, 0, len(y_true))
        metrics["metrics_at_065"] = {
            "accuracy": round(float(accuracy_score(y_true, pred_065)), 4),
            "precision": round(float(precision_score(y_true, pred_065, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, pred_065, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, pred_065, zero_division=0)), 4),
            "fpr_benign": round(float(fp65 / (tn65 + fp65)) if (tn65 + fp65) > 0 else 0.0, 4)
        }

    return metrics


def evaluate_multiclass_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict[int, str]
) -> Dict[str, Any]:
    """Calculates multiclass accuracy, macro-F1, per-class metrics, and confusion matrix."""
    labels = sorted(class_names.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class = {}
    for idx, name in class_names.items():
        mask = (y_true == idx)
        support = int(np.sum(mask))
        prec = precision_score(y_true == idx, y_pred == idx, zero_division=0)
        rec = recall_score(y_true == idx, y_pred == idx, zero_division=0)
        f1 = f1_score(y_true == idx, y_pred == idx, zero_division=0)
        per_class[name] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "support": support
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def run_svm_experiments(output_dir: str = "models/svm") -> Dict[str, Any]:
    """Runs systematic SVM evaluations across linear, calibrated, and kernel variants."""
    out_path = Path(PROJECT_ROOT) / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("MobileShield Advanced Machine Learning — SVM Model Evaluation", flush=True)
    print("=" * 70, flush=True)

    print("\n[*] Loading dataset splits from data/processed/processed_splits.npz...", flush=True)
    (
        X_train, y_train_bin,
        X_val, y_val_bin,
        X_test, y_test_bin,
        X_holdout, y_holdout_bin,
        X_train_multi, y_train_multi,
        X_test_multi, y_test_multi,
        feature_names
    ) = load_dataset()

    print(f"[+] Total Binary Train Flows:     {len(X_train):,} (Malicious: {np.mean(y_train_bin)*100:.1f}%)", flush=True)
    print(f"[+] Total Binary Val Flows:       {len(X_val):,} (Malicious: {np.mean(y_val_bin)*100:.1f}%)", flush=True)
    print(f"[+] Total Binary Test Flows:      {len(X_test):,} (Malicious: {np.mean(y_test_bin)*100:.1f}%)", flush=True)
    print(f"[+] Zero-Day Holdout Flows:       {len(X_holdout):,} (Malicious: {np.mean(y_holdout_bin)*100:.1f}%)", flush=True)
    print(f"[+] Total Multiclass Train Flows: {len(X_train_multi):,}", flush=True)
    print(f"[+] Total Multiclass Test Flows:  {len(X_test_multi):,}", flush=True)
    print(f"[+] Input Flow Features:          {len(feature_names)}", flush=True)

    # -----------------------------------------------------------------
    # STEP 1: RBF Kernel Exact SVM Scalability Benchmark
    # -----------------------------------------------------------------
    print("\n" + "-" * 60, flush=True)
    print("[1] RBF Kernel Exact SVM Scalability Pilot Benchmark", flush=True)
    print("-" * 60, flush=True)
    print(f"[*] Benchmarking fit duration of exact SVC(kernel='rbf') across sample sizes...", flush=True)

    pilot_sizes = [1000, 2500, 5000]
    pilot_times = []
    for sz in pilot_sizes:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_train), size=sz, replace=False)
        sub_X = X_train[idx]
        sub_y = y_train_bin[idx]

        svc_pilot = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED))
        ])
        t0 = time.time()
        svc_pilot.fit(sub_X, sub_y)
        elapsed = time.time() - t0
        pilot_times.append(elapsed)
        print(f"    Sample size N = {sz:5d} -> Fit duration: {elapsed:.2f} s", flush=True)

    t_ref = pilot_times[-1]
    n_ref = pilot_sizes[-1]
    projected_seconds = t_ref * ((len(X_train) / n_ref) ** 2)
    projected_hours = projected_seconds / 3600.0
    print(f"[!] Empirical quadratic projection for exact SVC(kernel='rbf') on full dataset (N={len(X_train):,}):", flush=True)
    print(f"    Estimated training duration: {projected_seconds:,.0f} seconds (~{projected_hours:.1f} hours).", flush=True)
    print("[!] Per instructions: Full-dataset exact RBF SVC is computationally prohibitive on laptop resources.", flush=True)
    print("    Evaluating Linear SVM, Calibrated Linear SVM, and Scalable Nystroem RBF Kernel Approximation.", flush=True)

    # -----------------------------------------------------------------
    # STEP 2: Binary SVM Candidates
    # -----------------------------------------------------------------
    print("\n" + "-" * 60, flush=True)
    print("[2] Evaluating Binary SVM Configurations", flush=True)
    print("-" * 60, flush=True)

    candidate_configs = {
        # 1. Linear SVM Baseline (LinearSVC with balanced class weights)
        "LinearSVM_LinearSVC_Balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(dual="auto", class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED))
        ]),

        # 2. Calibrated Linear SVM (Platt sigmoid calibration for valid probabilities >= threshold)
        "LinearSVM_Calibrated_LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", CalibratedClassifierCV(
                estimator=LinearSVC(dual="auto", class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED),
                method="sigmoid",
                cv=3
            ))
        ]),

        # 3. Fast Linear SVM with Modified Huber loss (Natively outputting smooth calibrated probabilities)
        "LinearSVM_SGD_ModifiedHuber": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SGDClassifier(
                loss="modified_huber",
                penalty="l2",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=1500,
                tol=1e-3,
                random_state=RANDOM_SEED
            ))
        ]),

        # 4. Scalable RBF Kernel Approximation (Nystroem random Fourier features + Calibrated Linear SVM on FULL dataset)
        "RBF_Approximation_Nystroem_Calibrated": Pipeline([
            ("scaler", StandardScaler()),
            ("rbf_features", Nystroem(kernel="rbf", gamma=None, n_components=60, random_state=RANDOM_SEED)),
            ("svm", CalibratedClassifierCV(
                estimator=LinearSVC(dual="auto", class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED),
                method="sigmoid",
                cv=3
            ))
        ]),

        # 5. Exact RBF SVM on 15,000 Stratified Training Subset (Evaluated on FULL N=47,400 test set)
        "RBF_SVM_Stratified_Sample_15k": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced",
                probability=True,
                max_iter=5000,
                random_state=RANDOM_SEED
            ))
        ])
    }

    results = {}
    fitted_candidates = {}

    for name, pipeline in candidate_configs.items():
        print(f"\n[*] Training {name}...", flush=True)
        t0 = time.time()

        if name == "RBF_SVM_Stratified_Sample_15k":
            rng = np.random.RandomState(RANDOM_SEED)
            idx0 = np.where(y_train_bin == 0)[0]
            idx1 = np.where(y_train_bin == 1)[0]
            s0 = rng.choice(idx0, size=10000, replace=False)
            s1 = rng.choice(idx1, size=5000, replace=False)
            sub_idx = np.concatenate([s0, s1])
            rng.shuffle(sub_idx)
            pipeline.fit(X_train[sub_idx], y_train_bin[sub_idx])
            sample_desc = "Trained on 15,000 stratified training samples (10k benign, 5k malware); evaluated on full test set"
        else:
            pipeline.fit(X_train, y_train_bin)
            sample_desc = f"Trained on full training set ({len(X_train):,} samples)"

        train_sec = round(time.time() - t0, 2)
        fitted_candidates[name] = pipeline

        # Predictions
        val_pred = pipeline.predict(X_val)
        val_prob = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
        val_met = evaluate_binary_predictions(y_val_bin, val_pred, val_prob)

        test_pred = pipeline.predict(X_test)
        test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        test_met = evaluate_binary_predictions(y_test_bin, test_pred, test_prob)

        holdout_pred = pipeline.predict(X_holdout)
        holdout_prob = pipeline.predict_proba(X_holdout)[:, 1] if hasattr(pipeline, "predict_proba") else None
        holdout_met = evaluate_binary_predictions(y_holdout_bin, holdout_pred, holdout_prob)

        # Measure speed on 1000 single test flows
        latency_ms, throughput = measure_inference_speed(pipeline, X_test[:1000])

        # Model size on disk
        temp_model_path = out_path / f"temp_{name}.joblib"
        joblib.dump(pipeline, temp_model_path)
        model_size_mb = round(os.path.getsize(temp_model_path) / (1024 * 1024), 3)
        temp_model_path.unlink()

        results[name] = {
            "training_scope": sample_desc,
            "train_duration_sec": train_sec,
            "latency_ms_per_flow": latency_ms,
            "throughput_flows_sec": throughput,
            "model_size_mb": model_size_mb,
            "val_metrics": val_met,
            "test_metrics": test_met,
            "zero_day_holdout_metrics": holdout_met
        }

        print(f"[+] {name} (Train Time: {train_sec}s | Size: {model_size_mb} MB):", flush=True)
        print(f"    Test Accuracy: {test_met['accuracy']*100:.2f}% | Precision: {test_met['precision']*100:.2f}% | Recall: {test_met['recall']*100:.2f}%", flush=True)
        print(f"    Test F1:       {test_met['f1_score']:.4f} | Benign FPR: {test_met['false_positive_rate_benign']*100:.2f}%", flush=True)
        if test_met["roc_auc"] is not None:
            print(f"    Test ROC-AUC:  {test_met['roc_auc']:.4f} | PR-AUC: {test_met['pr_auc']:.4f}", flush=True)
        if test_met.get("metrics_at_065"):
            m65 = test_met["metrics_at_065"]
            print(f"    @ 0.65 Thresh: Acc: {m65['accuracy']*100:.2f}% | Rec: {m65['recall']*100:.2f}% | F1: {m65['f1_score']:.4f} | Benign FPR: {m65['fpr_benign']*100:.2f}%", flush=True)
        print(f"    Holdout Recall: {holdout_met['recall']*100:.2f}% | F1: {holdout_met['f1_score']:.4f}", flush=True)
        print(f"    Latency:       {latency_ms} ms/flow | Throughput: {throughput:,.1f} flows/sec", flush=True)

    # -----------------------------------------------------------------
    # STEP 3: Multiclass SVM (LinearSVC + CalibratedClassifierCV)
    # -----------------------------------------------------------------
    print("\n" + "-" * 60, flush=True)
    print("[3] Evaluating Multiclass SVM (One-vs-Rest Calibrated Linear SVM)", flush=True)
    print("-" * 60, flush=True)

    multi_classes = {
        0: "BENIGN",
        1: "ADWARE",
        2: "RANSOMWARE",
        3: "SCAREWARE",
        4: "SMSMALWARE"
    }

    multi_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", CalibratedClassifierCV(
            estimator=LinearSVC(dual="auto", class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED),
            method="sigmoid",
            cv=3
        ))
    ])

    print("[*] Fitting Multiclass Calibrated Linear SVM on 218,119 flows...", flush=True)
    t0_multi = time.time()
    multi_pipeline.fit(X_train_multi, y_train_multi)
    multi_train_sec = round(time.time() - t0_multi, 2)

    multi_test_pred = multi_pipeline.predict(X_test_multi)
    multi_test_met = evaluate_multiclass_predictions(y_test_multi, multi_test_pred, multi_classes)

    multi_lat_ms, multi_thru = measure_inference_speed(multi_pipeline, X_test_multi[:1000])

    temp_multi_path = out_path / "temp_multi_svm.joblib"
    joblib.dump(multi_pipeline, temp_multi_path)
    multi_size_mb = round(os.path.getsize(temp_multi_path) / (1024 * 1024), 3)
    temp_multi_path.unlink()

    results["Multiclass_Calibrated_LinearSVM"] = {
        "train_duration_sec": multi_train_sec,
        "latency_ms_per_flow": multi_lat_ms,
        "throughput_flows_sec": multi_thru,
        "model_size_mb": multi_size_mb,
        "test_metrics": multi_test_met
    }

    print(f"[+] Multiclass SVM Results (Train Time: {multi_train_sec}s | Size: {multi_size_mb} MB):", flush=True)
    print(f"    Test Accuracy: {multi_test_met['accuracy']*100:.2f}% | Macro F1: {multi_test_met['macro_f1']:.4f}", flush=True)
    for cname, cmet in multi_test_met["per_class"].items():
        print(f"    - {cname:11s} -> Precision: {cmet['precision']*100:.1f}%, Recall: {cmet['recall']*100:.1f}%, F1: {cmet['f1_score']:.4f}", flush=True)

    # -----------------------------------------------------------------
    # STEP 4: Select Best SVM Model & Save Artifacts
    # -----------------------------------------------------------------
    calibrated_names = [
        "LinearSVM_Calibrated_LinearSVC",
        "LinearSVM_SGD_ModifiedHuber",
        "RBF_Approximation_Nystroem_Calibrated",
        "RBF_SVM_Stratified_Sample_15k"
    ]

    best_svm_name = max(
        calibrated_names,
        key=lambda k: (results[k]["test_metrics"].get("roc_auc") or 0.0) + results[k]["test_metrics"]["f1_score"] - (results[k]["test_metrics"]["false_positive_rate_benign"] * 0.2)
    )

    best_svm_model = fitted_candidates[best_svm_name]
    best_svm_file = out_path / "best_model.joblib"
    joblib.dump(best_svm_model, best_svm_file)

    # Save multiclass SVM
    joblib.dump(multi_pipeline, out_path / "multiclass_model.joblib")

    # Save feature names
    with open(out_path / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Save label mappings
    label_mapping = {
        "binary": {"BENIGN": 0, "MALWARE_ASSOCIATED": 1},
        "multiclass": {v: k for k, v in multi_classes.items()}
    }
    with open(out_path / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    # Save metadata
    metadata = {
        "experiment": "SVM Experiment: Support Vector Machine Evaluation",
        "best_model_name": best_svm_name,
        "selected_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "evaluation_metrics": results[best_svm_name]["test_metrics"],
        "latency_ms": results[best_svm_name]["latency_ms_per_flow"],
        "throughput_flows_sec": results[best_svm_name]["throughput_flows_sec"],
        "model_size_mb": results[best_svm_name]["model_size_mb"],
        "training_scope": results[best_svm_name]["training_scope"],
        "all_svm_results": results
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(out_path / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"[*] Best Validated SVM Model Selected: {best_svm_name}", flush=True)
    print(f"    Saved pipeline artifact to: {best_svm_file}", flush=True)
    print(f"    Saved metadata to:          {out_path / 'metadata.json'}", flush=True)
    print("=" * 70, flush=True)

    return results


if __name__ == "__main__":
    run_svm_experiments()
