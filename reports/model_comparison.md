# Model Comparison Report: Phase 1 Android Malware-Associated Flow Classification

This report details saved benchmarks from one group-aware train/validation/test split for both **Experiment A (Binary)** and **Experiment B (Multiclass)**. These are not cross-validation results.

## 1. Experiment A: Binary Classification (BENIGN vs MALWARE_ASSOCIATED)

| Model | Train Time (s) | Val F1 | Test F1 | Test Accuracy | Benign FPR (%) | Unseen-Family Holdout Recall (%) | Latency (ms) | Model Throughput (flows/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline_LogisticRegression** | 8.78 | 0.2952 | 0.3148 | 0.5386 | 46.5% | 55.19% | 0.6239 | 1,560,792.9 |
| **RandomForest** | 57.29 | 0.3594 | 0.3609 | 0.7018 | 23.63% | 42.24% | 56.5545 | 13,775.9 |
| **ExtraTrees** | 50.25 | 0.3364 | 0.344 | 0.6916 | 24.47% | 40.22% | 51.2607 | 13,833.0 |
| **LightGBM** | 4.15 | 0.3598 | 0.3738 | 0.6257 | 36.44% | 57.26% | 3.557 | 135,744.1 |

### Key Observations (Binary):
- RandomForest and LightGBM trade off benign false positives and malware recall; neither dominates both. RandomForest has test F1 0.3609 and benign FPR 23.63%; LightGBM has test F1 0.3738, benign FPR 36.44%, and malware recall 58.37%.
- The throughput values measure model prediction only, not end-to-end capture, aggregation, relay, or dashboard capacity.
- The binary test set is imbalanced: an always-benign classifier would achieve about 80.9% accuracy, above each candidate's saved test accuracy.

## 2. Experiment B: Multiclass Classification (5 Canonical Categories)

| Model | Train Time (s) | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Benign False Alarm (%) | Unseen-Family Holdout Macro-F1 | Latency (ms) | Model Throughput (flows/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline_LogisticRegression** | 754.06 | 0.174 | 0.1743 | 0.3435 | 62.22% | 0.1736 | 0.3194 | 234,115.3 |
| **RandomForest** | 36.93 | 0.2588 | 0.2552 | 0.5035 | 43.95% | 0.2081 | 55.8813 | 9,474.1 |
| **ExtraTrees** | 38.05 | 0.2527 | 0.2466 | 0.5192 | 41.18% | 0.1788 | 38.7304 | 24,478.0 |
| **LightGBM** | 6.08 | 0.2252 | 0.2171 | 0.3614 | 62.76% | 0.2628 | 1.6614 | 134,850.9 |

### Key Observations (Multiclass):
- RandomForest scored the highest Test Macro-F1 (0.2552); ExtraTrees had higher accuracy (51.92%) and lower benign false alarm (41.10%). All models have weak per-class malware performance.
- Separating malware families into 4 specific categories (Adware vs Ransomware vs Scareware vs SMSmalware) purely based on Layer 3/4 flow statistics is challenging because multiple malware types reuse common advertising SDKs and HTTP C2 mechanisms.
