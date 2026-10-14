# In-Depth Model Evaluation Report: Phase 1

## 1. Selected Best Models Summary

- **Binary Champion:** `RandomForest` (models/binary/best_model.joblib)
- **Multiclass Champion:** `RandomForest` (models/multiclass/best_model.joblib)

## 2. Binary Model Confusion Matrix (Test Set)

```
True Negative (Benign -> Benign):  29,271
False Positive (Benign -> Malware): 9,056
False Negative (Malware -> Benign): 5,081
True Positive (Malware -> Malware):  3,992
```

## 3. Multiclass Per-Class Metrics (Random Forest Test Set)

| Class Name | Precision | Recall | F1-Score | Support (Flows) |
| :--- | :--- | :--- | :--- | :--- |
| **BENIGN** | 0.8686 | 0.5605 | 0.6814 | 38,327 |
| **ADWARE** | 0.0882 | 0.279 | 0.134 | 2,100 |
| **RANSOMWARE** | 0.1384 | 0.3116 | 0.1917 | 2,250 |
| **SCAREWARE** | 0.0792 | 0.2165 | 0.1159 | 2,268 |
| **SMSMALWARE** | 0.1147 | 0.2291 | 0.1529 | 2,305 |

## 4. Selected Unseen-Family Holdout Evaluation

Four selected malware families were held out from training. This finite set is not a prospective or general zero-day evaluation:
- Adware: `Feiwo`
- Ransomware: `Charger`
- Scareware: `FakeAV`
- SMSmalware: `Fakeinst`

- **Binary unseen-family holdout recall:** 42.24%
- **Binary unseen-family holdout F1:** 0.5939
- **Multiclass unseen-family holdout macro-F1:** 0.2081

The binary family holdout contains malware examples only, so its zero benign
false-positive count is not an estimate of benign false-positive rate. These
metrics use estimator `predict` outputs, do not evaluate the dashboard's
separate 65% default confidence alert gate, and do not represent cross-validation or
physical-device malware-detection efficacy.
