# MobileShield Advanced Machine Learning — Support Vector Machine (SVM) Evaluation & Tuning Report

**Project**: MobileShield — Android Network Intrusion Detection System  
**Dataset**: CICAndMal2017-derived Network Flow Dataset  
**Feature Schema**: 65 Bidirectional Network Flow Features (Normalized & Standardized)  
**Evaluation Date**: October 14, 2026  

---

## 1. Existing Production Estimator Confirmed From Code

Inspection of the existing production training pipelines (`src/train_binary.py`, `src/train_multiclass.py`) and saved model metadata (`models/binary/metadata.json`, `models/multiclass/metadata.json`) confirms:

* **Production Estimator**: `RandomForestClassifier`
* **Hyperparameters**: `n_estimators=100`, `max_depth=20`, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight="balanced"`, `random_state=42`.
* **Artifact Path**: `models/binary/best_model.joblib`
* **Artifact Size**: 165.71 MB
* **Inference Pipeline**: `src/predict_flow.py` (`FlowPredictor`), `src/flow_engine.py` (`LiveCapturePipeline`), `src/companion_api.py` (FastAPI), `src/dashboard.py` (Streamlit).

---

## 2. SVM Variants Evaluated

To evaluate Support Vector Machines systematically and prevent hanging laptop resources on large-scale flow data, the following variants were trained and benchmarked:

1. **RBF Scalability Pilot Benchmark**:
   * Evaluated exact `SVC(kernel='rbf')` on pilot subsets of $N \in \{1000, 2500, 5000\}$.
   * Empirically projected fit time on the full training set ($N = 218,119$) is between **2,193 and 6,558 seconds** (~0.6 to 1.8 hours) with $O(N^2)$ memory scaling, confirming full-dataset exact RBF SVM is computationally prohibitive on local hardware.
2. **Linear SVM Baseline (`LinearSVC(class_weight='balanced')`)**:
   * Standardized linear SVM with balanced penalty weights, optimizing the primal/dual hinge loss.
3. **Calibrated Linear SVM (`CalibratedClassifierCV(LinearSVC)`)**:
   * 3-fold Platt sigmoid calibration to map decision margins to probability estimates $P(y=1|x) \in [0, 1]$.
4. **Stochastic Linear SVM (`SGDClassifier(loss='modified_huber')`)**:
   * Convex smooth loss outputting calibrated probability estimates natively without multi-fold Platt overhead.
5. **Scalable Nystroem RBF Approximation — Balanced LinearSVC**:
   * Approximated RBF kernel Hilbert space mapping using Monte Carlo Nystroem sampling (`n_components=100`, `gamma = 1 / n_features = 1/65 = 0.015385`). Evaluates the raw uncalibrated non-linear decision boundary.
6. **Scalable Nystroem RBF Approximation — Calibrated LinearSVC**:
   * Nystroem RBF feature map with 3-fold Platt sigmoid calibration for valid probability outputs.
7. **Scalable Nystroem RBF Approximation — Modified Huber**:
   * Nystroem RBF feature map with fast probabilistic SGD classification.

---

## 3. Dataset Size, Splits, and Preprocessing Contract

All experiments reused the exact pre-extracted dataset splits from `data/processed/processed_splits.npz`, ensuring zero data leakage:

* **Training Set**: $N = 218,119$ flows (178,419 Benign [81.8%], 39,700 Malicious [18.2%])
* **Validation Set**: $N = 46,500$ flows (38,250 Benign [82.3%], 8,250 Malicious [17.7%])
* **Held-Out Test Set**: $N = 47,400$ flows (38,328 Benign [80.86%], 9,072 Malicious [19.14%])
* **Zero-Day Holdout Set**: $N = 6,750$ flows (100.0% Malicious, unobserved malware families)
* **Features**: Exactly 65 continuous flow features, matching `models/feature_names.json`.
* **Preprocessing**: `StandardScaler` embedded within an `sklearn.pipeline.Pipeline`, fit strictly on the training partition.

### Resolution of ROC-AUC Single-Class Warning & Zero-Day Holdout Distinction
* **Root Cause**: The earlier runtime warning (`UndefinedMetricWarning: Only one class is present in y_true`) was triggered because `X_holdout`, `y_holdout_bin` consists entirely of malware samples ($N=6,750$, $100\%$ positive). Because true negatives are zero, ROC-AUC and FPR are mathematically undefined on the holdout split.
* **Resolution**: The evaluation suite checks `len(np.unique(y_true)) > 1` before computing ROC-AUC / PR-AUC. The **Zero-Day Holdout** is strictly evaluated as a **malware detection recall test set** on unseen malware variants, distinct from the **mixed-class held-out test set** ($N=47,400$) where full ROC-AUC, PR-AUC, and FPR metrics are computed.

### Gamma Choice Documentation
* **Formula**: $\gamma = \frac{1}{n_{\text{features}}} = \frac{1}{65} \approx 0.015385$.
* **Rationale**: Because input features are normalized with zero mean and unit variance, the expected squared Euclidean distance between two standardized vectors is $\mathbb{E}[\|x - x'\|^2] \approx 2 \times 65 = 130$. Using $\gamma = \frac{1}{65}$ ensures $\exp(-\gamma \|x - x'\|^2) \approx \exp(-2) \approx 0.135$, maintaining well-conditioned kernel matrices without vanishing kernel similarities or kernel collapse.

---

## 4. Baseline-versus-SVM Comparative Evaluation (Test Set $N=47,400$)

All models were evaluated on the exact same held-out test split ($N=47,400$):

| Model Architecture | Test Acc | Precision | Recall | F1-Score | Benign FPR | ROC-AUC | PR-AUC | At 0.65 Alert Threshold (Acc / Rec / FPR) | Zero-Day Recall | Train Time | Latency / Flow | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Random Forest (Production)** | **70.18%** | **30.59%** | **44.00%** | **0.3609** | **23.63%** | **0.6629** | **0.3609** | **81.34% / 15.45% / 3.07%** | **42.24%** | ~120s | 53.35 ms | 165.71 MB |
| LinearSVC (Balanced) | 53.82% | 22.02% | 55.59% | 0.3155 | 46.60% | N/A | N/A | N/A (Uncalibrated score) | 55.33% | 16.48s | 0.26 ms | 0.003 MB |
| Calibrated LinearSVC (Sigmoid CV=3) | 80.86% | 100.0% | 0.01% | 0.0002 | 0.00% | 0.5624 | 0.2325 | 80.86% / 0.00% / 0.00% | 0.00% | 28.95s | 1.80 ms | 0.005 MB |
| SGDClassifier (Modified Huber) | 49.53% | 21.11% | 59.80% | 0.3121 | 52.90% | 0.5399 | 0.2112 | 74.27% / 13.88% / 11.43% | 57.60% | 9.19s | 0.27 ms | 0.003 MB |
| Nystroem RBF LinearSVC (Balanced) | 57.51% | 22.68% | 50.63% | 0.3133 | 40.86% | 0.5728* | 0.2455* | N/A (Decision margin) | 49.07% | 36.88s | 2.22 ms | 0.130 MB |
| Nystroem RBF Calibrated LinearSVC | 80.86% | 33.33% | 0.01% | 0.0002 | 0.01% | 0.5720 | 0.2444 | 80.86% / 0.00% / 0.00% | 0.06% | 64.52s | 4.02 ms | 0.133 MB |
| **Nystroem RBF SGD (Modified Huber)** | 45.64% | 21.37% | **68.68%** | 0.3260 | **59.82%** | 0.5707 | 0.2441 | 80.18% / 4.59% / 1.93% | **68.22%** | 4.41s | 1.24 ms | 0.131 MB |

*\*Note: Computed from continuous raw `decision_function` margins.*

---

## 5. Detailed Metric Analysis & Key Findings

1. **Discriminative Capacity (ROC-AUC)**:
   * Random Forest achieves an ROC-AUC of **0.6629** and PR-AUC of **0.3609**.
   * In contrast, all linear and RBF-approximated SVMs cluster around ROC-AUC **0.54 to 0.57** (PR-AUC **0.21 to 0.24**). High-dimensional network flow statistics exhibit non-linear feature interactions and multimodality that simple linear hyperplanes and random Fourier projections struggle to separate.
2. **False Positive Rate (FPR) vs. Recall Tradeoff**:
   * While `Nystroem_RBF_SGD_ModifiedHuber` achieved high unthresholded recall (**68.68%** on test, **68.22%** on zero-day holdout), its Benign False Positive Rate was catastrophic: **59.82%**. In a live Android mobile deployment, this would cause approximately 6 out of every 10 benign background connections to trigger false malware alerts.
   * At the default MobileShield **0.65 alert threshold**, Nystroem SGD's recall drops to **4.59%** (F1: 0.0813), while Random Forest retains **15.45%** recall with a tight **3.07%** FPR (F1: 0.2407).
3. **Platt Calibration Collapse**:
   * When standard Platt sigmoid calibration (`CalibratedClassifierCV`) was applied over balanced LinearSVC, the sigmoid optimizer fitted on unweighted class marginals (81.8% benign vs. 18.2% malware) compressed output probabilities below 0.50, causing recall to collapse to **0.01%**.

---

## 6. Best SVM Configuration

* **Selected Candidate**: `Nystroem_RBF_SGD_ModifiedHuber`
* **Pipeline Structure**:
  ```python
  Pipeline([
      ("scaler", StandardScaler()),
      ("nystroem", Nystroem(kernel="rbf", gamma=0.0153846, n_components=100, random_state=42)),
      ("svm", SGDClassifier(loss="modified_huber", penalty="l2", alpha=1e-4, class_weight="balanced", random_state=42))
  ])
  ```
* **Strengths**: Highest F1 (0.3260) and Zero-Day Recall (68.22%) among SVM variants; ultra-fast training (4.41 seconds); compact footprint (0.131 MB vs. 165.7 MB); fast latency (1.24 ms/flow).
* **Saved Artifact**: `models/svm/best_model.joblib` and `models/svm/nystroem_evaluation_metrics.json`.

---

## 7. Production Integration Decision

### DECISION: RETAIN PRODUCTION RANDOM FOREST

**Measured Justification**:
Per the engineering requirements (*"Do not claim improvement unless the evaluation results demonstrate it. If the SVM is worse or too slow, retain the existing production model and report the measured reason. Do not force SVM into production just because it was requested."*):
1. **F1-Score**: Baseline Random Forest achieves **0.3609** vs. best SVM **0.3260**.
2. **ROC-AUC**: Random Forest achieves **0.6629** vs. best SVM **0.5707** (a +0.092 margin in discrimination).
3. **False Alarm Rate**: Random Forest maintains an acceptable benign FPR of **23.63%** (and **3.07%** at the 0.65 threshold), whereas the highest-recall SVM incurs an unviable **59.82%** false alarm rate.
4. **Production Stability**: The active production model in `models/binary/best_model.joblib` remains the proven Random Forest artifact. The candidate SVM pipeline is fully serialized and archived in `models/svm/` for reproducible benchmarking and future research.

---

## 8. Files Changed & Added

| File Path | Action | Description |
| :--- | :---: | :--- |
| `src/train_svm.py` | Updated | Added RBF scalability pilot benchmark, LinearSVC, SGD Huber, and Multiclass SVM runners. |
| `src/train_nystroem_svm.py` | Added | Dedicated, isolated Nystroem RBF approximation experiment script with explicit $\gamma = 1/n_{\text{features}}$, probability calibration, and threshold metrics. |
| `tests/test_svm.py` | Added | 6 targeted unit tests validating SVM pipeline loading, 65-feature contract, probability normalization, threshold 0.65 behavior, and backend compatibility. |
| `models/svm/best_model.joblib` | Added | Serialized Nystroem RBF SVM pipeline artifact. |
| `models/svm/feature_names.json` | Added | 65-feature contract specification for SVM inference. |
| `models/svm/nystroem_evaluation_metrics.json` | Added | Full machine-readable evaluation metrics across all configurations. |
| `reports/SVM_TUNING_REPORT.md` | Added | This comprehensive technical report. |

---

## 9. Test Execution & Verification

All automated tests and compilation checks were executed with verified results:

```powershell
# Unit Test Suite
python -m pytest tests/test_svm.py tests/test_models.py -v
```
**Result**: **10 passed in 15.01s** (100% pass rate).
* `test_svm_artifact_loading`: PASSED
* `test_svm_feature_contract`: PASSED
* `test_svm_threshold_and_probabilities`: PASSED
* `test_benign_flows_not_all_flagged_malicious`: PASSED
* `test_backend_interface_compatibility`: PASSED
* `test_backend_and_companion_imports`: PASSED
* `test_model_loading`: PASSED
* `test_binary_prediction`: PASSED
* `test_multiclass_prediction`: PASSED
* `test_prediction_determinism`: PASSED

```powershell
# Python Compilation Verification
python -m py_compile src/dashboard.py src/companion_api.py src/flow_engine.py src/predict_flow.py src/prepare_dataset.py src/train_nystroem_svm.py
```
**Result**: Exit code 0 (zero syntax or typing errors).

---

## 10. Operational Run Commands

To launch MobileShield services with the verified production pipeline:

### 1. Start FastAPI Companion API
```powershell
cd C:\Users\Varun\Desktop\AML
uvicorn src.companion_api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start Streamlit Interactive Dashboard
```powershell
cd C:\Users\Varun\Desktop\AML
streamlit run src/dashboard.py
```

### 3. Re-run Nystroem SVM Experiment (Reproducibility)
```powershell
cd C:\Users\Varun\Desktop\AML
python src/train_nystroem_svm.py
```

---

## 11. Remaining Limitations & Recommendations

1. **Feature Space Geometry**:
   * Android network flow distributions exhibit extreme skewness and heavy tails across duration, inter-arrival times (IAT), and byte lengths. Tree-based partitioning (Random Forest, XGBoost) natively accommodates these piecewise orthogonal boundaries without suffering from feature collinearity or extreme outlier distortion.
2. **Kernel Scaling vs. Memory**:
   * Exact kernel SVMs ($O(N^2)$ to $O(N^3)$ computational complexity) do not scale to large network flow datasets ($N > 200,000$) on standard hardware.
   * Nystroem random sampling scales linearly with sample size ($O(m^2 N)$), but with $m=100$ components, its empirical discriminative power (ROC-AUC ~0.57) lags behind tree ensembles.
3. **Recommendation**:
   * Maintain `RandomForestClassifier` for production flow inference.
   * If model size reduction (from 165 MB) is desired in future phases, explore tree pruning (`max_depth=12`, `min_samples_leaf=4`) or post-training quantization before considering linear kernel methods.
