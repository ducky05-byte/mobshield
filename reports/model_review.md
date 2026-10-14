# Saved Model Artifact Review

This review interprets the currently saved evaluation metrics and model
artifacts. It does not retrain a model or replace the configured Random Forest
artifacts.

## Evidence and evaluation boundary

- The saved processed split records 218,119 training flows, 46,500 validation
  flows, 47,400 binary test flows, and 6,750 malware-family holdout flows.
- The split manifest reports 1,455, 310, 316, and 45 APK samples in those
  partitions respectively. The training/validation/test partitioning is
  grouped by sample ID; the holdout contains selected families excluded from
  training.
- The binary test confusion matrix has 38,327 benign and 9,073
  malware-associated examples. An always-benign classifier would therefore
  achieve 80.86% test accuracy. Accuracy alone is not an adequate performance
  measure for these results.
- The multiclass test array contains 47,400 rows, of which 150 have no
  multiclass label; the saved per-class metrics account for the remaining
  47,250 labeled rows.
- The reported metrics are from one train/validation/test split, not
  cross-validation. Reported binary/multiclass classification metrics use the
  estimators' ordinary `predict` outputs; they do not evaluate the dashboard's
  live confidence alert gate. The checked-in backend, dashboard, and tests
  specify a 65% (`0.65`) default, not 60%; that configured value was not changed.
- Inference latency and throughput are model-prediction benchmarks, not
  end-to-end packet capture, flow aggregation, relay, or dashboard throughput.

## Binary model comparison

| Model | Test F1 | Test accuracy | Benign false-positive rate | Malware recall |
|---|---:|---:|---:|---:|
| Random Forest (saved model) | 0.3609 | 70.18% | 23.63% | 44.00% |
| LightGBM | 0.3738 | 62.57% | 36.44% | 58.37% |

Random Forest provides the lower benign false-positive rate, while LightGBM
has higher malware recall and F1 on the saved test split. Neither dominates
both false positives and false negatives. Random Forest's 23.63% benign false
positive rate and 44.00% malware recall are not strong evidence of reliable
malware detection. Its 70.18% accuracy is also below the 80.86% always-benign
baseline for this imbalanced test set.

The 6,750-record binary family holdout contains only malware-associated
examples. It can measure recall on those selected families, but cannot
estimate the benign false-positive rate. The saved Random Forest holdout
recall is 42.24%; the corresponding zero benign false-positive count is
inapplicable, not proof of no false alarms. This finite family holdout is not
a prospective or general zero-day evaluation.

## Multiclass model

The saved Random Forest has test macro-F1 0.2552, accuracy 50.35%, and benign
false-alarm rate 43.95%. The per-class test F1 scores in the saved evaluation
report range from 0.1159 to 0.1917 for malware families, indicating weak
family classification. ExtraTrees has lower macro-F1 (0.2466) but higher
accuracy (51.92%) and a slightly lower benign false-alarm rate (41.10%).
These results do not demonstrate dependable family identification.

## Conclusion and outstanding validation

Retain the existing models and alert policy unchanged, as requested, but treat
their predictions as diagnostics rather than verified malware findings. Do
not claim that a model or threshold policy has improved from these artifacts.
No cross-validation, prospective dataset evaluation, threshold-specific
evaluation, or measured physical-phone malware-detection test is represented.
Packet-delivery verification alone does not establish detection accuracy.
