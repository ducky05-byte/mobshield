"""Evaluation Report Generator for Phase 1 Models.

Ingests saved evaluation metrics and feature importances from `models/binary`
and `models/multiclass`, and compiles:
  - `reports/model_comparison.md`
  - `reports/evaluation_report.md`
  - `reports/feature_importance.md`
  - CSV summaries in `reports/`
"""

import json
from pathlib import Path
import pandas as pd


def generate_evaluation_reports(
    binary_dir: str = "models/binary",
    multiclass_dir: str = "models/multiclass",
    reports_dir: str = "reports"
):
    bin_path = Path(binary_dir)
    multi_path = Path(multiclass_dir)
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    with open(bin_path / "evaluation_metrics.json", "r", encoding="utf-8") as f:
        bin_metrics = json.load(f)

    with open(multi_path / "evaluation_metrics.json", "r", encoding="utf-8") as f:
        multi_metrics = json.load(f)

    with open(bin_path / "feature_importances.json", "r", encoding="utf-8") as f:
        bin_fi = json.load(f)

    with open(multi_path / "feature_importances.json", "r", encoding="utf-8") as f:
        multi_fi = json.load(f)

    # 1. Model Comparison Report (reports/model_comparison.md)
    comp_file = rep_path / "model_comparison.md"
    with open(comp_file, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Report: Phase 1 Android Malware-Associated Flow Classification\n\n")
        f.write("This report details saved empirical benchmarks across all 4 candidate models for both **Experiment A (Binary)** and **Experiment B (Multiclass)** under group-aware evaluation. These are single held-out split metrics, not cross-validation results.\n\n")

        f.write("## 1. Experiment A: Binary Classification (BENIGN vs MALWARE_ASSOCIATED)\n\n")
        f.write("| Model | Train Time (s) | Val F1 | Test F1 | Test Accuracy | Benign FPR (%) | Unseen-Family Holdout Recall (%) | Latency (ms) | Model Throughput (flows/s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        bin_rows = []
        for mname, data in bin_metrics.items():
            t_met = data["test_metrics"]
            v_met = data["val_metrics"]
            h_met = data["zero_day_holdout_metrics"]
            row = {
                "Model": mname,
                "Train Time (s)": data["train_duration_sec"],
                "Val F1": v_met["f1_score"],
                "Test F1": t_met["f1_score"],
                "Test Accuracy": t_met["accuracy"],
                "Benign FPR (%)": round(t_met["false_positive_rate_benign"] * 100, 2),
                "Unseen-Family Holdout Recall (%)": round(h_met["recall"] * 100, 2),
                "Latency (ms)": data["latency_ms_per_flow"],
                "Throughput (flows/s)": data["throughput_flows_sec"]
            }
            bin_rows.append(row)
            f.write(f"| **{mname}** | {row['Train Time (s)']} | {row['Val F1']} | {row['Test F1']} | {row['Test Accuracy']} | {row['Benign FPR (%)']}% | {row['Unseen-Family Holdout Recall (%)']}% | {row['Latency (ms)']} | {row['Throughput (flows/s)']:,} |\n")
        pd.DataFrame(bin_rows).to_csv(rep_path / "binary_model_comparison.csv", index=False)

        f.write("\n### Key Observations (Binary):\n")
        f.write("- RandomForest and LightGBM trade off benign false positives and malware recall; neither dominates both measures. The saved RandomForest has test F1 0.3609 and benign FPR 23.63%; LightGBM has test F1 0.3738, benign FPR 36.44%, and malware recall 58.37%.\n")
        f.write("- The reported inference throughput measures model prediction only. It does not include packet capture, flow aggregation, API delivery, or dashboard processing, and does not establish gigabit live-pipeline capacity.\n")
        f.write("- The binary test set is imbalanced: always predicting BENIGN would achieve about 80.9% accuracy, above each candidate's saved test accuracy. Accuracy alone is therefore misleading.\n\n")

        f.write("## 2. Experiment B: Multiclass Classification (5 Canonical Categories)\n\n")
        f.write("| Model | Train Time (s) | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Benign False Alarm (%) | Unseen-Family Holdout Macro-F1 | Latency (ms) | Model Throughput (flows/s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        multi_rows = []
        for mname, data in multi_metrics.items():
            t_met = data["test_metrics"]
            v_met = data["val_metrics"]
            h_met = data["zero_day_holdout_metrics"]
            row = {
                "Model": mname,
                "Train Time (s)": data["train_duration_sec"],
                "Val Macro-F1": v_met["macro_f1"],
                "Test Macro-F1": t_met["macro_f1"],
                "Test Accuracy": t_met["accuracy"],
                "Benign False Alarm (%)": round(t_met["false_alarm_rate_benign"] * 100, 2),
                "Unseen-Family Holdout Macro-F1": h_met["macro_f1"],
                "Latency (ms)": data["latency_ms_per_flow"],
                "Throughput (flows/s)": data["throughput_flows_sec"]
            }
            multi_rows.append(row)
            f.write(f"| **{mname}** | {row['Train Time (s)']} | {row['Val Macro-F1']} | {row['Test Macro-F1']} | {row['Test Accuracy']} | {row['Benign False Alarm (%)']}% | {row['Unseen-Family Holdout Macro-F1']} | {row['Latency (ms)']} | {row['Throughput (flows/s)']:,} |\n")
        pd.DataFrame(multi_rows).to_csv(rep_path / "multiclass_model_comparison.csv", index=False)

        f.write("\n### Key Observations (Multiclass):\n")
        f.write("- RandomForest scored the highest Test Macro-F1 (0.2552); ExtraTrees had higher accuracy (51.92%) and lower benign false alarm (41.10%). All models have weak per-class malware performance.\n")
        f.write("- Separating malware families into 4 specific categories (Adware vs Ransomware vs Scareware vs SMSmalware) purely based on Layer 3/4 flow statistics is challenging because multiple malware types reuse common advertising SDKs and HTTP C2 mechanisms.\n")

    # 2. Detailed Evaluation Report (reports/evaluation_report.md)
    eval_file = rep_path / "evaluation_report.md"
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write("# In-Depth Model Evaluation Report: Phase 1\n\n")
        f.write("## 1. Selected Best Models Summary\n\n")
        f.write("- **Binary Champion:** `RandomForest` (models/binary/best_model.joblib)\n")
        f.write("- **Multiclass Champion:** `RandomForest` (models/multiclass/best_model.joblib)\n\n")

        rf_bin_test = bin_metrics["RandomForest"]["test_metrics"]
        f.write("## 2. Binary Model Confusion Matrix (Test Set)\n\n")
        f.write("```\n")
        f.write(f"True Negative (Benign -> Benign):  {rf_bin_test['confusion_matrix']['TN']:,}\n")
        f.write(f"False Positive (Benign -> Malware): {rf_bin_test['confusion_matrix']['FP']:,}\n")
        f.write(f"False Negative (Malware -> Benign): {rf_bin_test['confusion_matrix']['FN']:,}\n")
        f.write(f"True Positive (Malware -> Malware):  {rf_bin_test['confusion_matrix']['TP']:,}\n")
        f.write("```\n\n")

        rf_multi_test = multi_metrics["RandomForest"]["test_metrics"]
        f.write("## 3. Multiclass Per-Class Metrics (Random Forest Test Set)\n\n")
        f.write("| Class Name | Precision | Recall | F1-Score | Support (Flows) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cname, p_data in rf_multi_test["per_class"].items():
            f.write(f"| **{cname}** | {p_data['precision']} | {p_data['recall']} | {p_data['f1_score']} | {p_data['support']:,} |\n")

        f.write("\n## 4. Selected Unseen-Family Holdout Evaluation\n\n")
        f.write("Four selected malware families were held out from training for an unseen-family evaluation. This finite holdout is not a prospective or general zero-day test:\n")
        f.write("- Adware: `Feiwo`\n- Ransomware: `Charger`\n- Scareware: `FakeAV`\n- SMSmalware: `Fakeinst`\n\n")
        f.write(f"- **Binary unseen-family holdout recall:** {bin_metrics['RandomForest']['zero_day_holdout_metrics']['recall']*100:.2f}%\n")
        f.write(f"- **Binary unseen-family holdout F1:** {bin_metrics['RandomForest']['zero_day_holdout_metrics']['f1_score']}\n")
        f.write(f"- **Multiclass unseen-family holdout macro-F1:** {multi_metrics['RandomForest']['zero_day_holdout_metrics']['macro_f1']}\n")
        f.write("- The binary family holdout contains malware flows only and no benign examples, so its zero false-positive count/rate is not evidence of benign discrimination.\n")
        f.write("- These stored metrics use estimator `predict` outputs. They do not measure the dashboard's separate 65% default confidence alert gate. No cross-validation or physical-device efficacy test is represented here.\n")

    # 3. Feature Importance Report (reports/feature_importance.md)
    fi_file = rep_path / "feature_importance.md"
    with open(fi_file, "w", encoding="utf-8") as f:
        f.write("# Feature Importance Analysis: Network Malware Detection\n\n")
        f.write("This report analyzes Gini feature importances extracted from the champion Random Forest models for both Binary and Multiclass tasks.\n\n")

        f.write("## 1. Top 20 Most Predictive Features (Binary Classification)\n\n")
        f.write("| Rank | Feature Name | Importance Weight | Network Engineering Meaning |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for idx, (fname, weight) in enumerate(list(bin_fi.items())[:20]):
            f.write(f"| {idx+1} | `{fname}` | {weight:.5f} | Key statistical discriminator for malware C2 |\n")

        f.write("\n## 2. Top 20 Most Predictive Features (Multiclass Classification)\n\n")
        f.write("| Rank | Feature Name | Importance Weight | Category Discrimination Role |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for idx, (fname, weight) in enumerate(list(multi_fi.items())[:20]):
            f.write(f"| {idx+1} | `{fname}` | {weight:.5f} | Distinguishes specific malware family behaviors |\n")

        f.write("\n## 3. Key Network Security Insights\n\n")
        f.write("- **Inter-Arrival Time (IAT) Features:** Features like `Flow IAT Mean`, `Fwd IAT Mean`, and `Flow IAT Min` rank among the most critical. Automated malware beaconing adheres to tighter periodic polling intervals compared to sporadic human browsing.\n")
        f.write("- **Packet Length Variations:** `Packet Length Std`, `Average Packet Size`, and `Fwd Packet Length Mean` provide heavy signal distinguishing short C2 heartbeat signals from bulky multimedia/content streaming.\n")
        f.write("- **TCP Window Dynamics:** `Init_Win_bytes_forward` and `Init_Win_bytes_backward` reflect socket layer implementation differences between custom malware sockets and Android OS system APIs.\n")

    print(f"[+] All evaluation reports and CSV exports compiled successfully in {rep_path}/")


if __name__ == "__main__":
    generate_evaluation_reports()
