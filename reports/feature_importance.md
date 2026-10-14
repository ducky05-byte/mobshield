# Feature Importance Analysis: Network Malware Detection

This report analyzes Gini feature importances extracted from the champion Random Forest models for both Binary and Multiclass tasks.

## 1. Top 20 Most Predictive Features (Binary Classification)

| Rank | Feature Name | Importance Weight | Network Engineering Meaning |
| :--- | :--- | :--- | :--- |
| 1 | `Flow IAT Min` | 0.05045 | Key statistical discriminator for malware C2 |
| 2 | `Init_Win_bytes_forward` | 0.04875 | Key statistical discriminator for malware C2 |
| 3 | `Flow IAT Mean` | 0.04612 | Key statistical discriminator for malware C2 |
| 4 | `Flow IAT Max` | 0.04593 | Key statistical discriminator for malware C2 |
| 5 | `Flow Duration` | 0.04370 | Key statistical discriminator for malware C2 |
| 6 | `Flow Packets/s` | 0.03692 | Key statistical discriminator for malware C2 |
| 7 | `Fwd Packets/s` | 0.03680 | Key statistical discriminator for malware C2 |
| 8 | `Fwd IAT Min` | 0.03188 | Key statistical discriminator for malware C2 |
| 9 | `Init_Win_bytes_backward` | 0.03165 | Key statistical discriminator for malware C2 |
| 10 | `Bwd Packets/s` | 0.03149 | Key statistical discriminator for malware C2 |
| 11 | `Fwd IAT Mean` | 0.02915 | Key statistical discriminator for malware C2 |
| 12 | `Flow Bytes/s` | 0.02806 | Key statistical discriminator for malware C2 |
| 13 | `Fwd IAT Max` | 0.02762 | Key statistical discriminator for malware C2 |
| 14 | `Fwd IAT Total` | 0.02740 | Key statistical discriminator for malware C2 |
| 15 | `Flow IAT Std` | 0.02332 | Key statistical discriminator for malware C2 |
| 16 | `Fwd IAT Std` | 0.01875 | Key statistical discriminator for malware C2 |
| 17 | `Packet Length Std` | 0.01835 | Key statistical discriminator for malware C2 |
| 18 | `Packet Length Mean` | 0.01767 | Key statistical discriminator for malware C2 |
| 19 | `Packet Length Variance` | 0.01745 | Key statistical discriminator for malware C2 |
| 20 | `Bwd IAT Min` | 0.01719 | Key statistical discriminator for malware C2 |

## 2. Top 20 Most Predictive Features (Multiclass Classification)

| Rank | Feature Name | Importance Weight | Category Discrimination Role |
| :--- | :--- | :--- | :--- |
| 1 | `Flow IAT Min` | 0.05453 | Distinguishes specific malware family behaviors |
| 2 | `Init_Win_bytes_forward` | 0.05191 | Distinguishes specific malware family behaviors |
| 3 | `Flow IAT Mean` | 0.05039 | Distinguishes specific malware family behaviors |
| 4 | `Flow IAT Max` | 0.04930 | Distinguishes specific malware family behaviors |
| 5 | `Flow Duration` | 0.04872 | Distinguishes specific malware family behaviors |
| 6 | `Flow Packets/s` | 0.04159 | Distinguishes specific malware family behaviors |
| 7 | `Fwd Packets/s` | 0.04137 | Distinguishes specific malware family behaviors |
| 8 | `Init_Win_bytes_backward` | 0.03564 | Distinguishes specific malware family behaviors |
| 9 | `Bwd Packets/s` | 0.03513 | Distinguishes specific malware family behaviors |
| 10 | `Flow Bytes/s` | 0.03063 | Distinguishes specific malware family behaviors |
| 11 | `Fwd IAT Min` | 0.03042 | Distinguishes specific malware family behaviors |
| 12 | `Fwd IAT Mean` | 0.02641 | Distinguishes specific malware family behaviors |
| 13 | `Fwd IAT Max` | 0.02589 | Distinguishes specific malware family behaviors |
| 14 | `Fwd IAT Total` | 0.02572 | Distinguishes specific malware family behaviors |
| 15 | `Flow IAT Std` | 0.02154 | Distinguishes specific malware family behaviors |
| 16 | `Packet Length Variance` | 0.01679 | Distinguishes specific malware family behaviors |
| 17 | `Average Packet Size` | 0.01656 | Distinguishes specific malware family behaviors |
| 18 | `Packet Length Std` | 0.01655 | Distinguishes specific malware family behaviors |
| 19 | `Fwd IAT Std` | 0.01655 | Distinguishes specific malware family behaviors |
| 20 | `Packet Length Mean` | 0.01621 | Distinguishes specific malware family behaviors |

## 3. Key Network Security Insights

- **Inter-Arrival Time (IAT) Features:** Features like `Flow IAT Mean`, `Fwd IAT Mean`, and `Flow IAT Min` rank among the most critical. Automated malware beaconing adheres to tighter periodic polling intervals compared to sporadic human browsing.
- **Packet Length Variations:** `Packet Length Std`, `Average Packet Size`, and `Fwd Packet Length Mean` provide heavy signal distinguishing short C2 heartbeat signals from bulky multimedia/content streaming.
- **TCP Window Dynamics:** `Init_Win_bytes_forward` and `Init_Win_bytes_backward` reflect socket layer implementation differences between custom malware sockets and Android OS system APIs.
