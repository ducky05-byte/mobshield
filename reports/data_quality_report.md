# Data Quality & Preprocessing Specification Report

**Audit Timestamp:** 2026-10-13T16:58:37.309656  
**Sampled Records Audited:** 116,896 flows  

## 1. Missing and Infinite Values

- **Missing / Null Columns:**
  - `ECE Flag Count`: 2 instances (0.002%)
  - `Down/Up Ratio`: 2 instances (0.002%)
  - `Average Packet Size`: 3 instances (0.003%)
  - `Avg Fwd Segment Size`: 3 instances (0.003%)
  - `Avg Bwd Segment Size`: 3 instances (0.003%)
  - `Fwd Header Length`: 3 instances (0.003%)
  - `Fwd Avg Bytes/Bulk`: 3 instances (0.003%)
  - `Fwd Avg Packets/Bulk`: 3 instances (0.003%)
  - `Fwd Avg Bulk Rate`: 3 instances (0.003%)
  - `Bwd Avg Bytes/Bulk`: 3 instances (0.003%)
  - `Bwd Avg Packets/Bulk`: 3 instances (0.003%)
  - `Bwd Avg Bulk Rate`: 3 instances (0.003%)
  - `Subflow Fwd Packets`: 3 instances (0.003%)
  - `Subflow Fwd Bytes`: 3 instances (0.003%)
  - `Subflow Bwd Packets`: 3 instances (0.003%)
  - `Subflow Bwd Bytes`: 3 instances (0.003%)
  - `Init_Win_bytes_forward`: 3 instances (0.003%)
  - `Init_Win_bytes_backward`: 3 instances (0.003%)
  - `act_data_pkt_fwd`: 3 instances (0.003%)
  - `min_seg_size_forward`: 3 instances (0.003%)
  - `Active Mean`: 3 instances (0.003%)
  - `Active Std`: 3 instances (0.003%)
  - `Active Max`: 3 instances (0.003%)
  - `Active Min`: 3 instances (0.003%)
  - `Idle Mean`: 3 instances (0.003%)
  - `Idle Std`: 3 instances (0.003%)
  - `Idle Max`: 3 instances (0.003%)
  - `Idle Min`: 3 instances (0.003%)
  - `Label`: 3 instances (0.003%)
  - `Packet Length Variance`: 1 instances (0.001%)
  - `FIN Flag Count`: 1 instances (0.001%)
  - `SYN Flag Count`: 1 instances (0.001%)
  - `RST Flag Count`: 1 instances (0.001%)
  - `PSH Flag Count`: 1 instances (0.001%)
  - `ACK Flag Count`: 1 instances (0.001%)
  - `URG Flag Count`: 1 instances (0.001%)
  - `CWE Flag Count`: 1 instances (0.001%)
  - `Flow ID`: 1 instances (0.001%)
- **Infinite (`inf` / `-inf`) Columns:**
  - Handled safely by runtime numerical cap / median imputation.

## 2. Zero-Variance & Constant Features

The following features exhibit zero or near-zero variance across mobile flows and must be eliminated:


## 3. Mandatory Leakage Prevention Drops

The following identifier columns must be stripped prior to model ingestion:

- `Flow ID`: Combines IP 5-tuple; leaks exact testbed client and server identities.
- `Source IP`: Contains testbed private client IP (`10.42.0.x`).
- `Destination IP`: Memorizes specific testbed C2 and public servers.
- `Source Port`: Ephemeral client ports specific to local OS session.
- `Destination Port`: Dropped to prevent port memorization unless specifically isolated.
- `Timestamp`: Date and time of capture in 2017; induces temporal overfitting.

## 4. Preprocessing Protocol Specification

1. **Header Normalization:** Duplicate column 61 is renamed to `Fwd Header Length.1`.
2. **Identifier Stripping:** Remove the 6 leakage identifier columns.
3. **Label Sanitation:** Prune 5 blank records; map verified family names to canonical classes.
4. **Imputation & Infinite Capping:** Inf values capped to 99.9th percentile or zero; NaNs imputed with training set median.
5. **Variance Threshold:** Remove invariant bulk rate features.
