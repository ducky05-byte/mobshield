"""Dataset Preprocessing and Group-Aware Splitting Pipeline.

Reads CICAndMal2017 CSVs, resolves duplicate column names, strips leaky
network identifiers, sanitizes infinite and missing values, creates leak-proof
group-aware splits (by APK sample), and prepares matrices for Experiment A (Binary)
and Experiment B (Multiclass).
"""

import os
import sys
import csv
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Raw column indices and names
IDENTIFIER_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Timestamp"]
DUPLICATE_COL_ORIG = "Fwd Header Length"
DUPLICATE_COL_RENAMED = "Fwd Header Length.1"

# Binary Label Mapping
BINARY_LABEL_MAP = {
    "BENIGN": 0,
    "MALWARE_ASSOCIATED": 1
}

# Multiclass Label Mapping
MULTICLASS_LABEL_MAP = {
    "BENIGN": 0,
    "ADWARE": 1,
    "RANSOMWARE": 2,
    "SCAREWARE": 3,
    "SMSMALWARE": 4
}
MULTICLASS_REVERSE_MAP = {v: k for k, v in MULTICLASS_LABEL_MAP.items()}


def map_binary_label(raw_label: str) -> Optional[int]:
    """Maps raw label string to binary target: 0 (BENIGN) or 1 (MALWARE_ASSOCIATED)."""
    lbl = raw_label.strip()
    if not lbl or lbl == "[BLANK]":
        return None  # Drop blank
    if lbl == "BENIGN":
        return 0
    # Any malware family or generic malware label
    return 1


def map_multiclass_label(raw_label: str, folder_category: str = "") -> Optional[int]:
    """Maps raw label string to multiclass target (0..4).
    
    Excludes ambiguous generic 'MALWARE' strings if family is unknown to maintain ground-truth integrity.
    """
    lbl = raw_label.strip()
    if not lbl or lbl == "[BLANK]":
        return None
    if lbl == "BENIGN":
        return 0
    
    upper = lbl.upper()
    if upper.startswith("ADWARE"):
        return 1
    elif upper.startswith("RANSOMWARE"):
        return 2
    elif upper.startswith("SCAREWARE"):
        return 3
    elif upper.startswith("SMSMALWARE"):
        return 4
    elif upper == "MALWARE":
        # Ambiguous label without family specification:
        # If folder context is available and matches a known category, document & map or exclude.
        # To maintain strict ground truth, exclude ambiguous generic label from multiclass.
        return None
    return None


class NetworkFlowPreprocessor:
    """Production-grade preprocessor that fits strictly on training data."""
    __module__ = "src.prepare_dataset"

    def __init__(self, variance_threshold: float = 0.0):
        self.variance_threshold = variance_threshold
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.dropped_identifier_cols_: List[str] = []
        self.dropped_constant_cols_: List[str] = []
        self.medians_: Dict[str, float] = {}
        self.clip_max_: Dict[str, float] = {}
        self.scaler = StandardScaler()
        self.is_fitted: bool = False

    def sanitize_column_names(self, columns: List[str]) -> List[str]:
        """Resolves duplicate column names (specifically Fwd Header Length)."""
        counts = Counter()
        sanitized = []
        for col in columns:
            clean_col = col.strip()
            counts[clean_col] += 1
            if counts[clean_col] > 1:
                sanitized.append(f"{clean_col}.{counts[clean_col]-1}")
            else:
                sanitized.append(clean_col)
        return sanitized

    def fit(self, df: pd.DataFrame) -> "NetworkFlowPreprocessor":
        """Fits imputer medians, constant feature detector, and scaler strictly on training df."""
        df_clean = df.copy()
        
        # 1. Clean headers
        df_clean.columns = self.sanitize_column_names(df_clean.columns.tolist())
        self.feature_names_in_ = df_clean.columns.tolist()

        # 2. Identify and remove identifiers
        to_drop = [c for c in IDENTIFIER_COLS if c in df_clean.columns]
        if "Label" in df_clean.columns:
            to_drop.append("Label")
        if "sample_id" in df_clean.columns:
            to_drop.append("sample_id")
        if "family" in df_clean.columns:
            to_drop.append("family")
        if "category" in df_clean.columns:
            to_drop.append("category")
        self.dropped_identifier_cols_ = to_drop
        
        features_df = df_clean.drop(columns=to_drop, errors="ignore")

        # 3. Clean numeric anomalies (inf, nan)
        # Convert all to numeric
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
            # Replace inf with nan for median calculation
            features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
            
            # Compute 99.9th percentile cap for extreme outliers
            valid_vals = features_df[col].dropna()
            if len(valid_vals) > 0:
                q99 = valid_vals.quantile(0.999)
                self.clip_max_[col] = float(q99) if q99 > 0 else 1e9
                med = float(valid_vals.median())
                self.medians_[col] = med if not np.isnan(med) else 0.0
            else:
                self.clip_max_[col] = 1e9
                self.medians_[col] = 0.0

        # Impute medians
        features_df = features_df.fillna(self.medians_)

        # 4. Detect zero-variance / constant features
        variances = features_df.var()
        constant_cols = variances[variances <= self.variance_threshold].index.tolist()
        self.dropped_constant_cols_ = constant_cols
        
        retained_df = features_df.drop(columns=constant_cols, errors="ignore")
        self.feature_names_out_ = retained_df.columns.tolist()

        # 5. Fit scaler
        self.scaler.fit(retained_df.values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms features using fitted statistics."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform!")
        
        df_clean = df.copy()
        df_clean.columns = self.sanitize_column_names(df_clean.columns.tolist())
        
        # Drop identifiers
        features_df = df_clean.drop(columns=self.dropped_identifier_cols_, errors="ignore")

        # Apply column selection to match training features_in (before constant drop)
        processed_cols = {}
        for col in self.feature_names_out_:
            if col in features_df.columns:
                series = pd.to_numeric(features_df[col], errors="coerce")
                # Replace inf
                series = series.replace([np.inf, -np.inf], np.nan)
                # Clip extreme values
                if col in self.clip_max_:
                    series = series.clip(upper=self.clip_max_[col])
                # Impute median
                med = self.medians_.get(col, 0.0)
                series = series.fillna(med)
                processed_cols[col] = series.values
            else:
                # Missing feature in live packet -> fill with median
                med = self.medians_.get(col, 0.0)
                processed_cols[col] = np.full(len(features_df), med)

        X_mat = np.column_stack([processed_cols[col] for col in self.feature_names_out_])
        # Scale
        X_scaled = self.scaler.transform(X_mat)
        return X_scaled

    def transform_single(self, flow_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms a single live flow dictionary for real-time inference."""
        df = pd.DataFrame([flow_dict])
        return self.transform(df)

    def save(self, filepath: str):
        """Saves fitted preprocessor state."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "NetworkFlowPreprocessor":
        """Loads fitted preprocessor state."""
        return joblib.load(filepath)


def build_apk_sample_manifest(csv_dir: Path) -> List[Dict[str, Any]]:
    """Builds a verified manifest of all 2,126 APK capture sessions."""
    samples = []
    for fpath in csv_dir.rglob("*.csv"):
        rel = fpath.relative_to(csv_dir)
        top_cat = rel.parts[0].replace("-CSVs", "")
        
        parts = fpath.stem.replace(".pcap_ISCX", "").split("-")
        if top_cat == "Benign":
            app_id = parts[-1] if len(parts) >= 4 else fpath.stem
            family = "BENIGN"
        else:
            family = fpath.parent.name if len(rel.parts) > 2 else (parts[2] if len(parts) >= 3 else "unknown")
            app_id = parts[-1] if len(parts) >= 5 else fpath.stem
        
        samples.append({
            "path": str(fpath),
            "sample_id": app_id,
            "category": top_cat,
            "family": family,
            "filename": fpath.name
        })
    return samples


def split_samples_group_aware(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    zero_day_holdout_families: Optional[Dict[str, str]] = None,
    random_seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """Partitions APK capture sessions group-wise to ensure 0 leakage across splits.
    
    Optionally holds out designated zero-day families entirely for unseen evaluation.
    """
    rng = random.Random(random_seed)
    
    if zero_day_holdout_families is None:
        # Default: hold out 1 unseen family per malware category
        zero_day_holdout_families = {
            "Adware": "Feiwo",
            "Ransomware": "Charger",
            "Scareware": "FakeAV",
            "SMSmalware": "Fakeinst"
        }

    holdout_samples = []
    eligible_samples = []

    for s in samples:
        cat = s["category"]
        fam = s["family"]
        if cat in zero_day_holdout_families and fam == zero_day_holdout_families[cat]:
            holdout_samples.append(s)
        else:
            eligible_samples.append(s)

    # Group eligible samples by category and partition by UNIQUE sample_id
    by_category_samples = defaultdict(list)
    by_category_unique_ids = defaultdict(set)
    for s in eligible_samples:
        by_category_samples[s["category"]].append(s)
        by_category_unique_ids[s["category"]].add(s["sample_id"])

    train_samples = []
    val_samples = []
    test_samples = []

    for cat, unique_ids_set in by_category_unique_ids.items():
        unique_ids = sorted(list(unique_ids_set))
        rng.shuffle(unique_ids)
        n = len(unique_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_id_set = set(unique_ids[:n_train])
        val_id_set = set(unique_ids[n_train:n_train + n_val])
        test_id_set = set(unique_ids[n_train + n_val:])

        for s in by_category_samples[cat]:
            sid = s["sample_id"]
            if sid in train_id_set:
                train_samples.append(s)
            elif sid in val_id_set:
                val_samples.append(s)
            else:
                test_samples.append(s)

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "zero_day_holdout": holdout_samples
    }


def load_split_dataframe(
    sample_list: List[Dict[str, Any]],
    max_flows_per_sample: int = 150,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Loads a balanced, representative flow sample across given APK sessions."""
    rng = random.Random(random_seed)
    all_rows = []
    stats = {"read": 0, "retained": 0, "dropped_blank": 0, "dropped_corrupt": 0}

    for sample in sample_list:
        fpath = sample["path"]
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            try:
                hdr = [h.strip() for h in next(reader)]
            except StopIteration:
                continue

            file_rows = []
            label_idx = hdr.index("Label") if "Label" in hdr else -1
            
            for row in reader:
                stats["read"] += 1
                if not row or len(row) < len(hdr):
                    stats["dropped_corrupt"] += 1
                    continue
                
                raw_lbl = row[label_idx].strip() if label_idx != -1 else ""
                if not raw_lbl:
                    stats["dropped_blank"] += 1
                    continue
                
                file_rows.append(row)

            # Subsample flows if file is very large to balance APK representation
            if len(file_rows) > max_flows_per_sample:
                selected_rows = rng.sample(file_rows, max_flows_per_sample)
            else:
                selected_rows = file_rows

            stats["retained"] += len(selected_rows)

            # Build dicts
            for r in selected_rows:
                row_dict = {hdr[i]: r[i] for i in range(len(hdr))}
                row_dict["sample_id"] = sample["sample_id"]
                row_dict["category"] = sample["category"]
                row_dict["family"] = sample["family"]
                all_rows.append(row_dict)

    df = pd.DataFrame(all_rows)
    return df, stats


def prepare_phase1_datasets(
    dataset_root: str,
    output_dir: str = "data/processed",
    max_flows_per_file: int = 150,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Runs end-to-end dataset preparation and writes split files and preprocessor."""
    root = Path(dataset_root)
    csv_dir = root / "CSVs" if (root / "CSVs").exists() else root
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[*] Building APK sample manifest from {csv_dir}...")
    manifest = build_apk_sample_manifest(csv_dir)
    print(f"[+] Total APK captures cataloged: {len(manifest):,}")

    print("[*] Creating leak-proof group-aware train/val/test splits...")
    splits = split_samples_group_aware(manifest, random_seed=random_seed)
    print(f"    Train APKs: {len(splits['train']):,}")
    print(f"    Val APKs:   {len(splits['val']):,}")
    print(f"    Test APKs:  {len(splits['test']):,}")
    print(f"    Zero-Day Holdout APKs: {len(splits['zero_day_holdout']):,}")

    # Load dataframes for each split
    print(f"[*] Loading flow records (max {max_flows_per_file} flows per APK session)...")
    train_df, s_train = load_split_dataframe(splits["train"], max_flows_per_sample=max_flows_per_file, random_seed=random_seed)
    val_df, s_val = load_split_dataframe(splits["val"], max_flows_per_sample=max_flows_per_file, random_seed=random_seed+1)
    test_df, s_test = load_split_dataframe(splits["test"], max_flows_per_sample=max_flows_per_file, random_seed=random_seed+2)
    holdout_df, s_hold = load_split_dataframe(splits["zero_day_holdout"], max_flows_per_sample=max_flows_per_file, random_seed=random_seed+3)

    print(f"[+] Loaded Train Flows:   {len(train_df):,}")
    print(f"[+] Loaded Val Flows:     {len(val_df):,}")
    print(f"[+] Loaded Test Flows:    {len(test_df):,}")
    print(f"[+] Loaded Holdout Flows: {len(holdout_df):,}")

    # Fit preprocessor strictly on train_df
    print("[*] Fitting NetworkFlowPreprocessor on training data...")
    preprocessor = NetworkFlowPreprocessor(variance_threshold=0.0)
    preprocessor.fit(train_df)

    print(f"[+] Preprocessor retained {len(preprocessor.feature_names_out_)} features.")
    print(f"    Dropped Identifiers: {preprocessor.dropped_identifier_cols_}")
    print(f"    Dropped Zero-Variance: {preprocessor.dropped_constant_cols_}")

    # Save preprocessor artifact
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    prep_path = models_dir / "preprocessor.joblib"
    preprocessor.save(str(prep_path))
    print(f"[+] Saved fitted preprocessor to {prep_path}")

    # Transform feature matrices
    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)
    X_holdout = preprocessor.transform(holdout_df)

    # Generate Binary Targets (Experiment A)
    y_train_bin = train_df["Label"].apply(map_binary_label).values
    y_val_bin = val_df["Label"].apply(map_binary_label).values
    y_test_bin = test_df["Label"].apply(map_binary_label).values
    y_holdout_bin = holdout_df["Label"].apply(map_binary_label).values

    # Generate Multiclass Targets (Experiment B)
    # Mask out ambiguous 'MALWARE' labels if any
    y_train_multi = train_df["Label"].apply(map_multiclass_label).values
    y_val_multi = val_df["Label"].apply(map_multiclass_label).values
    y_test_multi = test_df["Label"].apply(map_multiclass_label).values
    y_holdout_multi = holdout_df["Label"].apply(map_multiclass_label).values

    # Save arrays to disk in compressed NPZ for high performance
    npz_path = out / "processed_splits.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train_bin=y_train_bin, y_train_multi=y_train_multi,
        X_val=X_val, y_val_bin=y_val_bin, y_val_multi=y_val_multi,
        X_test=X_test, y_test_bin=y_test_bin, y_test_multi=y_test_multi,
        X_holdout=X_holdout, y_holdout_bin=y_holdout_bin, y_holdout_multi=y_holdout_multi,
        feature_names=preprocessor.feature_names_out_
    )
    print(f"[+] Processed matrices saved to {npz_path}")

    # Save split metadata & manifest
    split_meta = {
        "features_in_count": len(preprocessor.feature_names_in_),
        "features_retained_count": len(preprocessor.feature_names_out_),
        "features_names": preprocessor.feature_names_out_,
        "dropped_identifiers": preprocessor.dropped_identifier_cols_,
        "dropped_constant_features": preprocessor.dropped_constant_cols_,
        "counts": {
            "train_flows": len(train_df),
            "val_flows": len(val_df),
            "test_flows": len(test_df),
            "holdout_flows": len(holdout_df)
        },
        "binary_label_map": BINARY_LABEL_MAP,
        "multiclass_label_map": MULTICLASS_LABEL_MAP,
        "sample_counts": {
            "train_apks": len(splits["train"]),
            "val_apks": len(splits["val"]),
            "test_apks": len(splits["test"]),
            "holdout_apks": len(splits["zero_day_holdout"])
        }
    }
    with open(out / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    with open(models_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor.feature_names_out_, f, indent=2)

    with open(models_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"binary": BINARY_LABEL_MAP, "multiclass": MULTICLASS_LABEL_MAP}, f, indent=2)

    print(f"[+] Preprocessing specification written to {out / 'split_manifest.json'}")
    return split_meta


if __name__ == "__main__":
    dataset_root = r"c:\Users\Varun\Desktop\AML"
    prepare_phase1_datasets(dataset_root)
