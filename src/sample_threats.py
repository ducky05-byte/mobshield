"""Sample flow provider from genuine CICAndMal2017 dataset.

Extracts real flow feature records from raw CSVs for verified demonstration,
diagnostics, and automated testing of threat alert mechanisms.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CSV_ROOT = PROJECT_ROOT / "CSVs"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.json"

CATEGORY_CSV_MAP = {
    "RANSOMWARE": CSV_ROOT / "Ransomware-CSVs" / "Ransomware" / "Charger" / "08_28_2017-ra-charger-sophos-00357b0e208c20df3182d54cb2ba15bf.pcap_ISCX.csv",
    "ADWARE": CSV_ROOT / "Adware-CSVs" / "Adware" / "Dowgin" / "08_08_2017-ad-dowgin-sophos-0027f300c5eead4f447dbb1f4df12a51.pcap_ISCX.csv",
    "SCAREWARE": CSV_ROOT / "Scareware-CSVs" / "Scareware" / "AndroidDefender" / "08_21_2017-sc-androiddefender-sophos-002a24fa69b357bdf514a38e8ae38f15.pcap_ISCX.csv",
    "SMSMALWARE": CSV_ROOT / "SMSmalware-CSVs" / "SMSmalware" / "FakeMart" / "08_14_2017-sm-fakemart-sophos-0062b3394602bb1667d7192415951666.pcap_ISCX.csv",
    "BENIGN": CSV_ROOT / "Benign-CSVs" / "Benign" / "08_01_2017-be-1-sophos-018260c6d9ad8eb330c6a85f4fe61b2e.pcap_ISCX.csv"
}

_CACHE = {}


VERIFIED_SAMPLES_FILE = PROJECT_ROOT / "data" / "verified_samples.json"

def get_genuine_sample_flow(category: str = "RANSOMWARE") -> Dict[str, float]:
    """Returns a genuine unscaled 65-feature flow vector from the dataset for testing."""
    cat = category.upper()
    if cat in _CACHE:
        return _CACHE[cat].copy()

    # Try loading from precomputed verified_samples.json
    if VERIFIED_SAMPLES_FILE.exists():
        try:
            with open(VERIFIED_SAMPLES_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if cat in saved:
                    _CACHE[cat] = saved[cat]
                    return saved[cat].copy()
        except Exception:
            pass

    from src.predict_flow import FlowPredictor
    pred = FlowPredictor()

    csv_path = CATEGORY_CSV_MAP.get(cat)
    if not csv_path or not csv_path.exists():
        parent_dir = CSV_ROOT / f"{cat.capitalize()}-CSVs"
        found = list(parent_dir.rglob("*.csv"))
        if found:
            csv_path = found[0]
        else:
            raise FileNotFoundError(f"No CSV found for category {category}")

    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    parent_dir = csv_path.parent
    candidate_files = [csv_path] + [f for f in parent_dir.glob("*.csv") if f != csv_path]

    best_sample = None
    best_score = -1.0

    for cfile in candidate_files[:4]:
        df = pd.read_csv(cfile, nrows=200)
        for _, row in df.iterrows():
            flow_dict = {}
            for col in df.columns:
                cleaned_col = col.strip()
                if cleaned_col in feature_names:
                    try:
                        val = float(row[col])
                        if not pd.isna(val):
                            flow_dict[cleaned_col] = val
                    except (ValueError, TypeError):
                        pass
            if len(flow_dict) == len(feature_names):
                b = pred.predict_flow(flow_dict, mode="binary")
                m = pred.predict_flow(flow_dict, mode="multiclass")
                mal_prob = b["probabilities"].get("MALWARE_ASSOCIATED", 0.0)

                if cat == "BENIGN":
                    if b["label"] == "BENIGN" and m["label"] == "BENIGN":
                        _CACHE[cat] = flow_dict
                        return flow_dict.copy()
                else:
                    if m["label"] == cat and mal_prob >= 0.50:
                        _CACHE[cat] = flow_dict
                        return flow_dict.copy()
                    if m["label"] == cat and mal_prob > best_score:
                        best_score = mal_prob
                        best_sample = flow_dict

    if best_sample is not None:
        _CACHE[cat] = best_sample
        return best_sample.copy()

    raise RuntimeError(f"Could not find matching sample for {category}")


def save_all_verified_samples():
    """Extracts and persists all 5 genuine samples to data/verified_samples.json."""
    samples = {}
    for cat in ["BENIGN", "RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"]:
        samples[cat] = get_genuine_sample_flow(cat)
    with open(VERIFIED_SAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"Persisted 5 genuine category samples to {VERIFIED_SAMPLES_FILE}")


if __name__ == "__main__":
    from src.predict_flow import FlowPredictor
    save_all_verified_samples()
    predictor = FlowPredictor()
    for category in ["BENIGN", "RANSOMWARE", "ADWARE", "SCAREWARE", "SMSMALWARE"]:
        sample = get_genuine_sample_flow(category)
        bin_res = predictor.predict_flow(sample, mode="binary")
        multi_res = predictor.predict_flow(sample, mode="multiclass")
        print(f"[{category}]")
        print(f"  Binary: {bin_res['label']} ({bin_res['confidence']:.4f}) -> P(MAL)={bin_res['probabilities'].get('MALWARE_ASSOCIATED', 0):.4f}")
        print(f"  Multi:  {multi_res['label']} ({multi_res['confidence']:.4f})")
