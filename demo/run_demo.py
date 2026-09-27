"""Run a small, local prediction demonstration with the trained binary model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        help="Optional path to one CICIDS2017 machine-learning CSV file",
    )
    parser.add_argument("--rows", type=int, default=20)
    args = parser.parse_args()

    model_path = Path("outputs/models/binary_random_forest.joblib")
    metadata_path = Path("outputs/models/binary_model_metadata.json")
    if not model_path.exists():
        raise FileNotFoundError("Run scripts/run_experiments.ps1 before the demo.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    model = joblib.load(model_path)
    if args.csv:
        frame = pd.read_csv(args.csv, nrows=args.rows, low_memory=False)
        frame.columns = [" ".join(str(column).strip().split()) for column in frame.columns]
        actual = frame.pop("Label").astype(str).str.strip() if "Label" in frame else None
        source = Path(args.csv)
    else:
        sample_path = Path("data/processed/cicids2017_sample.pkl.gz")
        if not sample_path.exists():
            raise FileNotFoundError(
                "Provide a CICIDS2017 CSV or run the experiment to create the local sample."
            )
        sample = pd.read_pickle(sample_path, compression="gzip")
        benign_rows = max(1, args.rows // 2)
        attack_rows = max(1, args.rows - benign_rows)
        frame = pd.concat(
            [
                sample.loc[sample["Label"] == "Benign"].head(benign_rows),
                sample.loc[sample["Label"] != "Benign"].head(attack_rows),
            ],
            ignore_index=True,
        )
        actual = frame.pop("Label").astype(str).str.strip()
        frame = frame.drop(columns=["Source File"], errors="ignore")
        source = sample_path

    features = frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    if features.isna().any(axis=None):
        missing = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Input contains missing or non-numeric model features: {missing}")
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    output = pd.DataFrame(
        {
            "row": np.arange(1, len(features) + 1),
            "prediction": np.where(predictions == 1, "Attack", "Benign"),
            "attack_probability": probabilities.round(4),
        }
    )
    if actual is not None:
        output["actual"] = np.where(actual.str.upper() == "BENIGN", "Benign", "Attack")
    print(f"Source: {source}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
