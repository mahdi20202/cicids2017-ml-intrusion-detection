"""Independently validate generated dataset and metric artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.nids_experiment import LABEL_COLUMN, SOURCE_COLUMN


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    table_dir = Path("outputs/tables")
    sample = pd.read_pickle("data/processed/cicids2017_sample.pkl.gz", compression="gzip")
    profile = json.loads((table_dir / "dataset_profile.json").read_text(encoding="utf-8"))
    design = json.loads((table_dir / "experiment_design.json").read_text(encoding="utf-8"))
    feature_columns = [
        column for column in sample.columns if column not in {LABEL_COLUMN, SOURCE_COLUMN}
    ]

    checks: dict[str, object] = {}
    checks["sample_row_count_matches_profile"] = len(sample) == profile["sample_rows_after_deduplication"]
    checks["profile_rows_reconcile"] = (
        profile["raw_rows"] - profile["invalid_rows_removed"] == profile["valid_rows"]
    )
    checks["sample_has_no_missing_features"] = not sample[feature_columns].isna().any().any()
    checks["sample_has_only_finite_features"] = bool(
        np.isfinite(sample[feature_columns].to_numpy(dtype=np.float32)).all()
    )
    checks["sample_has_no_exact_feature_label_duplicates"] = not sample.duplicated(
        subset=feature_columns + [LABEL_COLUMN]
    ).any()
    observed_counts = {
        key: int(value)
        for key, value in sample[LABEL_COLUMN].value_counts().sort_index().items()
    }
    checks["sample_class_counts_match_profile"] = observed_counts == profile["sample_label_counts"]

    binary = pd.read_csv(table_dir / "binary_model_metrics.csv")
    binary_checks: dict[str, bool] = {}
    for row in binary.to_dict(orient="records"):
        total = int(row["tn"] + row["fp"] + row["fn"] + row["tp"])
        accuracy = (row["tn"] + row["tp"]) / total
        fpr = row["fp"] / (row["fp"] + row["tn"])
        fnr = row["fn"] / (row["fn"] + row["tp"])
        binary_checks[row["model"]] = (
            total == design["binary_test_rows"]
            and close(accuracy, row["accuracy"])
            and close(fpr, row["false_positive_rate"])
            and close(fnr, row["false_negative_rate"])
        )
    checks["binary_metrics_recompute_from_confusion_counts"] = binary_checks

    multiclass = pd.read_csv(table_dir / "multiclass_model_metrics.csv")
    day_shift = pd.read_csv(table_dir / "day_shift_binary_metrics.csv")
    checks["all_reported_metrics_are_finite"] = all(
        np.isfinite(frame.select_dtypes("number").to_numpy()).all()
        for frame in (binary, multiclass, day_shift)
    )
    checks["all_score_metrics_in_unit_interval"] = bool(
        binary[
            [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "false_negative_rate",
                "roc_auc",
            ]
        ].apply(lambda column: column.between(0, 1).all()).all()
    )
    required_figures = [
        "class_distribution_full.png",
        "binary_model_comparison.png",
        "binary_roc_curves.png",
        "multiclass_model_comparison.png",
        "multiclass_confusion_random_forest.png",
        "day_shift_model_comparison.png",
        "random_forest_feature_importance.png",
    ]
    checks["required_figures_present"] = all(
        (Path("outputs/figures") / name).stat().st_size > 10_000 for name in required_figures
    )

    def flatten(values: object) -> list[bool]:
        if isinstance(values, dict):
            result: list[bool] = []
            for nested in values.values():
                result.extend(flatten(nested))
            return result
        return [bool(values)]

    checks["all_checks_pass"] = all(flatten(checks))
    (table_dir / "validation_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not checks["all_checks_pass"]:
        raise AssertionError(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
