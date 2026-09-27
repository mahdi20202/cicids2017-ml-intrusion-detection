"""Resume bounded multiclass and day-shift experiments from the cleaned sample."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.nids_experiment import (
    LABEL_COLUMN,
    RANDOM_SEED,
    SOURCE_COLUMN,
    build_models,
    configure_plots,
    prepare_multiclass_sample,
    save_feature_importance,
    save_confusion_matrix,
    save_metric_bars,
    save_roc_curves,
    train_and_evaluate,
)


def main() -> None:
    configure_plots()
    sample_path = Path("data/processed/cicids2017_sample.pkl.gz")
    if not sample_path.exists():
        raise FileNotFoundError("The cleaned sample is missing. Run src.nids_experiment first.")

    figure_dir = Path("outputs/figures")
    table_dir = Path("outputs/tables")
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_pickle(sample_path, compression="gzip")
    feature_columns = [
        column for column in sample.columns if column not in {LABEL_COLUMN, SOURCE_COLUMN}
    ]

    binary_target = (sample[LABEL_COLUMN] != "Benign").astype(np.int8).to_numpy()
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        sample[feature_columns],
        binary_target,
        test_size=0.30,
        stratify=binary_target,
        random_state=RANDOM_SEED,
    )
    x_validation, x_test, _y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.50,
        stratify=y_holdout,
        random_state=RANDOM_SEED,
    )
    binary_results, binary_models, binary_predictions, binary_scores = train_and_evaluate(
        build_models(RANDOM_SEED, include_unweighted=True),
        x_train,
        y_train,
        x_test,
        y_test,
        task="binary",
    )
    binary_results.to_csv(table_dir / "binary_model_metrics.csv", index=False)
    save_metric_bars(
        binary_results,
        ["balanced_accuracy", "precision", "recall", "f1", "mcc"],
        "Binary intrusion-detection performance",
        figure_dir / "binary_model_comparison.png",
    )
    save_roc_curves(y_test, binary_scores, figure_dir / "binary_roc_curves.png")
    for model_name, predicted in binary_predictions.items():
        safe_name = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        save_confusion_matrix(
            y_test,
            predicted,
            [0, 1],
            f"{model_name}: binary confusion matrix",
            figure_dir / f"binary_confusion_{safe_name}.png",
            normalize=True,
        )
    forest = binary_models["Random Forest"]
    importance = save_feature_importance(
        forest, feature_columns, figure_dir / "random_forest_feature_importance.png"
    )
    importance.to_csv(table_dir / "random_forest_feature_importance.csv", index=False)
    joblib.dump(forest, Path("outputs/models/binary_random_forest.joblib"), compress=3)

    multiclass = prepare_multiclass_sample(sample, RANDOM_SEED)
    multiclass_target = multiclass[LABEL_COLUMN].to_numpy()
    mc_train, mc_test, mc_y_train, mc_y_test = train_test_split(
        multiclass[feature_columns],
        multiclass_target,
        test_size=0.20,
        stratify=multiclass_target,
        random_state=RANDOM_SEED,
    )
    multiclass_models = build_models(RANDOM_SEED, include_unweighted=False)
    multiclass_models.pop("Dummy")
    mc_results, _, mc_predictions, _ = train_and_evaluate(
        multiclass_models,
        mc_train,
        mc_y_train,
        mc_test,
        mc_y_test,
        task="multiclass",
    )
    mc_results.to_csv(table_dir / "multiclass_model_metrics.csv", index=False)
    save_metric_bars(
        mc_results,
        ["balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"],
        "Attack-category classification performance",
        figure_dir / "multiclass_model_comparison.png",
    )
    labels = sorted(np.unique(multiclass_target).tolist())
    for model_name, predicted in mc_predictions.items():
        report = classification_report(
            mc_y_test,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report_frame = pd.DataFrame(report).transpose().reset_index(names="class")
        safe_name = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        report_frame.to_csv(
            table_dir / f"multiclass_classification_report_{safe_name}.csv", index=False
        )
        if model_name == "Random Forest":
            save_confusion_matrix(
                mc_y_test,
                predicted,
                labels,
                "Random Forest: normalized attack-category confusion matrix",
                figure_dir / "multiclass_confusion_random_forest.png",
                normalize=True,
            )

    friday_mask = sample[SOURCE_COLUMN].str.startswith("Friday")
    day_train = sample.loc[~friday_mask]
    day_test = sample.loc[friday_mask]
    day_y_train = (day_train[LABEL_COLUMN] != "Benign").astype(np.int8).to_numpy()
    day_y_test = (day_test[LABEL_COLUMN] != "Benign").astype(np.int8).to_numpy()
    day_results, _, _, _ = train_and_evaluate(
        build_models(RANDOM_SEED, include_unweighted=False),
        day_train[feature_columns],
        day_y_train,
        day_test[feature_columns],
        day_y_test,
        task="binary",
    )
    day_results.to_csv(table_dir / "day_shift_binary_metrics.csv", index=False)
    save_metric_bars(
        day_results,
        ["balanced_accuracy", "precision", "recall", "f1", "mcc"],
        "Friday day-shift detection performance",
        figure_dir / "day_shift_model_comparison.png",
    )

    existing = table_dir / "experiment_design.json"
    design = json.loads(existing.read_text(encoding="utf-8")) if existing.exists() else {}
    design.update(
        {
            "feature_count": len(feature_columns),
            "binary_train_rows": len(x_train),
            "binary_validation_rows": len(x_validation),
            "binary_test_rows": len(x_test),
            "multiclass_rows": len(multiclass),
            "multiclass_train_rows": len(mc_train),
            "multiclass_test_rows": len(mc_test),
            "multiclass_cap_per_class": 7_500,
            "day_shift_train_rows": len(day_train),
            "day_shift_test_rows": len(day_test),
        }
    )
    existing.write_text(json.dumps(design, indent=2), encoding="utf-8")
    print("Resumed experiments completed successfully.", flush=True)


if __name__ == "__main__":
    main()
