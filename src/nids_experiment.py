"""Run memory-safe CICIDS2017 experiments and export thesis-ready evidence."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


RANDOM_SEED = 42
LABEL_COLUMN = "Label"
SOURCE_COLUMN = "Source File"
BASE_SAMPLE_FRACTION = 0.20
RARE_CLASS_THRESHOLD = 1_000
MULTICLASS_CAP_PER_CLASS = 7_500

PALETTE = {
    "Dummy": "#7A7A7A",
    "Logistic Regression": "#1F5A94",
    "Logistic Regression (unweighted)": "#D89B27",
    "Linear SVM": "#D96C3B",
    "Random Forest": "#557A46",
}


def normalize_label(value: object) -> str:
    """Normalize known encoding and whitespace variants without merging attacks."""
    label = str(value).strip().replace("\ufffd", "-")
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"\s*-\s*", " - ", label)
    aliases = {
        "BENIGN": "Benign",
        "Web Attack - Sql Injection": "Web Attack - SQL Injection",
    }
    return aliases.get(label, label)


def clean_columns(columns: list[str]) -> list[str]:
    """Strip inconsistent CSV header whitespace while preserving distinct fields."""
    return [re.sub(r"\s+", " ", str(column).strip()) for column in columns]


def discover_csv_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.csv"))
    if len(files) != 8:
        raise FileNotFoundError(
            f"Expected 8 CICIDS2017 CSV files in {data_dir}, found {len(files)}."
        )
    return files


def count_raw_labels(files: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for file_path in files:
        labels = pd.read_csv(
            file_path,
            usecols=lambda column: str(column).strip() == LABEL_COLUMN,
            low_memory=False,
        ).iloc[:, 0]
        counts.update(normalize_label(value) for value in labels)
    return counts


def load_sample(
    files: list[Path],
    raw_label_counts: Counter[str],
    chunk_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Read all files in chunks and retain a deterministic class-aware sample."""
    rng = np.random.default_rng(seed)
    sampled_chunks: list[pd.DataFrame] = []
    profile: dict[str, object] = {
        "raw_rows": 0,
        "valid_rows": 0,
        "invalid_rows_removed": 0,
        "sample_rows_before_deduplication": 0,
        "files": {},
    }

    for file_path in files:
        file_raw = 0
        file_valid = 0
        file_invalid = 0
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            chunk.columns = clean_columns(chunk.columns.tolist())
            if LABEL_COLUMN not in chunk.columns:
                raise ValueError(f"{file_path.name} does not contain {LABEL_COLUMN}.")

            file_raw += len(chunk)
            labels = chunk.pop(LABEL_COLUMN).map(normalize_label)
            numeric = chunk.apply(pd.to_numeric, errors="coerce")
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            valid_mask = ~numeric.isna().any(axis=1) & labels.notna()
            file_invalid += int((~valid_mask).sum())
            numeric = numeric.loc[valid_mask].astype(np.float32)
            labels = labels.loc[valid_mask]
            file_valid += len(numeric)

            probabilities = labels.map(
                lambda label: 1.0
                if raw_label_counts[label] < RARE_CLASS_THRESHOLD
                else BASE_SAMPLE_FRACTION
            ).to_numpy(dtype=np.float32)
            keep_mask = rng.random(len(numeric)) < probabilities
            if not np.any(keep_mask):
                continue
            sampled = numeric.loc[keep_mask].copy()
            sampled[LABEL_COLUMN] = labels.loc[keep_mask].to_numpy()
            sampled[SOURCE_COLUMN] = file_path.name
            sampled_chunks.append(sampled)

        profile["raw_rows"] = int(profile["raw_rows"]) + file_raw
        profile["valid_rows"] = int(profile["valid_rows"]) + file_valid
        profile["invalid_rows_removed"] = int(profile["invalid_rows_removed"]) + file_invalid
        profile["files"][file_path.name] = {
            "raw_rows": file_raw,
            "valid_rows": file_valid,
            "invalid_rows_removed": file_invalid,
        }

    sample = pd.concat(sampled_chunks, ignore_index=True)
    profile["sample_rows_before_deduplication"] = len(sample)
    feature_columns = [
        column for column in sample.columns if column not in {LABEL_COLUMN, SOURCE_COLUMN}
    ]
    sample = sample.drop_duplicates(subset=feature_columns + [LABEL_COLUMN])
    profile["duplicate_sample_rows_removed"] = (
        int(profile["sample_rows_before_deduplication"]) - len(sample)
    )
    profile["sample_rows_after_deduplication"] = len(sample)
    profile["sample_label_counts"] = {
        key: int(value)
        for key, value in sample[LABEL_COLUMN].value_counts().sort_index().items()
    }
    return sample, profile


def build_models(seed: int, include_unweighted: bool = True) -> dict[str, object]:
    models: dict[str, object] = {
        "Dummy": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=500,
                        random_state=seed,
                        solver="saga",
                        tol=1e-3,
                    ),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=5_000,
                        random_state=seed,
                        tol=1e-2,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=20,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=2,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    if include_unweighted:
        models["Logistic Regression (unweighted)"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        random_state=seed,
                        solver="saga",
                        tol=1e-3,
                    ),
                ),
            ]
        )
    return models


def score_values(model: object, features: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        if np.ndim(scores) == 1:
            return np.asarray(scores)
    return None


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "false_negative_rate": fn / (fn + tp) if fn + tp else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    result["roc_auc"] = roc_auc_score(y_true, scores) if scores is not None else np.nan
    return result


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def train_and_evaluate(
    models: dict[str, object],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    task: str,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray | None]]:
    rows: list[dict] = []
    fitted: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray | None] = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predicted = model.predict(x_test)
        predict_seconds = time.perf_counter() - started
        model_scores = score_values(model, x_test) if task == "binary" else None
        metrics = (
            binary_metrics(y_test, predicted, model_scores)
            if task == "binary"
            else multiclass_metrics(y_test, predicted)
        )
        rows.append(
            {
                "model": name,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                **metrics,
            }
        )
        fitted[name] = model
        predictions[name] = predicted
        scores[name] = model_scores
        print(f"[{task}] {name}: {metrics}", flush=True)
    return pd.DataFrame(rows), fitted, predictions, scores


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    title: str,
    output_path: Path,
    normalize: bool,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    fig_width = max(7.2, len(labels) * 0.62)
    fig, axis = plt.subplots(figsize=(fig_width, fig_width * 0.82))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    axis.set_title(title, fontsize=13, pad=12)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    if len(labels) <= 4:
        threshold = float(np.nanmax(matrix)) / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                text = f"{value:.3f}" if normalize else f"{int(value):,}"
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "#202020",
                    fontsize=10,
                )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metric_bars(results: pd.DataFrame, metrics: list[str], title: str, output_path: Path) -> None:
    plot_data = results.set_index("model")[metrics]
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(metrics))
    width = 0.82 / len(plot_data)
    for index, (model, row) in enumerate(plot_data.iterrows()):
        positions = x - 0.41 + width / 2 + index * width
        axis.bar(
            positions,
            row.to_numpy(dtype=float),
            width=width,
            label=model,
            color=PALETTE.get(model, "#555555"),
        )
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("Score")
    axis.set_xticks(x, [metric.replace("_", " ").title() for metric in metrics])
    axis.set_title(title, fontsize=13, pad=12)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_class_distribution(counts: Counter[str], output_path: Path) -> None:
    series = pd.Series(counts).sort_values()
    fig, axis = plt.subplots(figsize=(10.5, 6.8))
    axis.barh(series.index, series.values, color="#1F5A94")
    axis.set_xscale("log")
    axis.set_xlabel("Number of flows (log scale)")
    axis.set_title("CICIDS2017 class distribution", fontsize=13, pad=12)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    for index, value in enumerate(series.values):
        axis.text(value * 1.06, index, f"{value:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray | None],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 6.2))
    for model, values in scores.items():
        if values is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, values)
        auc = roc_auc_score(y_true, values)
        axis.plot(fpr, tpr, label=f"{model} (AUC {auc:.3f})", color=PALETTE.get(model))
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Binary detection ROC curves", fontsize=13, pad=12)
    axis.grid(color="#D9D9D9", linewidth=0.7)
    axis.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model: Pipeline, feature_names: list[str], output_path: Path) -> pd.DataFrame:
    forest = model.named_steps["model"]
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": forest.feature_importances_}
    ).sort_values("importance", ascending=False)
    top = importance.head(20).sort_values("importance")
    fig, axis = plt.subplots(figsize=(9.2, 6.8))
    axis.barh(top["feature"], top["importance"], color="#557A46")
    axis.set_xlabel("Mean decrease in impurity")
    axis.set_title("Random Forest feature importance", fontsize=13, pad=12)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return importance


def prepare_multiclass_sample(sample: pd.DataFrame, seed: int) -> pd.DataFrame:
    frames = []
    for _, group in sample.groupby(LABEL_COLUMN, sort=True):
        frames.append(
            group.sample(
                n=min(len(group), MULTICLASS_CAP_PER_CLASS),
                random_state=seed,
            )
        )
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#111111",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def run(args: argparse.Namespace) -> None:
    configure_plots()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    processed_dir = Path(args.processed_dir)
    for directory in (figure_dir, table_dir, model_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    files = discover_csv_files(data_dir)
    raw_label_counts = count_raw_labels(files)
    save_class_distribution(raw_label_counts, figure_dir / "class_distribution_full.png")
    print(f"Raw label counts: {dict(raw_label_counts)}", flush=True)

    sample, profile = load_sample(files, raw_label_counts, args.chunk_size, args.seed)
    profile["raw_label_counts"] = dict(sorted(raw_label_counts.items()))
    profile["sample_fraction_for_common_classes"] = BASE_SAMPLE_FRACTION
    profile["rare_class_threshold"] = RARE_CLASS_THRESHOLD
    (table_dir / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        sorted(raw_label_counts.items()), columns=["label", "raw_rows"]
    ).to_csv(table_dir / "class_distribution_full.csv", index=False)
    sample.to_pickle(processed_dir / "cicids2017_sample.pkl.gz", compression="gzip")

    feature_columns = [
        column for column in sample.columns if column not in {LABEL_COLUMN, SOURCE_COLUMN}
    ]
    features = sample[feature_columns]
    binary_target = (sample[LABEL_COLUMN] != "Benign").astype(np.int8).to_numpy()
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        binary_target,
        test_size=0.30,
        stratify=binary_target,
        random_state=args.seed,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.50,
        stratify=y_holdout,
        random_state=args.seed,
    )
    split_summary = {
        "binary_train_rows": len(x_train),
        "binary_validation_rows": len(x_validation),
        "binary_test_rows": len(x_test),
        "feature_count": len(feature_columns),
        "random_seed": args.seed,
    }

    binary_results, binary_models, binary_predictions, binary_scores = train_and_evaluate(
        build_models(args.seed, include_unweighted=True),
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
    joblib.dump(forest, model_dir / "binary_random_forest.joblib", compress=3)
    (model_dir / "binary_model_metadata.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "negative_class": "Benign",
                "positive_class": "Attack",
                "random_seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    multiclass = prepare_multiclass_sample(sample, args.seed)
    multiclass_features = multiclass[feature_columns]
    multiclass_target = multiclass[LABEL_COLUMN].to_numpy()
    mc_train, mc_test, mc_y_train, mc_y_test = train_test_split(
        multiclass_features,
        multiclass_target,
        test_size=0.20,
        stratify=multiclass_target,
        random_state=args.seed,
    )
    multiclass_models = build_models(args.seed, include_unweighted=False)
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
    day_x_train = day_train[feature_columns]
    day_y_train = (day_train[LABEL_COLUMN] != "Benign").astype(np.int8).to_numpy()
    day_x_test = day_test[feature_columns]
    day_y_test = (day_test[LABEL_COLUMN] != "Benign").astype(np.int8).to_numpy()
    day_models = build_models(args.seed, include_unweighted=False)
    day_results, _, _, _ = train_and_evaluate(
        day_models,
        day_x_train,
        day_y_train,
        day_x_test,
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

    split_summary.update(
        {
            "multiclass_rows": len(multiclass),
            "multiclass_train_rows": len(mc_train),
            "multiclass_test_rows": len(mc_test),
            "day_shift_train_rows": len(day_train),
            "day_shift_test_rows": len(day_test),
        }
    )
    (table_dir / "experiment_design.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (table_dir / "software_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    print("Experiment completed successfully.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/MachineLearningCVE")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
