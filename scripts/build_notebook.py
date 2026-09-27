"""Create and execute the reproducible experiment-results notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "cicids2017_experiment_results.ipynb"


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def main() -> None:
    cells = [
        markdown(
            """
# CICIDS2017 supervised intrusion-detection experiments

## tl;dr

This notebook is the executed, reviewable companion to the thesis. It loads the generated evidence rather than retraining models during ordinary review. Use `python -m src.nids_experiment` to reproduce the full pipeline from the official CSV files.
"""
        ),
        markdown(
            """
## Context & Methods

The study compares Logistic Regression, linear Support Vector Machine, and Random Forest under three conditions: a stratified binary holdout, multiclass attack-category classification, and a Friday-only day-shift test. A dummy classifier anchors the binary comparison. All preprocessing is fitted on training data, exact sampled duplicates are removed before splitting, and all rare-class rows are retained when valid.

### Key assumptions

- CICIDS2017 is a controlled benchmark and does not represent every operational network.
- Random row splits can reward day-specific benchmark regularities, so the day-shift test is interpreted alongside the main holdout.
- Very small classes have unstable per-class estimates. Support counts must be read with their scores.
"""
        ),
        code(
            """
from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

profile = json.loads((TABLES / "dataset_profile.json").read_text(encoding="utf-8"))
design = json.loads((TABLES / "experiment_design.json").read_text(encoding="utf-8"))
binary = pd.read_csv(TABLES / "binary_model_metrics.csv")
multiclass = pd.read_csv(TABLES / "multiclass_model_metrics.csv")
day_shift = pd.read_csv(TABLES / "day_shift_binary_metrics.csv")
validation = json.loads((TABLES / "validation_checks.json").read_text(encoding="utf-8"))
"""
        ),
        markdown("## Data"),
        code(
            """
pd.DataFrame({
    "Measure": [
        "Raw flows", "Invalid rows removed", "Sampled duplicates removed",
        "Clean sampled flows", "Features", "Multiclass sample rows"
    ],
    "Value": [
        profile["raw_rows"], profile["invalid_rows_removed"],
        profile["duplicate_sample_rows_removed"], profile["sample_rows_after_deduplication"],
        design.get("feature_count", 78), design["multiclass_rows"]
    ]
})
"""
        ),
        code("display(Image(filename=str(FIGURES / 'class_distribution_full.png'), width=900))"),
        markdown(
            """
The logarithmic axis reveals the class imbalance that an overall accuracy figure would conceal. Heartbleed, SQL injection, and Infiltration have fewer than 40 raw records each.
"""
        ),
        markdown("## Results"),
        code(
            """
binary[["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "false_positive_rate", "false_negative_rate", "mcc", "roc_auc"]].round(4)
"""
        ),
        code("display(Image(filename=str(FIGURES / 'binary_model_comparison.png'), width=950))"),
        markdown(
            """
Random Forest performs best on the stratified random holdout. The dummy baseline's high accuracy and zero attack recall show why class-aware metrics are required. Weighting Logistic Regression increases attack recall substantially, at the cost of more false positives.
"""
        ),
        code(
            """
multiclass[["model", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_f1", "mcc"]].round(4)
"""
        ),
        code("display(Image(filename=str(FIGURES / 'multiclass_model_comparison.png'), width=950))"),
        markdown(
            """
Macro scores remain below weighted scores because rare categories are harder to classify. Random Forest leads, but estimates for classes with only a few test examples remain uncertain.
"""
        ),
        code(
            """
day_shift[["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1",
           "false_positive_rate", "false_negative_rate", "mcc", "roc_auc"]].round(4)
"""
        ),
        code("display(Image(filename=str(FIGURES / 'day_shift_model_comparison.png'), width=950))"),
        markdown(
            """
All learned models lose attack recall on Friday-only traffic. The decline is largest for Random Forest, which supports a cautious interpretation of its near-perfect random-split performance.
"""
        ),
        markdown("## Takeaways"),
        code(
            """
summary = {
    "Random-split best macro/binary model": binary.loc[binary["f1"].idxmax(), "model"],
    "Random Forest random-split attack recall": float(binary.loc[binary["model"] == "Random Forest", "recall"].iloc[0]),
    "Random Forest Friday attack recall": float(day_shift.loc[day_shift["model"] == "Random Forest", "recall"].iloc[0]),
    "Best multiclass macro F1": float(multiclass["macro_f1"].max()),
    "Independent validation passed": validation["all_checks_pass"],
}
pd.Series(summary, name="Observed value")
"""
        ),
        markdown(
            """
The benchmark supports Random Forest as the strongest model under the defined random-split conditions. The day-shift results show that this ranking does not establish operational generalization. Future work should test later datasets, group-aware or temporal splits, calibrated thresholds, and feature sets designed for cross-network comparability.
"""
        ),
    ]

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(notebook, OUTPUT)
    print(f"Wrote executed notebook: {OUTPUT}")


if __name__ == "__main__":
    main()

