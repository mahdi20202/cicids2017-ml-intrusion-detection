# Machine Learning Network Intrusion Detection Using CICIDS2017

This repository contains the reproducible experimental work for a CSUN Computer Science master's thesis comparing supervised machine-learning classifiers for network intrusion detection. The evaluated models are Logistic Regression, a linear Support Vector Machine, and Random Forest. A dummy classifier provides a baseline.

The study uses the official CICIDS2017 machine-learning CSV files and reports binary detection, attack-category classification, class-weight sensitivity, and a day-shift robustness experiment.

## Repository structure

```text
data/                 Dataset acquisition notes and generated processed samples
demo/                 Command-line demonstration using the trained binary model
docs/                 Method, data dictionary, and reproducibility notes
notebooks/            Executed analytical notebook
outputs/figures/      Thesis-ready figures
outputs/tables/       Machine-readable results
outputs/models/       Locally generated trained model
reports/presentation/ 15-slide defense presentation (PPTX and PDF)
reports/proposal/     Research proposal source and PDF
reports/thesis/       CSUNthesis.cls source and compiled thesis PDF
scripts/              Reproducible experiment and build commands
src/                  Data preparation, modeling, evaluation, and chart generation
tests/                Focused pipeline tests
```

## Dataset

Download the MachineLearningCSV archive from the [Canadian Institute for Cybersecurity CICIDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html), then extract the eight CSV files into:

```text
data/raw/MachineLearningCVE/
```

The expected MD5 checksum for the archive used in this study is recorded in `data/raw/MachineLearningCSV.md5` when the local setup bundle is present. Raw data is excluded from Git because the archive exceeds GitHub's normal per-file limit and the source should remain traceable to the official distributor.

## Reproduce the experiments

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python -m src.nids_experiment --data-dir data/raw/MachineLearningCVE
python scripts/build_notebook.py
python -m src.validate_results
```

The experiment uses deterministic sampling and fixed random seeds. Preprocessing is fitted on training data only. The raw-data sample is cleaned before splitting, including removal of rows with non-finite feature values and exact duplicate feature-label records.

## Evaluation design

- Binary task: benign versus attack on a shared stratified holdout.
- Multiclass task: original attack labels, including rare classes where valid records are available.
- Class-weight sensitivity: weighted and unweighted Logistic Regression.
- Day-shift task: training on non-Friday files and testing on Friday traffic.
- Metrics: accuracy, balanced accuracy, precision, recall, F1-score, false-positive rate, false-negative rate, Matthews correlation coefficient, ROC-AUC where defined, and confusion matrices.

Results describe this benchmark and experimental protocol. They do not establish production readiness or zero-day detection capability.

## Thesis and defense

The thesis uses the supplied CSUN `CSUNthesis.cls` class. Personal and committee information remains in brace placeholders for the student to complete. The final defense deck contains 15 slides and is available in editable PPTX and PDF formats.

Current deliverables:

- `reports/thesis/main.pdf` - 62-page thesis draft in Times New Roman with one-inch margins and double-spaced body text.
- `notebooks/cicids2017_experiment_results.ipynb` - executed review notebook.
- `reports/presentation/ML_NIDS_CICIDS2017_Defense_Final.pptx` - editable 15-slide defense deck.
- `reports/presentation/ML_NIDS_CICIDS2017_Defense_Final.pdf` - defense PDF.
- `demo/run_demo.py` - local inference demonstration. Run without arguments to use the cleaned sample, or pass a CICIDS2017 CSV path.

## Academic integrity

All external claims, datasets, methods, and prior findings must remain cited. The student is responsible for reviewing the complete thesis, confirming the advisor's rules for AI-assisted work, running the institution-approved similarity check, and defending the methodology and results.
