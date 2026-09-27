# Experimental methodology

## Research tasks

The experiments address binary intrusion detection, attack-category classification, the effect of class weighting, and robustness to a source-day shift. Logistic Regression, a linear Support Vector Machine, and Random Forest use common data partitions within each task.

## Data preparation

The pipeline reads all eight official CICIDS2017 machine-learning CSV files in bounded chunks. It standardizes header whitespace and known label-encoding variants, converts every predictor to numeric form, removes rows containing missing or infinite predictor values, and retains the source filename as provenance metadata. The source filename is never used as a predictor.

Common classes are sampled at a deterministic 20% rate to fit the available 8 GB workstation. Every valid row from classes with fewer than 1,000 raw records is retained. Exact duplicate predictor-label rows are removed from the sampled data before splitting. This procedure yields a reproducible computational sample while preserving the rarest attacks.

## Splits and leakage controls

The binary experiment uses a fixed 70% training, 15% validation, and 15% test allocation with stratification. The reported comparison uses the untouched test portion. The multiclass experiment caps each label at 7,500 sampled rows and uses a fixed stratified 80/20 train-test split. Preprocessing transformations are fitted inside scikit-learn pipelines using training data only.

The day-shift experiment trains on Monday through Thursday source files and evaluates on all Friday source files. This split is not a complete production simulation because attack types differ by day, but it provides a stricter generalization check than a random row split.

## Models

- Logistic Regression uses standardized predictors, the SAGA solver, and balanced class weights in the main comparison.
- Linear SVM uses standardized predictors, balanced class weights, `C=1.0`, a tolerance of `0.01`, and at most 5,000 iterations.
- Random Forest uses 80 trees, maximum depth 20, minimum leaf size 2, and balanced bootstrap class weights.
- A prior-probability dummy classifier provides the binary baseline.

An additional unweighted Logistic Regression model isolates the effect of class weighting.

## Metrics

Binary evaluation reports accuracy, balanced accuracy, precision, recall, F1-score, Matthews correlation coefficient, false-positive rate, false-negative rate, ROC-AUC, and confusion counts. Attack is the positive class.

Multiclass evaluation reports accuracy, balanced accuracy, macro precision, macro recall, macro F1-score, weighted F1-score, Matthews correlation coefficient, and per-class precision, recall, F1-score, and support.

## Interpretation limits

The random split measures performance on held-out flows drawn from the same benchmark construction. It can reward repeated or day-specific patterns and must not be interpreted as evidence of zero-day detection. The day-shift experiment exposes some of this sensitivity, while later-dataset and cross-network testing remain future work. Estimates based on very small classes are unstable and must be reported with support counts.

