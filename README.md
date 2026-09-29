# Machine Learning Network Intrusion Detection Using CICIDS2017

This repository contains the implementation and experimental evaluation developed for my M.S. thesis in Computer Science at California State University, Northridge (CSUN).

The study investigates supervised machine-learning methods for network intrusion detection using the **CICIDS2017** dataset. Logistic Regression, Linear Support Vector Machine (SVM), and Random Forest are evaluated against a baseline classifier. The experiments address binary intrusion detection, multiclass attack classification, class-imbalance sensitivity, and cross-day generalization.

## Repository Structure

```text
data/              Dataset information and local data directory
demo/              Inference demonstration
docs/              Methodology and supporting documentation
notebooks/         Experimental analysis notebook
outputs/           Figures, tables, and trained models
scripts/           Experiment and utility scripts
src/               Preprocessing, modeling, and evaluation code
tests/             Pipeline tests
```

## Dataset

This research uses the **CICIDS2017** dataset provided by the Canadian Institute for Cybersecurity at the University of New Brunswick.

The dataset is available from the official source:

[CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

After downloading the MachineLearningCSV files, place the eight CSV files in:

```text
data/raw/MachineLearningCVE/
```

The raw dataset is excluded from this repository because of its size and should be obtained directly from the official source.

## Installation

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Experiments

Run the project tests:

```powershell
python -m unittest discover -s tests -v
```

Run the main experiment:

```powershell
python -m src.nids_experiment --data-dir data/raw/MachineLearningCVE
```

Validate the generated results:

```powershell
python -m src.validate_results
```

The experimental pipeline performs data preprocessing, model training, binary and multiclass evaluation, class-weight analysis, cross-day evaluation, and generation of figures and result tables.

## Experimental Design

The study evaluates four primary components:

- **Binary classification:** benign versus malicious network traffic.
- **Multiclass classification:** classification of individual attack categories.
- **Class-weight analysis:** comparison of weighted and unweighted classification.
- **Cross-day evaluation:** training on non-Friday traffic and evaluation on Friday traffic.

Preprocessing is fitted using training data only to reduce information leakage. Fixed random seeds and deterministic procedures are used where applicable to support reproducibility.

## Evaluation

Model performance is assessed using:

- Accuracy and balanced accuracy
- Precision, recall, and F1-score
- False-positive and false-negative rates
- Matthews correlation coefficient (MCC)
- ROC-AUC, where applicable
- Confusion matrices

Generated results are stored under:

```text
outputs/
├── figures/
├── models/
└── tables/
```

## Demonstration

After training the binary model, run:

```powershell
python demo/run_demo.py
```

A CICIDS2017 CSV file may also be provided directly:

```powershell
python demo/run_demo.py "data/raw/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"
```

## Research Scope

This study provides a controlled comparison of supervised machine-learning methods for intrusion detection using CICIDS2017. The reported results are specific to the dataset, preprocessing procedures, model configurations, and evaluation protocol used in this research and should not be interpreted as evidence of production deployment readiness.

## Author

**Mahdi Talebiroveshti**  
M.S. Computer Science  
California State University, Northridge

## Thesis Committee

**Dr. Jeff Wiegley** — Thesis Chair  
**Dr. Katya Mkrtchyan** — Committee Member  
**Dr. Marjan Asadinia** — Committee Member  

Department of Computer Science  
California State University, Northridge

## Acknowledgment

This research uses the CICIDS2017 dataset provided by the Canadian Institute for Cybersecurity at the University of New Brunswick. Relevant datasets, methods, software tools, and prior research are cited in the accompanying master's thesis.
