# Machine Learning Network Intrusion Detection Using CICIDS2017

This repository contains the implementation, experiments, and evaluation developed for my Master's thesis in Computer Science at California State University, Northridge (CSUN).

The research investigates supervised machine-learning approaches for network intrusion detection using the CICIDS2017 dataset. The models evaluated in this study include Logistic Regression, Linear Support Vector Machine (SVM), and Random Forest, with a Dummy Classifier included as a baseline for comparison.

The experimental analysis covers binary intrusion detection, multiclass attack classification, class-weight sensitivity, and cross-day robustness evaluation.

---

## Repository Structure

```text
data/                 Dataset information and processed data
demo/                 Command-line demonstration using the trained model
docs/                 Methodology and supporting documentation
notebooks/            Experimental analysis notebook
outputs/figures/      Generated figures and visualizations
outputs/tables/       Experimental results and evaluation metrics
outputs/models/       Trained machine-learning models
scripts/              Experiment and utility scripts
src/                  Data preprocessing, modeling, evaluation, and visualization
tests/                Pipeline and functionality tests
README.md             Project documentation
requirements.txt      Python dependencies
```

---

## Dataset

This project uses the **CICIDS2017** dataset developed by the Canadian Institute for Cybersecurity at the University of New Brunswick.

The dataset contains benign network traffic and multiple categories of network attacks and is used in this research to evaluate machine-learning approaches for network intrusion detection.

The CICIDS2017 MachineLearningCSV dataset can be obtained from the official dataset website:

[CICIDS2017 Dataset - Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

After downloading and extracting the dataset, place the eight CICIDS2017 CSV files in:

```text
data/raw/MachineLearningCVE/
```

The raw CICIDS2017 CSV files are not included in this GitHub repository because of their size. Keeping the raw dataset separate also allows the original data to be obtained directly from its official source.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mahdi20202/cicids2017-ml-intrusion-detection.git
```

Navigate to the project directory:

```bash
cd cicids2017-ml-intrusion-detection
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Tests

Before running the main experiments, the project tests can be executed with:

```powershell
python -m unittest discover -s tests -v
```

These tests verify important components of the data-processing and experimental pipeline.

---

## Running the Main Experiment

After placing the CICIDS2017 CSV files in:

```text
data/raw/MachineLearningCVE/
```

run the main experiment using:

```powershell
python -m src.nids_experiment --data-dir data/raw/MachineLearningCVE
```

The experimental pipeline performs the major stages of the research workflow, including:

- Loading the CICIDS2017 CSV files
- Cleaning and preparing the network traffic data
- Removing invalid and non-finite values
- Removing exact duplicate feature-label records
- Creating training, validation, and testing partitions
- Training the baseline classifier
- Training Logistic Regression
- Training Linear SVM
- Training Random Forest
- Performing binary intrusion detection
- Performing multiclass attack classification
- Evaluating class-weight sensitivity
- Performing cross-day evaluation
- Calculating evaluation metrics
- Generating figures and result tables
- Saving the trained model

---

## Experimental Design

### Binary Classification

The binary classification experiment distinguishes between:

- **Benign traffic**
- **Attack traffic**

The following models are evaluated:

- Dummy Classifier
- Logistic Regression
- Linear Support Vector Machine
- Random Forest

The Dummy Classifier serves as a baseline for comparison with the trained machine-learning models.

### Multiclass Classification

The multiclass experiment evaluates the ability of the selected machine-learning models to distinguish among attack categories represented in the CICIDS2017 dataset.

This analysis provides additional information beyond binary attack detection by examining how effectively individual attack categories can be identified.

### Class-Weight Analysis

Class imbalance is an important consideration in network intrusion detection.

Weighted and unweighted Logistic Regression configurations are evaluated to examine how class weighting affects detection performance.

### Cross-Day Evaluation

A cross-day experiment is included to investigate model robustness when the training and testing traffic originate from different portions of the dataset.

The experiment trains models using non-Friday traffic and evaluates them using Friday network traffic.

This provides an additional evaluation of model behavior under a distribution shift between training and testing data.

---

## Data Preprocessing

The preprocessing pipeline prepares CICIDS2017 network-flow features for machine-learning analysis.

Important preprocessing steps include:

- Cleaning feature names
- Handling invalid values
- Removing non-finite numerical values
- Removing exact duplicate feature-label records
- Preparing binary and multiclass labels
- Creating reproducible data partitions
- Applying preprocessing based on training data
- Preparing features for the selected classifiers

The experimental workflow uses fixed random seeds and deterministic sampling where applicable to improve reproducibility.

Preprocessing is fitted using training data to reduce the risk of information leakage between training and evaluation data.

---

## Machine-Learning Models

### Logistic Regression

Logistic Regression provides a linear classification model and serves as an interpretable machine-learning approach for intrusion detection.

### Linear Support Vector Machine

The Linear SVM evaluates whether a linear decision boundary can effectively distinguish benign and malicious network traffic in the CICIDS2017 feature space.

### Random Forest

Random Forest provides a nonlinear ensemble-learning approach based on multiple decision trees.

The model is also used to examine feature importance and identify network-flow characteristics that contribute strongly to classification decisions.

### Dummy Classifier

A Dummy Classifier is included as a baseline.

This allows the performance of the trained machine-learning models to be compared against a simple reference strategy.

---

## Evaluation Metrics

Model performance is evaluated using multiple metrics because network intrusion datasets can contain substantial class imbalance.

The evaluation includes:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Matthews Correlation Coefficient (MCC)
- ROC-AUC, where applicable
- Confusion Matrices

Using multiple metrics provides a more complete assessment than relying on accuracy alone.

---

## Results

Generated experimental results are stored in:

```text
outputs/
├── figures/
├── models/
└── tables/
```

### Figures

The `outputs/figures/` directory contains visualizations generated during the experimental analysis, including:

- Binary classification confusion matrices
- Binary model comparison
- ROC curves
- Multiclass confusion matrices
- Multiclass model comparison
- Class distribution visualization
- Cross-day model comparison
- Random Forest feature importance

### Tables

The `outputs/tables/` directory contains machine-readable experimental results, including:

- Binary classification metrics
- Multiclass classification metrics
- Classification reports
- Dataset statistics
- Cross-day evaluation results
- Feature importance results
- Experimental configuration information

### Models

The `outputs/models/` directory contains the trained machine-learning model used for demonstration and additional analysis.

---

## Results Validation

After completing the experiment, the generated results can be validated using:

```powershell
python -m src.validate_results
```

The validation stage performs consistency checks on the generated experimental outputs.

---

## Jupyter Notebook

The project includes an analytical Jupyter notebook:

```text
notebooks/cicids2017_experiment_results.ipynb
```

The notebook provides an additional way to review experimental results and visualizations.

The notebook can be built or updated using:

```powershell
python scripts/build_notebook.py
```

It can then be opened in Jupyter Notebook or Visual Studio Code.

---

## Running the Demonstration

After the main experiment has generated the trained binary model, run the demonstration with:

```powershell
python demo/run_demo.py
```

A CICIDS2017 CSV file can also be supplied directly:

```powershell
python demo/run_demo.py "data/raw/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"
```

The demonstration loads the trained binary classifier and produces intrusion-detection predictions for sample network traffic.

---

## Reproducing the Experimental Workflow

To reproduce the experimental workflow:

1. Clone this repository.
2. Download CICIDS2017 from the official source.
3. Extract the eight MachineLearningCSV files.
4. Place the CSV files in `data/raw/MachineLearningCVE/`.
5. Create and activate the Python virtual environment.
6. Install the dependencies from `requirements.txt`.
7. Run the project tests.
8. Run the main experiment.
9. Validate the generated results.
10. Review the generated figures, tables, and notebook.

On Windows PowerShell, the primary commands are:

```powershell
.\.venv\Scripts\Activate.ps1

python -m unittest discover -s tests -v

python -m src.nids_experiment --data-dir data/raw/MachineLearningCVE

python scripts/build_notebook.py

python -m src.validate_results
```

---

## Research Scope

This research focuses on evaluating traditional supervised machine-learning methods for network intrusion detection using the CICIDS2017 benchmark dataset.

The study examines model behavior across several experimental settings, including:

- Binary benign-versus-attack detection
- Multiclass attack classification
- Class-imbalance sensitivity
- Cross-day generalization

The results are specific to the CICIDS2017 dataset, selected features, preprocessing methodology, machine-learning models, and experimental design used in this study.

The experimental results should not be interpreted as evidence of production deployment readiness or guaranteed detection of previously unseen or zero-day attacks.

---

## Reproducibility

Reproducibility is an important component of this research.

The repository includes:

- Source code for data preparation
- Model-training implementations
- Evaluation procedures
- Automated tests
- Experimental configuration
- Generated result tables
- Generated figures
- An analytical notebook
- A command-line demonstration

Fixed random seeds and consistent experimental procedures are used where applicable to improve repeatability.

The raw CICIDS2017 dataset is intentionally excluded from the repository and should be obtained from its official source.

---

## Project Requirements

The Python packages required to run the project are listed in:

```text
requirements.txt
```

Install them using:

```powershell
pip install -r requirements.txt
```

---

## Author

**Mahdi Talebiroveshti**  
M.S. Computer Science  
California State University, Northridge

---

## Thesis Committee

**Dr. Jeff Wiegley**  
Thesis Chair  
Department of Computer Science  
California State University, Northridge

**Dr. Katya Mkrtchyan**  
Committee Member  
Department of Computer Science  
California State University, Northridge

**Dr. Marian Asadinia**  
Committee Member  
Department of Computer Science  
California State University, Northridge

---

## Acknowledgment

This research uses the CICIDS2017 dataset provided by the Canadian Institute for Cybersecurity at the University of New Brunswick.

Relevant datasets, methodologies, software libraries, and prior research used in the study are cited in the accompanying master's thesis.

---

## Disclaimer

This project was developed for academic research in network intrusion detection. The experimental results are intended for research and educational analysis and should not be interpreted as a guarantee of performance in operational cybersecurity environments.
