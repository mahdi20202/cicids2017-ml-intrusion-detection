$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
& ".\.venv\Scripts\python.exe" -m src.nids_experiment --data-dir "data\raw\MachineLearningCVE"
& ".\.venv\Scripts\python.exe" "scripts\build_notebook.py"
& ".\.venv\Scripts\python.exe" -m src.validate_results
