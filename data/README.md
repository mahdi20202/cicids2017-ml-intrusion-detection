# CICIDS2017 data

The raw dataset is not committed to ordinary Git history because the official archive is approximately 235 MB and individual extracted files can exceed GitHub's normal file-size limit.

1. Download `MachineLearningCSV.zip` from the official [CICIDS2017 dataset page](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Extract the eight CSV files into `data/raw/MachineLearningCVE/`.
3. Run `python -m src.nids_experiment` from the repository root.

The local archive used for the reported experiment had MD5 checksum:

```text
4f83860afbf29cac8163854095bf6cf7
```

Generated processed data is excluded from Git. The experiment script records row counts, invalid-value removal, sampled duplicate removal, class counts, sampling rules, and split sizes under `outputs/tables/`.

