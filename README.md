# PredictX

PredictX estimates the remaining useful life (RUL) of turbofan engines from multivariate sensor time series. Stage 1 prepares the NASA C-MAPSS FD001 data for a classical regression baseline and later PyTorch sequence models.

## Project layout

```text
project-X/
├── data/
│   ├── raw/                 # Three NASA FD001 source files
│   └── processed/           # Generated CSVs and fitted scaler
├── notebooks/
│   └── 01_eda.ipynb         # Summary, correlations and trajectory plots
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   └── preprocessing.py
├── outputs/figures/         # Generated PNGs
├── models/                  # Reserved for Stage 2
├── tests/
├── requirements.txt
└── .gitignore
```

## 1. Set up Python

Open `project-X` in VS Code or a terminal. Use Python 3.10 or newer.

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```sh
source .venv/bin/activate
```

Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

PyTorch will be used for sequence models in Stage 2. When starting that stage, install the build for your system using the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

## 2. Check the data

If you have the project archive, the supplied FD001 files are already in `data/raw/`. If you cloned the repository, download the original archive from the [NASA Turbofan Engine Degradation Simulation Data Set](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) and extract `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` into that directory. FD001 covers one operating condition and one fault mode. Each measurement row has 26 columns: engine ID, cycle, three operating settings, and 21 sensor measurements.

## 3. Run Stage 1

Run these commands from the project directory:

```sh
python -m src.eda
python -m src.preprocessing
python -m pytest -q
```

The EDA script prints sensor statistics and writes three figures to `outputs/figures/`. It uses the 80 fitting engines so the 20 validation engines stay held out when exploring features. Run `jupyter notebook` and open `notebooks/01_eda.ipynb` to inspect the data interactively, including the correlation matrix for varying sensors.

The preprocessing command creates `train.csv`, `validation.csv`, `test.csv`, `test_targets.csv`, and `scaler.joblib` in `data/processed/`. The measurement CSVs preserve the engine ID and operating cycle for grouping and sequencing. Their 24 operating and sensor features are standardized. Training and validation also include RUL in cycles.

## RUL labels and data leakage

For each run-to-failure engine, subtract the current cycle from its last cycle. For example, cycle 150 of an engine that fails at cycle 200 has an RUL of 50. Official test trajectories stop before failure. Each test label applies to the last observed cycle of its test engine and is exported separately.

The pipeline uses a reproducible split at the engine level with seed 42. Every trajectory belongs wholly to one partition. The standardization parameters come from fitting rows only, then the same scaler transforms validation and official test rows. When adding a model, use only readings at or before the prediction cycle, build windows within individual engines, and reserve official test labels for final evaluation.

Stage 1 stops at verified data preparation. It produces no model performance metrics.
