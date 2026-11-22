# PredictX

PredictX estimates the remaining useful life (RUL) of turbofan engines from multivariate sensor time series. The project uses NASA C-MAPSS FD001 to prepare engine trajectories and benchmark regression models.

## Project layout

```text
project-X/
├── data/
│   ├── raw/                 # Three NASA FD001 source files
│   └── processed/           # Generated CSVs and fitted scaler
├── notebooks/
│   └── 01_eda.ipynb         # Summary, correlations and trajectory plots
├── src/
│   ├── baseline.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── lstm.py
│   ├── preprocessing.py
│   └── windows.py
├── outputs/figures/         # Generated PNGs
├── models/                  # Local model artifacts
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

The baseline uses scikit-learn. The LSTM also requires [PyTorch](https://pytorch.org/get-started/locally/), installed separately.

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

## 4. Compare baseline models

From the project directory, run:

```sh
python -m src.baseline
```

The script fits models on 80 engines and compares them on 59 observations from 20 held-out engines. Evaluation uses cycles 50, 100, and 150 wherever the engine reached that cycle. Each prediction uses the current cycle and, for the full model, the measurements recorded at that cycle. The engine ID only determines which partition a row belongs to. The script fits directly on raw measurements because decision trees split features by thresholds and do not require standardization.

| Model | Validation MAE (cycles) | Validation RMSE (cycles) |
| --- | ---: | ---: |
| Training median | 46.31 | 56.69 |
| Random Forest, age only | 36.99 | 41.04 |
| Random Forest, age and telemetry | 25.54 | 32.88 |

MAE averages the absolute error in cycles. RMSE gives larger misses more weight. The two forests use the same hyperparameters, so the comparison measures the effect of adding sensor readings and operating settings on this validation split. Keep the official test labels for final evaluation after selecting a model.

## 5. Build sequence inputs

Run `python -m src.windows` to inspect the shapes and see a labeled example from FD001. The script splits by engine first, then builds overlapping windows of 20 consecutive cycles within each engine. Each input has 20 rows and 25 features per row (cycle, three settings, and 21 sensor measurements). Engine ID stays in the endpoint metadata and is used for grouping.

If a window ends at cycle 101, it uses measurements from cycles 82 through 101 and takes its RUL target from cycle 101. Validation windows end at the same cycles used for the Random Forest benchmark (50, 100, and 150), so the two approaches can later be scored on the same observations. This step builds model inputs from raw features. The LSTM training script fits its feature scaler on the training engines before applying it to both partitions.

## 6. Train the LSTM

Install a CPU build of PyTorch in the project environment using the [PyTorch installer](https://pytorch.org/get-started/locally/). Then run:

```sh
python -m src.lstm
```

The LSTM reads one 20-cycle window in order and updates its hidden state after each reading. A linear layer uses the last hidden state to predict one RUL value. The script fits the feature scaler on the original training rows before transforming the overlapping windows, so each original row contributes equally to the scaler. It trains for 15 epochs with Adam and evaluates once on the same 59 validation endpoints used by the Random Forest.

| Model | Validation MAE (cycles) | Validation RMSE (cycles) |
| --- | ---: | ---: |
| Random Forest, age and telemetry | 25.54 | 32.88 |
| LSTM, 20 cycles of age and telemetry | 24.14 | 32.48 |

These results use one FD001 validation split (seed 42). The LSTM result is from PyTorch 2.15.0 on CPU. The script reports running training error while the model is being updated, followed by validation error on held-out engines. The 1.40-cycle difference in validation MAE is specific to this run. Official test labels are reserved for final evaluation.
