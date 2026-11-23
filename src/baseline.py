"""Compare FD001 RUL baselines at fixed cycles on held-out engines."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.data_loader import FEATURES, RAW_DIR, load_trajectories
from src.preprocessing import VALIDATION_CYCLES, add_rul, split_engines


FOREST_FEATURES = ["cycle", *FEATURES]


def fit_random_forest(train: pd.DataFrame, columns: list[str]) -> RandomForestRegressor:
    """Fit the fixed forest configuration used for both benchmark variants."""
    forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    return forest.fit(train[columns], train["RUL"])


def predict_baselines(train: pd.DataFrame, snapshots: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit the three baseline models and predict in snapshot row order."""
    age = ["cycle"]
    median = DummyRegressor(strategy="median").fit(train[age], train["RUL"])
    age_only = fit_random_forest(train, age)
    full_forest = fit_random_forest(train, FOREST_FEATURES)
    return {
        "Training median": median.predict(snapshots[age]),
        "Random Forest, age only": age_only.predict(snapshots[age]),
        "Random Forest, age + telemetry": full_forest.predict(snapshots[FOREST_FEATURES]),
    }


def main() -> None:
    trajectories = load_trajectories(RAW_DIR / "train_FD001.txt")
    train, validation = split_engines(add_rul(trajectories))
    snapshots = validation.loc[validation["cycle"].isin(VALIDATION_CYCLES)]

    print(f"Fitting engines: {train['unit_id'].nunique()}")
    print(f"Validation: {len(snapshots)} snapshots from {snapshots['unit_id'].nunique()} engines")
    print(f"Snapshots by cycle: {snapshots['cycle'].value_counts().sort_index().to_dict()}")
    print(f"\n{'Model':34} {'MAE (cycles)':>12} {'RMSE (cycles)':>13}")
    for name, predictions in predict_baselines(train, snapshots).items():
        mae = mean_absolute_error(snapshots["RUL"], predictions)
        rmse = root_mean_squared_error(snapshots["RUL"], predictions)
        print(f"{name:34} {mae:12.2f} {rmse:13.2f}")


if __name__ == "__main__":
    main()
