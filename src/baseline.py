"""Compare FD001 RUL baselines at fixed cycles on held-out engines."""

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.data_loader import FEATURES, RAW_DIR, load_trajectories
from src.preprocessing import add_rul, split_engines

VALIDATION_CYCLES = (50, 100, 150)


def main() -> None:
    trajectories = load_trajectories(RAW_DIR / "train_FD001.txt")
    train, validation = split_engines(add_rul(trajectories))
    snapshots = validation.loc[validation["cycle"].isin(VALIDATION_CYCLES)]

    # Use the same forest settings to isolate the value of the extra inputs.
    forest_settings = {"n_estimators": 100, "min_samples_leaf": 5, "random_state": 42}
    comparisons = (
        ("Training median", DummyRegressor(strategy="median"), ["cycle"]),
        ("Random Forest, age only", RandomForestRegressor(**forest_settings), ["cycle"]),
        ("Random Forest, age + telemetry", RandomForestRegressor(**forest_settings), ["cycle", *FEATURES]),
    )

    print(f"Fitting engines: {train['unit_id'].nunique()}")
    print(f"Validation: {len(snapshots)} snapshots from {snapshots['unit_id'].nunique()} engines")
    print(f"Snapshots by cycle: {snapshots['cycle'].value_counts().sort_index().to_dict()}")
    print(f"\n{'Model':34} {'MAE (cycles)':>12} {'RMSE (cycles)':>13}")
    for name, model, columns in comparisons:
        model.fit(train[columns], train["RUL"])
        predictions = model.predict(snapshots[columns])
        mae = mean_absolute_error(snapshots["RUL"], predictions)
        rmse = root_mean_squared_error(snapshots["RUL"], predictions)
        print(f"{name:34} {mae:12.2f} {rmse:13.2f}")


if __name__ == "__main__":
    main()
