"""Prepare engine-grouped FD001 partitions and training-fitted features."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data_loader import FEATURES, ROOT, load_fd001

PROCESSED_DIR = ROOT / "data" / "processed"
VALIDATION_CYCLES = (50, 100, 150)


def add_rul(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Label run-to-failure trajectories in remaining operating cycles."""
    final_cycle = trajectories.groupby("unit_id")["cycle"].transform("max")
    return trajectories.assign(RUL=final_cycle - trajectories["cycle"])


def split_engines(
    rows: pd.DataFrame, validation_fraction: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_indices, val_indices = next(splitter.split(rows, groups=rows["unit_id"]))
    return (
        rows.iloc[train_indices].reset_index(drop=True),
        rows.iloc[val_indices].reset_index(drop=True),
    )


def prepare_fd001(output_dir: Path = PROCESSED_DIR) -> None:
    full_train, test, test_targets = load_fd001()
    train, validation = split_engines(add_rul(full_train))
    scaler = StandardScaler().fit(train[FEATURES])

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        scaled = rows.astype({feature: "float64" for feature in FEATURES})
        scaled[FEATURES] = scaler.transform(rows[FEATURES])
        scaled.to_csv(output_dir / f"{name}.csv", index=False)

    test_targets.to_csv(output_dir / "test_targets.csv")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    print(f"Fit engines: {train.unit_id.nunique()} | Validation engines: {validation.unit_id.nunique()}")
    print(f"Official test engines: {test.unit_id.nunique()}")
    print(f"Processed data saved to {output_dir}")


if __name__ == "__main__":
    prepare_fd001()
