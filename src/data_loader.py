"""Read the NASA C-MAPSS FD001 text files."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
FEATURES = SETTINGS + SENSORS
COLUMNS = ["unit_id", "cycle", *FEATURES]


def _read_numbers(path: Path, column_count: int) -> pd.DataFrame:
    rows = pd.read_csv(path, sep=r"\s+", header=None)
    if rows.shape[1] != column_count or not np.isfinite(rows.to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name} must contain {column_count} finite numbers per row.")
    return rows


def load_trajectories(path: Path) -> pd.DataFrame:
    """Return measurements in engine and cycle order."""
    rows = _read_numbers(path, len(COLUMNS))
    rows.columns = COLUMNS
    for key in ("unit_id", "cycle"):
        values = rows[key]
        if ((values < 1) | (values % 1 != 0)).any():
            raise ValueError(f"{path.name} has invalid {key} values.")
        rows[key] = values.astype("int64")

    rows = rows.sort_values(["unit_id", "cycle"], ignore_index=True)
    expected_cycles = rows.groupby("unit_id").cumcount() + 1
    if not rows["cycle"].eq(expected_cycles).all():
        raise ValueError(f"{path.name} must contain cycles 1 through N once per engine.")
    return rows


def load_fd001(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return run-to-failure training rows, partial test rows, and separate test labels."""
    train = load_trajectories(raw_dir / "train_FD001.txt")
    test = load_trajectories(raw_dir / "test_FD001.txt")
    labels = _read_numbers(raw_dir / "RUL_FD001.txt", 1).iloc[:, 0]
    if ((labels < 0) | (labels % 1 != 0)).any():
        raise ValueError("Test RUL must be a nonnegative whole number of cycles.")

    expected_ids = np.arange(1, len(labels) + 1)
    if not np.array_equal(test["unit_id"].unique(), expected_ids):
        raise ValueError("Expected exactly one RUL label per test engine, in unit ID order.")
    test_targets = pd.Series(
        labels.to_numpy(dtype="int64"),
        index=pd.Index(expected_ids, name="unit_id"),
        name="RUL",
    )
    return train, test, test_targets
