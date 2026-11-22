"""Build fixed-length windows for a future sequence model."""

import numpy as np
import pandas as pd

from src.data_loader import FEATURES, RAW_DIR, load_trajectories
from src.preprocessing import VALIDATION_CYCLES, add_rul, split_engines

WINDOW_SIZE = 20
WINDOW_FEATURES = ["cycle", *FEATURES]
ENDPOINT_KEYS = ["unit_id", "cycle"]
ENDPOINT_COLUMNS = [*ENDPOINT_KEYS, "RUL"]


def make_windows(
    rows: pd.DataFrame, window_size: int = WINDOW_SIZE
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (windows, cycles, features) inputs and their endpoints.

    Pass consecutive cycles, as checked by the FD001 loader. RUL is optional metadata.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive.")

    endpoint_columns = ENDPOINT_COLUMNS if "RUL" in rows else ENDPOINT_KEYS
    windows = []
    endpoints = []
    for _, engine in rows.groupby("unit_id"):
        engine = engine.sort_values("cycle")
        if len(engine) < window_size:
            continue

        values = engine[WINDOW_FEATURES].to_numpy(dtype=np.float32)
        for stop in range(window_size, len(engine) + 1):
            # The slice excludes stop, so stop - 1 is the final reading.
            windows.append(values[stop - window_size : stop])

        endpoints.append(engine.iloc[window_size - 1 :][endpoint_columns])

    if not windows:
        empty_inputs = np.empty((0, window_size, len(WINDOW_FEATURES)), dtype=np.float32)
        return empty_inputs, rows[endpoint_columns].iloc[:0].reset_index(drop=True)

    return np.stack(windows), pd.concat(endpoints, ignore_index=True)


def main() -> None:
    rows = load_trajectories(RAW_DIR / "train_FD001.txt")
    train, validation = split_engines(add_rul(rows))
    train_windows, _ = make_windows(train)
    validation_windows, endpoints = make_windows(validation)
    benchmark_mask = endpoints["cycle"].isin(VALIDATION_CYCLES).to_numpy()
    benchmark_windows = validation_windows[benchmark_mask]

    print(f"Training input shape: {train_windows.shape}")
    print(f"Validation benchmark input shape: {benchmark_windows.shape}")
    print("Axes: windows, cycles per window, features per cycle")

    example_index = endpoints.index[endpoints["cycle"].eq(100)][0]
    example = endpoints.loc[example_index]
    start_cycle = int(validation_windows[example_index, 0, 0])
    print(
        f"Validation engine {example.unit_id}: cycles {start_cycle}–{example.cycle}; "
        f"target = {example.RUL} remaining cycles"
    )


if __name__ == "__main__":
    main()
