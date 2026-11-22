"""Score FD001 models once per engine at the end of its official test log."""

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.baseline import predict_baselines
from src.data_loader import load_fd001
from src.lstm import fit_lstm, predict_windows
from src.preprocessing import add_rul
from src.windows import ENDPOINT_KEYS, WINDOW_SIZE, make_windows


def main() -> None:
    train, test, test_targets = load_fd001()
    train = add_rul(train)

    # The loader sorts by engine and cycle. Keep the same endpoint for every model.
    test_groups = test.groupby("unit_id")
    last_rows = test_groups.tail(1).reset_index(drop=True)
    last_windows, endpoints = make_windows(test_groups.tail(WINDOW_SIZE))
    pd.testing.assert_frame_equal(last_rows[ENDPOINT_KEYS], endpoints)

    print(f"Fitting engines: {train['unit_id'].nunique()}")
    print(f"Official test engines: {len(last_rows)} | One prediction per engine")
    predictions = predict_baselines(train, last_rows)
    model, scaler, _ = fit_lstm(train)
    predictions["LSTM, 20-cycle window"] = predict_windows(model, scaler, last_windows)

    # Line up the answers by engine ID only after generating every prediction.
    actual = test_targets.loc[last_rows["unit_id"]]
    print(f"\n{'Model':34} {'MAE (cycles)':>12} {'RMSE (cycles)':>13}")
    for name, values in predictions.items():
        mae = mean_absolute_error(actual, values)
        rmse = root_mean_squared_error(actual, values)
        print(f"{name:34} {mae:12.2f} {rmse:13.2f}")


if __name__ == "__main__":
    main()
