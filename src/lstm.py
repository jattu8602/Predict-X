"""Train a PyTorch LSTM and evaluate it on the FD001 baseline endpoints."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import RAW_DIR, load_trajectories
from src.preprocessing import VALIDATION_CYCLES, add_rul, split_engines
from src.windows import WINDOW_FEATURES, make_windows

SEED = 42
EPOCHS = 15
BATCH_SIZE = 128
TARGET_SCALE = 100.0


class LSTMRegressor(nn.Module):
    """Read a sequence and use its final hidden state to predict RUL."""

    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        states, _ = self.lstm(windows)
        return self.head(states[:, -1, :]).squeeze(-1)


def scale_windows(windows: np.ndarray, scaler: StandardScaler) -> torch.Tensor:
    """Apply the same feature scaling at every cycle in every window."""
    flat_readings = windows.reshape(-1, windows.shape[-1])
    scaled = scaler.transform(flat_readings).reshape(windows.shape)
    return torch.as_tensor(scaled, dtype=torch.float32)


def train_and_predict(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[list[float], pd.DataFrame, np.ndarray]:
    """Return the training MAE by epoch, benchmark endpoints, and their predictions."""
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    # Fit once on raw fitting rows so overlapping windows do not reweight them.
    scaler = StandardScaler().fit(train[WINDOW_FEATURES].to_numpy())
    train_windows, train_endpoints = make_windows(train)
    validation_windows, validation_endpoints = make_windows(validation)

    train_inputs = scale_windows(train_windows, scaler)
    train_targets = torch.as_tensor(train_endpoints["RUL"].to_numpy(), dtype=torch.float32)
    dataset = TensorDataset(train_inputs, train_targets / TARGET_SCALE)
    batches = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED)
    )

    model = LSTMRegressor(input_size=len(WINDOW_FEATURES))
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        absolute_error = 0.0
        for inputs, targets in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            absolute_error += loss.item() * len(targets) * TARGET_SCALE
        epoch_mae = absolute_error / len(dataset)
        history.append(epoch_mae)
        print(f"Epoch {epoch:2}/{EPOCHS} | training MAE: {epoch_mae:.2f} cycles")

    # Use the same engine visits as the Random Forest comparison.
    benchmark_mask = validation_endpoints["cycle"].isin(VALIDATION_CYCLES).to_numpy()
    benchmark_inputs = scale_windows(validation_windows[benchmark_mask], scaler)
    endpoints = validation_endpoints.loc[benchmark_mask].reset_index(drop=True)
    model.eval()
    with torch.inference_mode():
        predicted = (model(benchmark_inputs) * TARGET_SCALE).clamp_min(0).numpy()

    return history, endpoints, predicted


def main() -> None:
    rows = load_trajectories(RAW_DIR / "train_FD001.txt")
    train, validation = split_engines(add_rul(rows))
    _, endpoints, predicted = train_and_predict(train, validation)
    actual = endpoints["RUL"]
    print(f"Validation windows: {len(actual)}")
    print(f"Validation MAE: {mean_absolute_error(actual, predicted):.2f} cycles")
    print(f"Validation RMSE: {root_mean_squared_error(actual, predicted):.2f} cycles")


if __name__ == "__main__":
    main()
