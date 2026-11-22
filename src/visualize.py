"""Regenerate the README figures from FD001 and the project training scripts."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.baseline import predict_baselines
from src.data_loader import RAW_DIR, ROOT, load_trajectories
from src.lstm import train_and_predict
from src.preprocessing import VALIDATION_CYCLES, add_rul, split_engines
from src.windows import ENDPOINT_COLUMNS, WINDOW_SIZE

FIGURE_DIR = ROOT / "docs" / "figures"
BLUE = "#255b9b"
TEAL = "#087f73"
GRAY = "#64748b"
LSTM_NAME = "LSTM, 20-cycle window"


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURE_DIR / filename, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_engine(rows: pd.DataFrame) -> None:
    example = rows.loc[rows["unit_id"].eq(1)]
    end = 100
    start = end - WINDOW_SIZE + 1
    remaining = int(example.loc[example["cycle"].eq(end), "RUL"].iloc[0])
    failure = int(example["cycle"].max())
    fig, (sensor, rul) = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, layout="constrained")
    fig.suptitle("Engine 1 · From a sensor reading to a countdown", fontsize=16, fontweight="bold")
    for ax in (sensor, rul):
        ax.axvspan(start, end, color=TEAL, alpha=0.13)
        ax.axvline(end, color=TEAL, linestyle="--", linewidth=1.3)

    sensor.plot(example["cycle"], example["sensor_11"], color=BLUE, linewidth=1.5)
    sensor.set(ylabel="Sensor 11 reading", title=f"The shaded window covers {WINDOW_SIZE} readings from cycles {start}–{end}")

    rul.plot(example["cycle"], example["RUL"], color=BLUE, linewidth=2)
    rul.scatter([end, failure], [remaining, 0], color=TEAL, zorder=4)
    rul.annotate(f"{remaining} cycles left", (end, remaining), xytext=(12, 15), textcoords="offset points")
    rul.annotate(f"Failure at cycle {failure}", (failure, 0), xytext=(-8, 42),
                 textcoords="offset points", ha="right")
    rul.set(xlabel="Operating cycle", ylabel="Cycles remaining", ylim=(-10, failure))
    save(fig, "engine-window.png")


def plot_training(history: list[float]) -> None:
    epochs = range(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
    ax.plot(epochs, history, marker="o", markersize=4, linewidth=2, color=TEAL)
    ax.annotate(f"{history[0]:.2f}", (1, history[0]), xytext=(9, 5), textcoords="offset points")
    ax.annotate(f"{history[-1]:.2f}", (len(history), history[-1]), xytext=(-2, 12),
                ha="right", textcoords="offset points")
    ax.set(title="The LSTM’s error while training", xlabel="Pass through the training data (epoch)",
           ylabel="Average error (cycles)", ylim=(0, max(history) * 1.16))
    ax.set_xticks([1, 3, 6, 9, 12, len(history)])
    ax.grid(axis="y", alpha=0.22)
    save(fig, "lstm-training.png")


def plot_comparison(scores: pd.DataFrame, snapshot_count: int) -> None:
    labels = ["Training median", "Forest · age only", "Forest · age + readings", "LSTM · 20-cycle window"]
    fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
    bars = ax.barh(labels, scores["MAE"], color=[GRAY, BLUE, BLUE, TEAL])
    ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=11)
    ax.invert_yaxis()
    ax.set(title=f"Same {snapshot_count} quiz questions, different models", xlabel="Average validation error (cycles) · lower is better",
           xlim=(0, scores["MAE"].max() * 1.2))
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    save(fig, "validation-comparison.png")


def plot_predictions(actual: pd.Series, predictions: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.65), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("Closer to the dashed line means a better prediction", fontsize=14, fontweight="bold")
    plotted_models = ("Random Forest, age + telemetry", LSTM_NAME)
    max_value = max(actual.max(), *(predictions[name].max() for name in plotted_models))
    axis_limit = np.ceil(max_value / 25) * 25
    for ax, name, label, color in zip(
        axes,
        plotted_models,
        ("Random Forest", "LSTM"),
        (BLUE, TEAL),
    ):
        ax.axline((0, 0), slope=1, color=GRAY, linestyle="--", linewidth=1)
        ax.scatter(actual, predictions[name], color=color, alpha=0.7, s=24, edgecolors="none")
        ax.set(title=label, xlabel="Actual cycles remaining", xlim=(0, axis_limit), ylim=(0, axis_limit))
        ax.set_aspect("equal")
    axes[0].set_ylabel("Predicted cycles remaining")
    save(fig, "actual-vs-predicted.png")


def main() -> None:
    rows = add_rul(load_trajectories(RAW_DIR / "train_FD001.txt"))
    train, validation = split_engines(rows)
    snapshots = validation.loc[validation["cycle"].isin(VALIDATION_CYCLES)]
    predictions = predict_baselines(train, snapshots)
    history, endpoints, predictions[LSTM_NAME] = train_and_predict(train, validation)
    pd.testing.assert_frame_equal(snapshots[ENDPOINT_COLUMNS].reset_index(drop=True), endpoints)
    actual = endpoints["RUL"]
    scores = pd.DataFrame(
        {
            name: {"MAE": mean_absolute_error(actual, values), "RMSE": root_mean_squared_error(actual, values)}
            for name, values in predictions.items()
        }
    ).T

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_engine(rows)
    plot_training(history)
    plot_comparison(scores, len(actual))
    plot_predictions(actual, predictions)
    print(scores.round(2).to_string())
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
