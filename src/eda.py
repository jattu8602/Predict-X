"""Summarize training trajectories and generate three EDA figures."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.data_loader import RAW_DIR, ROOT, SENSORS, load_trajectories
from src.preprocessing import add_rul, split_engines

FIGURE_DIR = ROOT / "outputs" / "figures"


def generate_eda(output_dir: Path = FIGURE_DIR) -> None:
    complete = add_rul(load_trajectories(RAW_DIR / "train_FD001.txt"))
    train, _ = split_engines(complete)
    lifetimes = train.groupby("unit_id")["cycle"].max()
    sample_ids = lifetimes.index[:4]
    constants = [sensor for sensor in SENSORS if train[sensor].nunique() == 1]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fitting partition: {len(train):,} rows across {len(lifetimes)} engines")
    print(f"Engine lifetime: {lifetimes.min()} to {lifetimes.max()} cycles")
    print("Constant sensors:", ", ".join(constants) or "none")
    print(train[SENSORS].describe().T.round(4).to_string())

    fig, ax = plt.subplots(layout="constrained")
    ax.hist(lifetimes, bins=12, color="#136f76", edgecolor="white")
    ax.set(title="FD001 engine lifetimes", xlabel="Cycles to failure", ylabel="Engines")
    fig.savefig(output_dir / "engine_lifetimes.png", dpi=150)
    plt.close(fig)

    for column, ylabel, title, filename in (
        ("RUL", "Remaining cycles", "Remaining useful life by cycle", "rul_trajectories.png"),
        ("sensor_11", "Sensor 11 reading", "Sensor 11 by cycle", "sensor_11_trajectories.png"),
    ):
        fig, ax = plt.subplots(layout="constrained")
        for engine in sample_ids:
            rows = train.loc[train["unit_id"].eq(engine)]
            ax.plot(rows["cycle"], rows[column], label=f"Engine {engine}")
        ax.set(title=title, xlabel="Operating cycle", ylabel=ylabel)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    generate_eda()
