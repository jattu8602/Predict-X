"""Use a saved forest to estimate RUL at an engine's latest recorded cycle."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.baseline import FOREST_FEATURES
from src.data_loader import load_trajectories
from src.train_model import MODEL_PATH


def load_forest(path: Path) -> RandomForestRegressor:
    """Load a forest saved by the project training command."""
    artifact = joblib.load(path)
    forest = artifact["model"]
    if artifact["feature_names"] != FOREST_FEATURES or forest.feature_names_in_.tolist() != FOREST_FEATURES:
        raise ValueError("The saved forest does not match the FD001 inference feature schema.")
    return forest


def predict_latest(
    forest: RandomForestRegressor, rows: pd.DataFrame, engine_id: int
) -> tuple[int, float]:
    """Return the latest cycle and predicted RUL for one engine."""
    engine_rows = rows.loc[rows["unit_id"].eq(engine_id)]
    if engine_rows.empty:
        raise ValueError(f"Engine {engine_id} was not found in the input log.")

    latest = engine_rows.nlargest(1, "cycle")
    prediction = forest.predict(latest[FOREST_FEATURES])[0]
    return int(latest["cycle"].iloc[0]), float(prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NASA FD001-format engine readings")
    parser.add_argument("--engine", type=int, required=True, help="engine ID in the input file")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    try:
        forest = load_forest(args.model)
        rows = load_trajectories(args.input)
        cycle, estimate = predict_latest(forest, rows, args.engine)
    except FileNotFoundError as error:
        parser.error(f"File not found: {error.filename}")
    except ValueError as error:
        parser.error(str(error))

    print(f"Engine {args.engine} at cycle {cycle}: {estimate:.2f} cycles of estimated remaining life.")


if __name__ == "__main__":
    main()
