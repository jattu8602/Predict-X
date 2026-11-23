"""Train the FD001 telemetry forest once and save it for inference."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor

from src.baseline import FOREST_FEATURES, fit_random_forest
from src.data_loader import RAW_DIR, ROOT, load_trajectories
from src.preprocessing import add_rul

MODEL_PATH = ROOT / "models" / "random_forest_fd001.joblib"
DATASET_NAME = "NASA C-MAPSS FD001"


def save_forest(forest: RandomForestRegressor, destination: Path) -> None:
    """Save the fitted estimator with the exact inputs it expects."""
    artifact = {
        "model": forest,
        "dataset": DATASET_NAME,
        "feature_names": forest.feature_names_in_.tolist(),
        "sklearn_version": sklearn_version,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp_dir:
        staged_path = Path(temp_dir) / destination.name
        joblib.dump(artifact, staged_path, compress=3)
        staged_path.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    train = add_rul(load_trajectories(RAW_DIR / "train_FD001.txt"))
    forest = fit_random_forest(train, FOREST_FEATURES)
    save_forest(forest, args.output)
    print(f"Trained on {train['unit_id'].nunique()} complete FD001 engine histories.")
    print(f"Saved the forest and its {len(FOREST_FEATURES)} input columns to {args.output}.")


if __name__ == "__main__":
    main()
