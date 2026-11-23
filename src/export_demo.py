"""Export FD001 readings and saved-model estimates for the hosted dashboard."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.baseline import FOREST_FEATURES
from src.data_loader import RAW_DIR, ROOT, SETTINGS, load_trajectories
from src.predict import load_forest
from src.train_model import MODEL_PATH

DEMO_OUTPUT = ROOT / "dashboard" / "public" / "demo"


def export_demo(model_path: Path, input_path: Path, output: Path) -> None:
    """Write one data file per engine and a small dashboard index."""
    forest = load_forest(model_path)
    rows = load_trajectories(input_path)
    rows = rows.assign(predicted_rul_cycles=forest.predict(rows[FOREST_FEATURES]))
    output.mkdir(parents=True, exist_ok=True)
    engines = []
    with TemporaryDirectory(dir=output) as temporary:
        staging = Path(temporary)
        for engine_id, engine in rows.groupby("unit_id"):
            readings = [
                {
                    "cycle": cycle,
                    "settings": inputs[:len(SETTINGS)],
                    "sensors": inputs[len(SETTINGS):],
                    "predicted_rul_cycles": predicted,
                }
                for cycle, predicted, *inputs in engine[
                    ["cycle", "predicted_rul_cycles", *FOREST_FEATURES[1:]]
                ].itertuples(index=False, name=None)
            ]
            engines.append({"engine_id": int(engine_id), "latest_cycle": readings[-1]["cycle"]})
            payload = {"engine_id": int(engine_id), "readings": readings}
            (staging / f"engine-{engine_id}.json").write_text(
                json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8"
            )

        index = staging / "engines.json"
        index.write_text(json.dumps(engines, separators=(",", ":")), encoding="utf-8")
        for engine in engines:
            filename = f"engine-{engine['engine_id']}.json"
            (staging / filename).replace(output / filename)
        index.replace(output / index.name)

    print(f"Exported {len(rows):,} readings from {len(engines)} engines to {output}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEMO_OUTPUT)
    args = parser.parse_args()
    export_demo(MODEL_PATH, RAW_DIR / "test_FD001.txt", args.output)


if __name__ == "__main__":
    main()
