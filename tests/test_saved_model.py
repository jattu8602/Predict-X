import numpy as np
import pandas as pd

from src.baseline import FOREST_FEATURES, fit_random_forest
from src.predict import load_forest, predict_latest
from src.train_model import save_forest


def test_saved_forest_reproduces_a_prediction_at_the_latest_cycle(tmp_path):
    rows = pd.DataFrame({feature: np.arange(1, 21) for feature in FOREST_FEATURES})
    rows["unit_id"] = [1] * 10 + [2] * 10
    rows["RUL"] = 20 - rows["cycle"]
    forest = fit_random_forest(rows, FOREST_FEATURES)
    expected = forest.predict(rows.loc[rows["cycle"].eq(10), FOREST_FEATURES])[0]
    path = tmp_path / "forest.joblib"
    save_forest(forest, path)

    restored = load_forest(path)
    cycle, prediction = predict_latest(restored, rows.iloc[::-1], engine_id=1)

    assert cycle == 10
    assert prediction == expected
