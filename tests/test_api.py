import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import MODEL_ID, create_app
from src.baseline import FOREST_FEATURES, fit_random_forest
from src.data_loader import SENSORS, SETTINGS
from src.train_model import save_forest


def reading(cycle):
    return {"cycle": cycle, "settings": [cycle] * len(SETTINGS), "sensors": [cycle] * len(SENSORS)}


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    rows = pd.DataFrame({feature: np.arange(1, 31) for feature in FOREST_FEATURES})
    rows["RUL"] = 30 - rows["cycle"]
    forest = fit_random_forest(rows, FOREST_FEATURES)
    path = tmp_path_factory.mktemp("api-model") / "forest.joblib"
    save_forest(forest, path)
    with TestClient(create_app(path)) as client:
        yield client, forest, rows


def test_single_and_batch_predictions_use_the_saved_forest(service):
    client, forest, rows = service
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/models").json()[0]["input_columns"] == FOREST_FEATURES

    snapshots = [reading(27), reading(3)]
    batch = client.post("/predict/batch", json={"readings": snapshots})
    assert batch.status_code == 200
    expected = forest.predict(rows.iloc[[26, 2]][FOREST_FEATURES])
    assert batch.json() == [
        {"cycle": snapshot["cycle"], "predicted_rul_cycles": value, "model": MODEL_ID}
        for snapshot, value in zip(snapshots, expected, strict=True)
    ]
    assert client.post("/predict", json=snapshots[0]).json() == batch.json()[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cycle", 0),
        ("cycle", True),
        ("settings", [0] * 2),
        ("sensors", [0] * 20),
        ("sensors", [0] * 20 + ["invalid"]),
        ("sensors", [0] * 20 + [1e100]),
    ],
)
def test_invalid_readings_return_field_errors(service, field, value):
    client, _, _ = service
    response = client.post("/predict", json={**reading(1), field: value})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][:2] == ["body", field]


def test_nonfinite_reading_and_empty_batch_are_rejected(service):
    client, _, _ = service
    body = '{"cycle": 1, "settings": [1e309, 0, 0], "sensors": [' + ",".join(["0"] * 21) + "]}"
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "settings", 0]
    assert client.post("/predict/batch", json={"readings": []}).status_code == 422


def test_missing_model_explains_how_to_train_it(tmp_path):
    with pytest.raises(RuntimeError, match="python -m src.train_model"):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass
