import numpy as np
import pandas as pd

from src.data_loader import FEATURES, load_fd001
from src.preprocessing import add_rul, prepare_fd001, split_engines


def test_rul_counts_down_to_failure():
    rows = pd.DataFrame({"unit_id": [1, 1, 2, 2, 2], "cycle": [1, 2, 1, 2, 3]})
    assert add_rul(rows)["RUL"].tolist() == [1, 0, 2, 1, 0]
    assert "RUL" not in rows


def test_split_keeps_each_engine_in_one_partition():
    rows, _, _ = load_fd001()
    train, validation = split_engines(rows)
    assert train.unit_id.nunique() == 80
    assert validation.unit_id.nunique() == 20
    assert set(train.unit_id).isdisjoint(validation.unit_id)
    assert len(train) + len(validation) == len(rows)


def test_fd001_pipeline_fits_on_training_rows_and_keeps_test_labels_separate(tmp_path):
    train_raw, test_raw, targets = load_fd001()
    prepare_fd001(tmp_path)
    train = pd.read_csv(tmp_path / "train.csv")
    validation = pd.read_csv(tmp_path / "validation.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    exported_targets = pd.read_csv(tmp_path / "test_targets.csv", index_col="unit_id")["RUL"]
    assert (len(train) + len(validation), len(test), len(targets)) == (20631, 13096, 100)
    assert "RUL" not in test
    pd.testing.assert_series_equal(exported_targets, targets)
    pd.testing.assert_frame_equal(test[["unit_id", "cycle"]], test_raw[["unit_id", "cycle"]])
    expected_train, _ = split_engines(train_raw)
    varying = [feature for feature in FEATURES if expected_train[feature].nunique() > 1]
    np.testing.assert_allclose(train[varying].mean(), 0, atol=1e-9)
    np.testing.assert_allclose(train[varying].std(ddof=0), 1, atol=1e-9)
