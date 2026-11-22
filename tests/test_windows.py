import numpy as np
import pandas as pd

from src.data_loader import FEATURES
from src.preprocessing import add_rul
from src.windows import WINDOW_FEATURES, make_windows


def test_windows_stay_in_one_engine_and_use_the_final_cycle_label():
    rows = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2, 3],
            "cycle": [2, 1, 3, 1, 2, 1],
            **{feature: [10, 10, 10, 20, 20, 30] for feature in FEATURES},
        }
    )
    inputs, endpoints = make_windows(add_rul(rows), window_size=2)

    assert inputs.shape == (3, 2, len(WINDOW_FEATURES))
    np.testing.assert_array_equal(inputs[:, :, 0], [[1, 2], [2, 3], [1, 2]])
    np.testing.assert_array_equal(inputs[:, :, 1], [[10, 10], [10, 10], [20, 20]])
    assert endpoints["unit_id"].tolist() == [1, 1, 2]
    assert endpoints["cycle"].tolist() == [2, 3, 2]
    assert endpoints["RUL"].tolist() == [1, 0, 0]


def test_short_trajectories_return_a_correctly_shaped_empty_batch():
    rows = pd.DataFrame({"unit_id": [1], "cycle": [1], "RUL": [0], **dict.fromkeys(FEATURES, 10)})
    inputs, endpoints = make_windows(rows, window_size=2)

    assert inputs.shape == (0, 2, len(WINDOW_FEATURES))
    assert endpoints.empty


def test_unlabeled_windows_keep_their_engine_and_final_cycle():
    rows = pd.DataFrame(
        {"unit_id": [1, 1, 2, 2], "cycle": [1, 2, 1, 2], **dict.fromkeys(FEATURES, 10)}
    )
    inputs, endpoints = make_windows(rows, window_size=2)

    np.testing.assert_array_equal(inputs[:, :, 0], [[1, 2], [1, 2]])
    pd.testing.assert_frame_equal(
        endpoints, pd.DataFrame({"unit_id": [1, 2], "cycle": [2, 2]})
    )
