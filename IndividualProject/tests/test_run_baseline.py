import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from run_baseline import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    clean_frame,
    make_models,
)


def load_config():
    with (PROJECT_DIR / "config.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def synthetic_frame(rows: int = 120) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "year": [2010 + index % 10 for index in range(rows)],
            "manufacturer": ["ford" if index % 2 else "toyota" for index in range(rows)],
            "model": ["sedan" if index % 3 else "truck" for index in range(rows)],
            "condition": ["good" if index % 2 else "excellent" for index in range(rows)],
            "odometer": [50000 + index * 100 for index in range(rows)],
            "fuel": ["gas" for index in range(rows)],
            "transmission": ["automatic" for index in range(rows)],
            "drive": ["fwd" if index % 2 else "rwd" for index in range(rows)],
            "type": ["car" for index in range(rows)],
            "paint_color": ["blue" if index % 2 else "white" for index in range(rows)],
            "state": ["NY" if index % 2 else "CA" for index in range(rows)],
        }
    )
    frame["price"] = 30000 + (frame["year"] - 2010) * 500 - frame["odometer"] * 0.05
    return frame


def test_clean_frame_applies_price_year_and_odometer_filters() -> None:
    frame = synthetic_frame()
    frame.loc[0, "price"] = 100
    frame.loc[1, "year"] = 1800
    frame.loc[2, "odometer"] = 900000

    cleaned, stats = clean_frame(frame, load_config())

    assert len(cleaned) == 117
    assert stats["raw_rows"] == 120
    assert stats["excluded_rows"] == 3
    assert stats["clean_rows"] == 117


def test_both_models_fit_and_predict_synthetic_data() -> None:
    frame = synthetic_frame()
    x = frame[FEATURES]
    y = frame["price"]

    for model in make_models(load_config()).values():
        model.fit(x, y)
        predictions = model.predict(x.iloc[:5])
        assert len(predictions) == 5


def test_feature_groups_are_disjoint_and_cover_inputs() -> None:
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(FEATURES)
