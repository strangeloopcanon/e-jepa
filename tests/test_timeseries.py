from __future__ import annotations

from pathlib import Path

import pandas as pd

from structured_jepa.storage import load_processed_dataset
from structured_jepa.timeseries import prepare_timeseries_dataset


def test_prepare_timeseries_dataset_writes_expected_schema(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "account_id": ["a", "a", "a", "b", "b", "b"],
            "event_time": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "revenue": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "tickets_open": [3, 2, 1, 1, 2, 3],
            "segment": ["smb", "smb", "smb", "enterprise", "enterprise", "enterprise"],
            "discount_rate": [0.1, 0.2, 0.1, 0.0, 0.0, 0.1],
            "target_backlog": [6.0, 5.0, 4.0, 2.0, 3.0, 4.0],
        }
    )
    input_path = tmp_path / "business.csv"
    raw.to_csv(input_path, index=False)

    prepared = prepare_timeseries_dataset(
        input_path=input_path,
        output_dir=tmp_path / "dataset",
        entity_column="account_id",
        timestamp_column="event_time",
        observation_categorical_columns=["segment"],
        action_numeric_columns=["discount_rate"],
        auxiliary_numeric_target_columns=["target_backlog"],
        seed=3,
    )
    loaded = load_processed_dataset(prepared.root)

    assert loaded.schema.dataset_kind == "timeseries"
    assert "obs_num__revenue" in loaded.frame.columns
    assert "mask__obs_num__revenue" in loaded.frame.columns
    assert "obs_cat__segment" in loaded.frame.columns
    assert "act_num__discount_rate" in loaded.frame.columns
    assert "aux_num__target_backlog" in loaded.frame.columns
    assert set(loaded.frame["split"]) <= {"train", "val", "test"}
