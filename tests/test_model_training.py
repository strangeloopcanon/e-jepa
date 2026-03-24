from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from structured_jepa.schema import ModelConfig, StepBatch, TrainConfig
from structured_jepa.storage import WindowDataset, collate_step_batches, load_processed_dataset
from structured_jepa.timeseries import prepare_timeseries_dataset
from structured_jepa.training import (
    evaluate_model,
    fit_linear_probe,
    fit_summary_decoder,
    load_trained_model,
    train_model,
)


def test_training_smoke_beats_naive_baseline_and_probe(tmp_path: Path) -> None:
    rows = []
    for episode_index in range(6):
        for step_index in range(24):
            control = float(step_index % 3 == 0)
            base = float(episode_index * 5 + step_index)
            rows.append(
                {
                    "entity_id": f"acct-{episode_index}",
                    "event_ts": f"2024-01-{(step_index % 9) + 1:02d}T00:00:00Z",
                    "signal_a": base + control,
                    "signal_b": base * 0.5,
                    "segment": "enterprise" if episode_index % 2 else "smb",
                    "control": control,
                    "target_load": base + control + 2.0,
                }
            )
    raw = pd.DataFrame(rows)
    input_path = tmp_path / "synthetic.csv"
    raw.to_csv(input_path, index=False)

    prepared = prepare_timeseries_dataset(
        input_path=input_path,
        output_dir=tmp_path / "dataset",
        entity_column="entity_id",
        timestamp_column="event_ts",
        observation_categorical_columns=["segment"],
        action_numeric_columns=["control"],
        auxiliary_numeric_target_columns=["target_load"],
        seed=7,
    )
    artifacts = train_model(
        dataset_root=prepared.root,
        output_dir=tmp_path / "model",
    )
    metrics = evaluate_model(
        dataset_root=prepared.root,
        checkpoint_path=artifacts.model_path,
    )
    probe = fit_linear_probe(
        dataset_root=prepared.root,
        checkpoint_path=artifacts.model_path,
        target_column="aux_num__target_load",
    )
    summary_decoder = fit_summary_decoder(
        dataset_root=prepared.root,
        checkpoint_path=artifacts.model_path,
        columns=["obs_num__signal_a", "obs_num__signal_b"],
    )

    assert artifacts.history.train_losses[-1] <= artifacts.history.train_losses[0]
    assert metrics.latent_mse < metrics.naive_observation_mse
    assert probe.mse < probe.baseline_mse
    assert summary_decoder["columns"] == ["obs_num__signal_a", "obs_num__signal_b"]
    assert len(summary_decoder["weights"]) == 129


@pytest.mark.parametrize("encoder_type", ["flat", "tokenized"])
def test_encoder_variants_train_and_support_summary_decoding(
    tmp_path: Path, encoder_type: str
) -> None:
    rows = []
    for episode_index in range(8):
        level = float(episode_index * 3)
        for step_index in range(28):
            control = float((step_index + episode_index) % 4 == 0)
            pressure = level + step_index * 0.5 + control * 2.0
            rows.append(
                {
                    "entity_id": f"team-{episode_index}",
                    "event_ts": f"2024-03-{(step_index % 9) + 1:02d}T00:00:00Z",
                    "throughput": pressure + 1.0,
                    "queue_depth": pressure * 0.75,
                    "segment": "enterprise" if episode_index % 2 else "midmarket",
                    "region": "west" if step_index % 2 else "east",
                    "control": control,
                    "target_load": pressure + 3.0,
                }
            )
    input_path = tmp_path / f"{encoder_type}.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    prepared = prepare_timeseries_dataset(
        input_path=input_path,
        output_dir=tmp_path / f"dataset-{encoder_type}",
        entity_column="entity_id",
        timestamp_column="event_ts",
        observation_categorical_columns=["segment", "region"],
        action_numeric_columns=["control"],
        auxiliary_numeric_target_columns=["target_load"],
        seed=11,
    )
    artifacts = train_model(
        dataset_root=prepared.root,
        output_dir=tmp_path / f"model-{encoder_type}",
        model_config=ModelConfig(encoder_type=encoder_type, feature_token_depth=1),
        train_config=TrainConfig(epochs=6, batch_size=8, seed=11),
    )
    metrics = evaluate_model(
        dataset_root=prepared.root,
        checkpoint_path=artifacts.model_path,
    )
    probe = fit_linear_probe(
        dataset_root=prepared.root,
        checkpoint_path=artifacts.model_path,
        target_column="aux_num__target_load",
    )
    summary_decoder = fit_summary_decoder(
        dataset_root=prepared.root,
        checkpoint_path=artifacts.model_path,
        columns=["obs_num__throughput", "obs_num__queue_depth"],
    )

    assert artifacts.history.train_losses[-1] <= artifacts.history.train_losses[0]
    assert metrics.mean_surprise >= 0.0
    assert probe.mse < probe.baseline_mse
    assert summary_decoder["columns"] == ["obs_num__throughput", "obs_num__queue_depth"]
    assert len(summary_decoder["weights"]) == 129


def test_surprise_score_rises_on_large_regime_change(tmp_path: Path) -> None:
    rows = []
    for episode_index in range(4):
        for step_index in range(20):
            rows.append(
                {
                    "entity_id": f"acct-{episode_index}",
                    "event_ts": f"2024-02-{(step_index % 9) + 1:02d}T00:00:00Z",
                    "signal_a": float(step_index),
                    "signal_b": float(step_index * 2),
                }
            )
    raw = pd.DataFrame(rows)
    input_path = tmp_path / "surprise.csv"
    raw.to_csv(input_path, index=False)

    prepared = prepare_timeseries_dataset(
        input_path=input_path,
        output_dir=tmp_path / "dataset",
        entity_column="entity_id",
        timestamp_column="event_ts",
    )
    artifacts = train_model(dataset_root=prepared.root, output_dir=tmp_path / "model")
    model, _ = load_trained_model(artifacts.model_path)
    dataset = WindowDataset(load_processed_dataset(prepared.root), split="train")
    batch = collate_step_batches([dataset[0], dataset[1]])
    clean_score = float(model.surprise_score(batch).mean().item())

    perturbed = StepBatch(
        observation_numeric=batch.observation_numeric.clone(),
        observation_masks=batch.observation_masks.clone(),
        observation_categorical=batch.observation_categorical.clone(),
        action_numeric=batch.action_numeric.clone(),
        action_masks=batch.action_masks.clone(),
        action_categorical=batch.action_categorical.clone(),
        auxiliary_numeric_targets=batch.auxiliary_numeric_targets.clone(),
        valid_mask=batch.valid_mask.clone(),
    )
    perturbed.observation_numeric[:, -1, 0] += 25.0
    noisy_score = float(model.surprise_score(perturbed).mean().item())

    assert noisy_score > clean_score


def test_prepare_timeseries_respects_custom_split_fractions(tmp_path: Path) -> None:
    rows = []
    for episode_index in range(10):
        for step_index in range(2):
            rows.append(
                {
                    "entity_id": f"acct-{episode_index}",
                    "event_ts": f"2024-04-{step_index + 1:02d}T00:00:00Z",
                    "signal_a": float(episode_index + step_index),
                }
            )
    input_path = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    prepared = prepare_timeseries_dataset(
        input_path=input_path,
        output_dir=tmp_path / "dataset",
        entity_column="entity_id",
        timestamp_column="event_ts",
        train_fraction=0.6,
        val_fraction=0.2,
        seed=5,
    )

    episode_splits = (
        prepared.frame[["episode_id", "split"]].drop_duplicates()["split"].value_counts().to_dict()
    )
    assert episode_splits == {"train": 6, "val": 2, "test": 2}
