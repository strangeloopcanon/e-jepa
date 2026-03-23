from __future__ import annotations

from pathlib import Path

import pandas as pd

from structured_jepa.schema import StepBatch
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
