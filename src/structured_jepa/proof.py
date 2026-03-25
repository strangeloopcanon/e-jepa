from __future__ import annotations

import json
from collections.abc import Sized
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import StructuredStateJEPA
from .readouts import apply_linear_readout, fit_least_squares
from .schema import DatasetSchema, ModelConfig, NumericFeatureSpec, StepBatch, TrainConfig
from .storage import (
    PreparedDataset,
    WindowDataset,
    collate_step_batches,
    dataset_paths,
    load_processed_dataset,
    rows_to_step_batch,
)
from .training import evaluate_model, load_trained_model, train_model
from .utils import ensure_directory, json_dump
from .vei_support import load_run_surface_summary, load_snapshot_diff_summary

VARIANT_LABELS = {
    "flat": "Flat encoder",
    "tokenized": "Tokenized encoder",
    "flat_no_actions": "Flat encoder without actions",
    "persistence": "Persistence baseline",
}


def benchmark_timeseries(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    train_config: TrainConfig | None = None,
) -> Path:
    prepared = load_processed_dataset(dataset_root)
    if prepared.schema.dataset_kind != "timeseries":
        raise ValueError("benchmark_timeseries requires a prepared timeseries dataset")

    output_root = ensure_directory(output_dir)
    model_root = ensure_directory(output_root / "models")
    selected_columns = _select_timeseries_columns(prepared.schema)
    if not selected_columns:
        raise ValueError("timeseries benchmark needs at least one observation numeric column")

    run_config = train_config or TrainConfig()
    no_action_root = _write_no_action_dataset(prepared, output_root / "_derived" / "no_actions")
    variants = [
        ("flat", prepared.root, ModelConfig(encoder_type="flat")),
        ("tokenized", prepared.root, ModelConfig(encoder_type="tokenized", feature_token_depth=1)),
        ("flat_no_actions", no_action_root, ModelConfig(encoder_type="flat")),
    ]

    metrics: dict[str, Any] = {
        "dataset_kind": prepared.schema.dataset_kind,
        "dataset_root": str(prepared.root),
        "selected_columns": selected_columns,
        "selected_labels": _display_labels(prepared.schema, selected_columns),
        "variants": {},
    }

    persistence_mse: float | None = None
    best_variant: str | None = None
    best_variant_mse: float | None = None
    best_action_aware: str | None = None
    best_action_aware_mse: float | None = None

    for variant_name, variant_dataset_root, model_config in variants:
        variant_output = model_root / variant_name
        artifacts = train_model(
            dataset_root=variant_dataset_root,
            output_dir=variant_output,
            model_config=model_config,
            train_config=run_config,
        )
        evaluation = evaluate_model(
            dataset_root=variant_dataset_root,
            checkpoint_path=artifacts.model_path,
            device=run_config.device,
        )
        variant_prepared = load_processed_dataset(variant_dataset_root)
        model, _ = load_trained_model(artifacts.model_path)
        model = model.to(run_config.device)

        decoded_metrics = _evaluate_decoded_predictions(
            prepared=variant_prepared,
            model=model,
            selected_columns=selected_columns,
            device=run_config.device,
        )
        surprise_metrics = _evaluate_surprise_separation(
            prepared=variant_prepared,
            model=model,
            selected_columns=selected_columns,
            device=run_config.device,
        )

        variant_metrics = {
            "label": VARIANT_LABELS[variant_name],
            "model_path": str(Path(artifacts.model_path).resolve()),
            "train_loss_start": artifacts.history.train_losses[0],
            "train_loss_end": artifacts.history.train_losses[-1],
            "val_loss_end": artifacts.history.val_losses[-1],
            "latent_mse": evaluation.latent_mse,
            "mean_surprise": evaluation.mean_surprise,
            "decoded_prediction_mse": decoded_metrics["decoded_prediction_mse"],
            "persistence_mse": decoded_metrics["persistence_mse"],
            "surprise_clean": surprise_metrics["clean_surprise"],
            "surprise_perturbed": surprise_metrics["perturbed_surprise"],
            "surprise_lift": surprise_metrics["surprise_lift"],
        }
        metrics["variants"][variant_name] = variant_metrics

        if persistence_mse is None:
            persistence_mse = decoded_metrics["persistence_mse"]
        if best_variant_mse is None or decoded_metrics["decoded_prediction_mse"] < best_variant_mse:
            best_variant = variant_name
            best_variant_mse = decoded_metrics["decoded_prediction_mse"]
        if variant_name in {"flat", "tokenized"} and (
            best_action_aware_mse is None
            or decoded_metrics["decoded_prediction_mse"] < best_action_aware_mse
        ):
            best_action_aware = variant_name
            best_action_aware_mse = decoded_metrics["decoded_prediction_mse"]

    if persistence_mse is None:
        raise ValueError("benchmark did not produce a persistence baseline")

    metrics["variants"]["persistence"] = {
        "label": VARIANT_LABELS["persistence"],
        "decoded_prediction_mse": persistence_mse,
    }
    metrics["best_variant"] = best_variant
    metrics["best_action_aware_variant"] = best_action_aware
    metrics["action_aware_beats_persistence"] = (
        best_action_aware_mse is not None and best_action_aware_mse < persistence_mse
    )
    if best_action_aware_mse is not None:
        metrics["action_aware_improvement_vs_persistence"] = (
            persistence_mse - best_action_aware_mse
        ) / max(persistence_mse, 1e-8)

    metrics_path = output_root / "metrics.json"
    summary_path = output_root / "summary.md"
    prediction_plot_path = output_root / "prediction_quality.svg"
    surprise_plot_path = output_root / "surprise_separation.svg"

    json_dump(metrics_path, metrics)
    summary_path.write_text(_build_benchmark_summary(metrics), encoding="utf-8")
    _write_simple_bar_chart(
        prediction_plot_path,
        title="Normalized Next-State Prediction Error",
        values={
            VARIANT_LABELS["flat"]: metrics["variants"]["flat"]["decoded_prediction_mse"],
            VARIANT_LABELS["tokenized"]: metrics["variants"]["tokenized"]["decoded_prediction_mse"],
            VARIANT_LABELS["flat_no_actions"]: metrics["variants"]["flat_no_actions"][
                "decoded_prediction_mse"
            ],
            VARIANT_LABELS["persistence"]: persistence_mse,
        },
        subtitle="Lower is better",
    )
    _write_simple_bar_chart(
        surprise_plot_path,
        title="Surprise Separation On Perturbed Transitions",
        values={
            "Flat clean": metrics["variants"]["flat"]["surprise_clean"],
            "Flat perturbed": metrics["variants"]["flat"]["surprise_perturbed"],
            "Tokenized clean": metrics["variants"]["tokenized"]["surprise_clean"],
            "Tokenized perturbed": metrics["variants"]["tokenized"]["surprise_perturbed"],
            "No-actions clean": metrics["variants"]["flat_no_actions"]["surprise_clean"],
            "No-actions perturbed": metrics["variants"]["flat_no_actions"]["surprise_perturbed"],
        },
        subtitle="Higher perturbed bars are better",
    )
    return output_root


def benchmark_vei_demo(
    *,
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    episode_id: str | None = None,
    max_steps: int = 5,
    device: str = "cpu",
) -> Path:
    prepared = load_processed_dataset(dataset_root)
    if not prepared.schema.dataset_kind.startswith("vei_"):
        raise ValueError("benchmark_vei_demo requires a prepared VEI dataset")

    model, schema = load_trained_model(checkpoint_path)
    model = model.to(device)
    output_root = ensure_directory(output_dir)
    selected_columns = _select_demo_columns(schema)
    if not selected_columns:
        raise ValueError("VEI demo needs at least one numeric state column to summarize")

    decoder_weights = _fit_step_decoder(
        prepared=prepared,
        model=model,
        selected_columns=selected_columns,
        device=device,
    )
    episode_rows = _select_demo_episode(prepared.frame, schema, episode_id)
    run_surface_summary = _resolve_vei_run_surface_summary(
        prepared=prepared,
        episode_rows=episode_rows,
    )
    step_count = min(max_steps, len(episode_rows) - schema.window_size + 1)
    if step_count <= 0:
        raise ValueError("selected VEI episode is too short for the current context length")

    steps: list[dict[str, Any]] = []
    for start_index in range(step_count):
        window = episode_rows.iloc[start_index : start_index + schema.window_size].reset_index(
            drop=True
        )
        single = rows_to_step_batch(window, schema)
        batch = collate_step_batches([single]).to(device)
        with torch.no_grad():
            forward_pass = model.forward(batch)
            predicted_next = apply_linear_readout(
                forward_pass.predicted_latents[0, -1:].cpu(),
                decoder_weights,
            )[0]
            surprise_score = float(
                model.surprise_score(batch, forward_pass=forward_pass)[0, -1].cpu().item()
            )

        current_row = window.iloc[schema.context_length - 1]
        next_row = window.iloc[schema.context_length]
        actual_change_summary = _resolve_vei_snapshot_diff_summary(
            prepared=prepared,
            current_row=current_row,
            next_row=next_row,
        )
        step_entry = {
            "episode_id": str(current_row["episode_id"]),
            "current_step_idx": int(current_row["step_idx"]),
            "current_timestamp": str(current_row["timestamp"]),
            "next_timestamp": str(next_row["timestamp"]),
            "current_snapshot_id": _int_metadata_value(current_row, "meta__snapshot_id"),
            "next_snapshot_id": _int_metadata_value(next_row, "meta__snapshot_id"),
            "current_snapshot_label": _string_metadata_value(current_row, "meta__snapshot_label"),
            "next_snapshot_label": _string_metadata_value(next_row, "meta__snapshot_label"),
            "action": _build_action_summary(current_row, schema),
            "current_state_summary": _row_state_summary(current_row, schema, selected_columns),
            "predicted_next_state_summary": _decoded_state_summary(
                predicted_next,
                schema,
                selected_columns,
            ),
            "actual_next_state_summary": _row_state_summary(next_row, schema, selected_columns),
            "surprise_score": surprise_score,
            "actual_change_summary": actual_change_summary,
        }
        steps.append(step_entry)

    decoded_summary = {
        "selected_columns": selected_columns,
        "selected_labels": _display_labels(schema, selected_columns),
        "run_surface_summary": run_surface_summary,
        "steps": [
            {
                "current_step_idx": step["current_step_idx"],
                "current_state_summary": step["current_state_summary"],
                "predicted_next_state_summary": step["predicted_next_state_summary"],
                "actual_next_state_summary": step["actual_next_state_summary"],
                "actual_change_summary": step["actual_change_summary"],
            }
            for step in steps
        ],
    }

    json_dump(output_root / "demo_steps.json", steps)
    json_dump(output_root / "decoded_summary.json", decoded_summary)
    if run_surface_summary is not None:
        json_dump(output_root / "run_surface_summary.json", run_surface_summary)
    (output_root / "summary.md").write_text(
        _build_vei_demo_summary(steps, selected_columns, schema, run_surface_summary),
        encoding="utf-8",
    )
    return output_root


def write_brief(
    *,
    benchmark_dir: str | Path,
    output_path: str | Path,
    vei_demo_dir: str | Path | None = None,
) -> Path:
    benchmark_root = Path(benchmark_dir).expanduser().resolve()
    metrics = json.loads((benchmark_root / "metrics.json").read_text(encoding="utf-8"))

    vei_steps: list[dict[str, Any]] = []
    if vei_demo_dir is not None:
        demo_root = Path(vei_demo_dir).expanduser().resolve()
        vei_steps = json.loads((demo_root / "demo_steps.json").read_text(encoding="utf-8"))

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_build_brief(metrics, vei_steps), encoding="utf-8")
    return output


def _write_no_action_dataset(prepared: PreparedDataset, output_root: Path) -> Path:
    steps = prepared.frame.copy()
    for numeric_feature in prepared.schema.action_numeric:
        steps[numeric_feature.name] = 0.0
        steps[numeric_feature.mask_column] = 0.0
    for categorical_feature in prepared.schema.action_categorical:
        steps[categorical_feature.name] = categorical_feature.missing_token
    steps["action_name"] = "__none__"

    dataset_root = ensure_directory(output_root)
    steps_path, schema_path = dataset_paths(dataset_root)
    steps.to_parquet(steps_path, index=False)
    schema_path.write_text(
        prepared.schema.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return dataset_root


def _evaluate_decoded_predictions(
    *,
    prepared: PreparedDataset,
    model: StructuredStateJEPA,
    selected_columns: list[str],
    device: str,
) -> dict[str, float]:
    weights = _fit_next_step_decoder(
        prepared=prepared,
        model=model,
        selected_columns=selected_columns,
        device=device,
    )
    eval_loader = _select_eval_loader(prepared, batch_size=32)
    column_indexes = _column_indexes(prepared.schema, selected_columns)

    predicted_rows: list[torch.Tensor] = []
    actual_rows: list[torch.Tensor] = []
    persistence_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in eval_loader:
            moved = batch.to(device)
            forward_pass = model.forward(moved)
            predicted = forward_pass.predicted_latents.reshape(
                -1, forward_pass.predicted_latents.size(-1)
            )
            actual = moved.observation_numeric[:, 1:, column_indexes].reshape(
                -1, len(column_indexes)
            )
            persistence = moved.observation_numeric[:, :-1, column_indexes].reshape(
                -1, len(column_indexes)
            )
            predicted_rows.append(apply_linear_readout(predicted.cpu(), weights).cpu())
            actual_rows.append(actual.cpu())
            persistence_rows.append(persistence.cpu())

    predicted_matrix = torch.cat(predicted_rows)
    actual_matrix = torch.cat(actual_rows)
    persistence_matrix = torch.cat(persistence_rows)
    decoded_prediction_mse = float(torch.mean((predicted_matrix - actual_matrix) ** 2).item())
    persistence_mse = float(torch.mean((persistence_matrix - actual_matrix) ** 2).item())
    return {
        "decoded_prediction_mse": decoded_prediction_mse,
        "persistence_mse": persistence_mse,
    }


def _fit_next_step_decoder(
    *,
    prepared: PreparedDataset,
    model: StructuredStateJEPA,
    selected_columns: list[str],
    device: str,
) -> torch.Tensor:
    train_loader = _build_loader(prepared, split="train", batch_size=32)
    column_indexes = _column_indexes(prepared.schema, selected_columns)
    latent_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in train_loader:
            moved = batch.to(device)
            state_latents = model.encode_steps(moved)[:, 1:].reshape(-1, model.config.d_state)
            targets = moved.observation_numeric[:, 1:, column_indexes].reshape(
                -1, len(column_indexes)
            )
            latent_rows.append(state_latents.cpu())
            target_rows.append(targets.cpu())

    return fit_least_squares(torch.cat(latent_rows), torch.cat(target_rows))


def _fit_step_decoder(
    *,
    prepared: PreparedDataset,
    model: StructuredStateJEPA,
    selected_columns: list[str],
    device: str,
) -> torch.Tensor:
    train_loader = _build_loader(prepared, split="train", batch_size=32)
    column_indexes = _column_indexes(prepared.schema, selected_columns)
    latent_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in train_loader:
            moved = batch.to(device)
            latents = model.encode_steps(moved).reshape(-1, model.config.d_state)
            targets = moved.observation_numeric[:, :, column_indexes].reshape(
                -1, len(column_indexes)
            )
            latent_rows.append(latents.cpu())
            target_rows.append(targets.cpu())

    return fit_least_squares(torch.cat(latent_rows), torch.cat(target_rows))


def _evaluate_surprise_separation(
    *,
    prepared: PreparedDataset,
    model: StructuredStateJEPA,
    selected_columns: list[str],
    device: str,
) -> dict[str, float]:
    eval_loader = _select_eval_loader(prepared, batch_size=32)
    perturb_indexes = _column_indexes(
        prepared.schema, selected_columns[: min(3, len(selected_columns))]
    )
    clean_scores: list[torch.Tensor] = []
    perturbed_scores: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in eval_loader:
            moved = batch.to(device)
            clean_scores.append(model.surprise_score(moved).reshape(-1).cpu())
            perturbed = _clone_step_batch(moved)
            perturbed.observation_numeric[:, -1, perturb_indexes] += 4.0
            perturbed_scores.append(model.surprise_score(perturbed).reshape(-1).cpu())

    clean_surprise = float(torch.cat(clean_scores).mean().item())
    perturbed_surprise = float(torch.cat(perturbed_scores).mean().item())
    return {
        "clean_surprise": clean_surprise,
        "perturbed_surprise": perturbed_surprise,
        "surprise_lift": perturbed_surprise - clean_surprise,
    }


def _clone_step_batch(batch: StepBatch) -> StepBatch:
    return StepBatch(
        observation_numeric=batch.observation_numeric.clone(),
        observation_masks=batch.observation_masks.clone(),
        observation_categorical=batch.observation_categorical.clone(),
        action_numeric=batch.action_numeric.clone(),
        action_masks=batch.action_masks.clone(),
        action_categorical=batch.action_categorical.clone(),
        auxiliary_numeric_targets=batch.auxiliary_numeric_targets.clone(),
        valid_mask=batch.valid_mask.clone(),
    )


def _build_loader(
    prepared: PreparedDataset, *, split: str, batch_size: int
) -> DataLoader[StepBatch]:
    return DataLoader(
        WindowDataset(prepared, split=split),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_step_batches,
    )


def _select_eval_loader(prepared: PreparedDataset, *, batch_size: int) -> DataLoader[StepBatch]:
    for split in ["test", "val", "train"]:
        loader = _build_loader(prepared, split=split, batch_size=batch_size)
        if len(cast(Sized, loader.dataset)) > 0:
            return loader
    raise ValueError("dataset does not contain any usable windows")


def _select_timeseries_columns(schema: DatasetSchema) -> list[str]:
    return [
        feature.name
        for feature in schema.observation_numeric
        if feature.source_column not in {"delta_t_s"}
    ]


def _select_demo_columns(schema: DatasetSchema, max_columns: int = 6) -> list[str]:
    priority_patterns = [
        "obs_num__pending_total__all",
        "obs_num__surface_panel_count__all",
        "obs_num__surface_status__",
        "obs_num__surface_signal__",
        "obs_num__clock_time_ms",
        "obs_num__graph_count__",
        "obs_num__graph_status__",
        "obs_num__orientation_count__",
        "obs_num__interval_event_count__all",
    ]
    selected: list[str] = []
    for pattern in priority_patterns:
        for column in schema.observation_numeric_columns:
            if column in selected:
                continue
            if column == pattern or column.startswith(pattern):
                selected.append(column)
                if len(selected) == max_columns:
                    return selected
    for feature in schema.observation_numeric:
        if feature.name in selected or feature.source_column == "delta_t_s":
            continue
        selected.append(feature.name)
        if len(selected) == max_columns:
            break
    return selected


def _column_indexes(schema: DatasetSchema, columns: list[str]) -> list[int]:
    return [schema.observation_numeric_columns.index(column) for column in columns]


def _display_labels(schema: DatasetSchema, columns: list[str]) -> list[str]:
    return [_display_label(_numeric_spec(schema, column)) for column in columns]


def _display_label(feature: NumericFeatureSpec) -> str:
    return feature.source_column.replace("__", " / ")


def _numeric_spec(schema: DatasetSchema, column_name: str) -> NumericFeatureSpec:
    for feature in schema.observation_numeric:
        if feature.name == column_name:
            return feature
    raise ValueError(f"numeric feature not found: {column_name}")


def _denormalize_value(schema: DatasetSchema, column_name: str, normalized_value: float) -> float:
    feature = _numeric_spec(schema, column_name)
    return float(normalized_value * feature.std + feature.mean)


def _row_state_summary(
    row: pd.Series,
    schema: DatasetSchema,
    selected_columns: list[str],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for column in selected_columns:
        feature = _numeric_spec(schema, column)
        summary[_display_label(feature)] = round(
            _denormalize_value(schema, column, float(row[column])),
            3,
        )
    return summary


def _decoded_state_summary(
    values: torch.Tensor,
    schema: DatasetSchema,
    selected_columns: list[str],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for index, column in enumerate(selected_columns):
        feature = _numeric_spec(schema, column)
        summary[_display_label(feature)] = round(
            _denormalize_value(schema, column, float(values[index].detach().cpu().item())),
            3,
        )
    return summary


def _select_demo_episode(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    episode_id: str | None,
) -> pd.DataFrame:
    if episode_id is not None:
        selected = frame.loc[frame["episode_id"] == episode_id].copy()
        if selected.empty:
            raise ValueError(f"episode not found in VEI dataset: {episode_id}")
        if len(selected) < schema.window_size:
            raise ValueError(
                "selected VEI episode does not have enough steps for the current context"
            )
        return selected.sort_values("step_idx").reset_index(drop=True)

    grouped = frame.groupby("episode_id", sort=False)
    for _candidate, rows in grouped:
        if len(rows) >= schema.window_size:
            return rows.sort_values("step_idx").reset_index(drop=True)
    raise ValueError("no VEI episode has enough steps for the current context")


def _build_action_summary(row: pd.Series, schema: DatasetSchema) -> dict[str, str | float]:
    summary: dict[str, str | float] = {"action_name": str(row["action_name"])}
    for categorical_feature in schema.action_categorical:
        summary[categorical_feature.source_column] = str(row[categorical_feature.name])
    for numeric_feature in schema.action_numeric:
        summary[numeric_feature.source_column] = round(
            _denormalize_numeric_feature(numeric_feature, float(row[numeric_feature.name])),
            3,
        )
    return summary


def _denormalize_numeric_feature(feature: NumericFeatureSpec, normalized_value: float) -> float:
    return float(normalized_value * feature.std + feature.mean)


def _resolve_vei_run_surface_summary(
    *,
    prepared: PreparedDataset,
    episode_rows: pd.DataFrame,
) -> dict[str, object] | None:
    if prepared.schema.dataset_kind != "vei_runs":
        return None
    if episode_rows.empty:
        return None

    first_row = episode_rows.iloc[0]
    from_metadata = _surface_summary_from_metadata(first_row)
    if from_metadata is not None:
        return from_metadata

    workspace_root = prepared.schema.notes.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root.strip():
        return None

    run_id = str(first_row["episode_id"])
    surface_summary, used_public_api = load_run_surface_summary(
        workspace_root=Path(workspace_root),
        run_id=run_id,
    )
    if not used_public_api:
        return None
    if (
        _coerce_int(surface_summary.get("panel_count", 0)) == 0
        and not str(surface_summary.get("current_tension", "")).strip()
    ):
        return None
    return surface_summary


def _resolve_vei_snapshot_diff_summary(
    *,
    prepared: PreparedDataset,
    current_row: pd.Series,
    next_row: pd.Series,
) -> dict[str, object] | None:
    if prepared.schema.dataset_kind != "vei_runs":
        return None

    workspace_root = prepared.schema.notes.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root.strip():
        return None

    snapshot_from = _int_metadata_value(current_row, "meta__snapshot_id")
    snapshot_to = _int_metadata_value(next_row, "meta__snapshot_id")
    if snapshot_from is None or snapshot_to is None:
        return None

    return load_snapshot_diff_summary(
        workspace_root=Path(workspace_root),
        run_id=str(current_row["episode_id"]),
        snapshot_from=snapshot_from,
        snapshot_to=snapshot_to,
    )


def _surface_summary_from_metadata(row: pd.Series) -> dict[str, object] | None:
    if "meta__surface_panel_count" not in row.index:
        return None

    panel_titles = [
        title.strip()
        for title in _string_metadata_value(row, "meta__surface_panel_titles").split(",")
        if title.strip()
    ]
    summary = {
        "company_name": _string_metadata_value(row, "meta__surface_company_name"),
        "vertical_name": _string_metadata_value(row, "meta__surface_vertical_name"),
        "current_tension": _string_metadata_value(row, "meta__surface_current_tension"),
        "panel_titles": panel_titles,
        "panel_count": _int_metadata_value(row, "meta__surface_panel_count") or 0,
        "item_count": _int_metadata_value(row, "meta__surface_item_count") or 0,
        "ok_count": _int_metadata_value(row, "meta__surface_ok_count") or 0,
        "attention_count": _int_metadata_value(row, "meta__surface_attention_count") or 0,
        "warning_count": _int_metadata_value(row, "meta__surface_warning_count") or 0,
        "critical_count": _int_metadata_value(row, "meta__surface_critical_count") or 0,
    }
    if not any(
        [
            summary["company_name"],
            summary["vertical_name"],
            summary["current_tension"],
            summary["panel_count"],
            summary["item_count"],
        ]
    ):
        return None
    return summary


def _string_metadata_value(row: pd.Series, column_name: str) -> str:
    if column_name not in row.index:
        return ""
    raw_value = row[column_name]
    if pd.isna(raw_value):
        return ""
    return str(raw_value)


def _int_metadata_value(row: pd.Series, column_name: str) -> int | None:
    if column_name not in row.index:
        return None
    raw_value = row[column_name]
    if pd.isna(raw_value):
        return None
    return _coerce_int(raw_value, default=None)


def _coerce_int(value: object, *, default: int | None = 0) -> int | None:
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return int(float(value))
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _build_benchmark_summary(metrics: dict[str, Any]) -> str:
    lines = [
        "# E-JEPA Benchmark Summary",
        "",
        "This report compares the current model variants on the same prepared business dataset.",
        "",
        f"- Best overall variant: `{metrics['best_variant']}`",
        f"- Best action-aware variant: `{metrics['best_action_aware_variant']}`",
        f"- Action-aware variant beats persistence: `{metrics['action_aware_beats_persistence']}`",
        "",
        "## Selected State Signals",
        "",
    ]
    lines.extend([f"- {label}" for label in metrics["selected_labels"]])
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Variant | Normalized next-state MSE | Surprise clean | Surprise perturbed |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for variant in ["flat", "tokenized", "flat_no_actions"]:
        row = metrics["variants"][variant]
        lines.append(
            f"| {row['label']} | {row['decoded_prediction_mse']:.4f} | "
            f"{row['surprise_clean']:.4f} | {row['surprise_perturbed']:.4f} |"
        )
    lines.append(
        f"| {metrics['variants']['persistence']['label']} | "
        f"{metrics['variants']['persistence']['decoded_prediction_mse']:.4f} | - | - |"
    )
    return "\n".join(lines) + "\n"


def _build_vei_demo_summary(
    steps: list[dict[str, Any]],
    selected_columns: list[str],
    schema: DatasetSchema,
    run_surface_summary: dict[str, object] | None,
) -> str:
    lines = [
        "# VEI Demo Summary",
        "",
        (
            "This report shows what the model believed would happen next "
            "after a short VEI state history."
        ),
        "",
        "## Decoded State Signals",
        "",
    ]
    lines.extend([f"- {label}" for label in _display_labels(schema, selected_columns)])
    if run_surface_summary is not None:
        lines.extend(
            [
                "",
                "## Run Surface Context",
                "",
                f"- Company: `{run_surface_summary.get('company_name', '')}`",
                f"- Vertical: `{run_surface_summary.get('vertical_name', '')}`",
                f"- Current tension: `{run_surface_summary.get('current_tension', '')}`",
                (
                    "- Panel counts: "
                    f"`{run_surface_summary.get('panel_count', 0)}` panels, "
                    f"`{run_surface_summary.get('item_count', 0)}` items"
                ),
                (
                    "- Status mix: "
                    f"`ok={run_surface_summary.get('ok_count', 0)}`, "
                    f"`attention={run_surface_summary.get('attention_count', 0)}`, "
                    f"`warning={run_surface_summary.get('warning_count', 0)}`, "
                    f"`critical={run_surface_summary.get('critical_count', 0)}`"
                ),
            ]
        )
        panel_titles = run_surface_summary.get("panel_titles", [])
        if isinstance(panel_titles, list) and panel_titles:
            lines.append("- Panels: " + ", ".join(f"`{str(title)}`" for title in panel_titles[:6]))
    for index, step in enumerate(steps, start=1):
        lines.extend(
            [
                "",
                f"## Step {index}",
                "",
                f"- Current step: `{step['current_step_idx']}`",
                f"- Action taken: `{step['action']['action_name']}`",
                f"- Surprise score: `{step['surprise_score']:.4f}`",
                "",
                "Current state:",
            ]
        )
        lines.extend([f"- {key}: {value}" for key, value in step["current_state_summary"].items()])
        lines.append("")
        lines.append("Predicted next state:")
        lines.extend(
            [f"- {key}: {value}" for key, value in step["predicted_next_state_summary"].items()]
        )
        lines.append("")
        lines.append("Actual next state:")
        lines.extend(
            [f"- {key}: {value}" for key, value in step["actual_next_state_summary"].items()]
        )
        actual_change_summary = step.get("actual_change_summary")
        if isinstance(actual_change_summary, dict):
            lines.extend(
                [
                    "",
                    "Observed VEI changes:",
                    (
                        f"- Changed: `{actual_change_summary.get('changed_count', 0)}`, "
                        f"added: `{actual_change_summary.get('added_count', 0)}`, "
                        f"removed: `{actual_change_summary.get('removed_count', 0)}`"
                    ),
                ]
            )
            top_changes = actual_change_summary.get("top_changes", [])
            if isinstance(top_changes, list):
                lines.extend(f"- {change}" for change in top_changes[:6])
    return "\n".join(lines) + "\n"


def _build_brief(metrics: dict[str, Any], vei_steps: list[dict[str, Any]]) -> str:
    best_variant = metrics["best_action_aware_variant"]
    best_metrics = metrics["variants"][best_variant]
    persistence = metrics["variants"]["persistence"]["decoded_prediction_mse"]
    no_actions = metrics["variants"]["flat_no_actions"]["decoded_prediction_mse"]

    lines = [
        "# E-JEPA Brief",
        "",
        "## What this is",
        "",
        (
            "E-JEPA learns how enterprise state changes after decisions, "
            "using structured business state instead of pixels."
        ),
        "",
        "## Why this is interesting",
        "",
        (
            "It gives us one model that can learn state transitions, react to actions, "
            "and flag surprising changes in enterprise workflows."
        ),
        "",
        "## What it predicts",
        "",
        (
            "It predicts the next enterprise state summary from recent state and action "
            "history. Today that means things like business measures, VEI graph summaries, "
            "pending counts, and workflow state features."
        ),
        "",
        "## Evidence it learns useful dynamics",
        "",
        f"- Best action-aware variant: `{best_metrics['label']}`",
        (
            "- Normalized next-state error for best action-aware variant: "
            f"`{best_metrics['decoded_prediction_mse']:.4f}`"
        ),
        f"- Same benchmark with actions removed: `{no_actions:.4f}`",
        f"- Persistence baseline: `{persistence:.4f}`",
        f"- Surprise on clean transitions: `{best_metrics['surprise_clean']:.4f}`",
        f"- Surprise on perturbed transitions: `{best_metrics['surprise_perturbed']:.4f}`",
        "",
        "## What it does not claim yet",
        "",
        "- It is not yet a planner.",
        "- It does not predict the next action.",
        "- It is not yet a large real-enterprise benchmark result.",
    ]

    if vei_steps:
        first_step = vei_steps[0]
        lines.extend(
            [
                "",
                "## VEI Demo Snapshot",
                "",
                f"- Example action: `{first_step['action']['action_name']}`",
                f"- Example surprise score: `{first_step['surprise_score']:.4f}`",
                (
                    "- Predicted next state fields shown: "
                    f"`{len(first_step['predicted_next_state_summary'])}`"
                ),
            ]
        )

    return "\n".join(lines) + "\n"


def _write_simple_bar_chart(
    path: Path,
    *,
    title: str,
    values: dict[str, float],
    subtitle: str,
) -> None:
    font_family = "Helvetica, Arial, sans-serif"
    width = 860
    height = 420
    margin_left = 90
    margin_bottom = 90
    margin_top = 70
    chart_height = height - margin_top - margin_bottom
    chart_width = width - margin_left - 40
    max_value = max(values.values()) if values else 1.0
    scale = chart_height / max(max_value, 1e-6)
    bar_width = chart_width / max(len(values) * 1.5, 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white" />',
        (
            f'<text x="{margin_left}" y="32" font-size="24" '
            f'font-family="{font_family}">{title}</text>'
        ),
        (
            f'<text x="{margin_left}" y="56" font-size="14" fill="#555" '
            f'font-family="{font_family}">{subtitle}</text>'
        ),
        (
            f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
            f'x2="{width - 20}" y2="{height - margin_bottom}" stroke="#222" />'
        ),
        (
            f'<line x1="{margin_left}" y1="{margin_top}" '
            f'x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222" />'
        ),
    ]

    for index, (label, value) in enumerate(values.items()):
        x = margin_left + (index * bar_width * 1.5) + 30
        bar_height = value * scale
        y = height - margin_bottom - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="#2b6cb0" />'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" '
            f'text-anchor="middle" font-size="12" font-family="{font_family}">'
            f"{value:.3f}</text>"
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 18:.1f}" '
            f'text-anchor="middle" font-size="12" font-family="{font_family}">'
            f"{label}</text>"
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
