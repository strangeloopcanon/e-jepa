from __future__ import annotations

import glob
import json

import typer

from .api import (
    evaluate_model,
    fit_linear_probe,
    fit_summary_decoder,
    prepare_timeseries_dataset,
    prepare_vei_context_dataset,
    prepare_vei_runs_dataset,
    train_model,
)
from .schema import ModelConfig, TrainConfig

app = typer.Typer(add_completion=False)
RUN_ID_OPTION = typer.Option(None, "--run-id", help="Specific run id(s) to include")


@app.command("prepare-timeseries")
def prepare_timeseries_command(
    input_path: str = typer.Option(..., "--input", help="CSV or Parquet file"),
    output_dir: str = typer.Option(..., "--output", help="Prepared dataset directory"),
    entity_column: str = typer.Option(..., "--entity-column", help="Entity identifier column"),
    timestamp_column: str = typer.Option(..., "--timestamp-column", help="Timestamp column"),
    observation_categorical_columns: str = typer.Option(
        "",
        "--observation-categorical-columns",
        help="Comma-separated categorical observation columns",
    ),
    action_numeric_columns: str = typer.Option(
        "", "--action-numeric-columns", help="Comma-separated numeric action columns"
    ),
    action_categorical_columns: str = typer.Option(
        "", "--action-categorical-columns", help="Comma-separated categorical action columns"
    ),
    auxiliary_numeric_targets: str = typer.Option(
        "", "--auxiliary-numeric-targets", help="Comma-separated auxiliary numeric target columns"
    ),
    seed: int = typer.Option(7, "--seed", help="Split seed"),
    train_fraction: float = typer.Option(0.7, "--train-fraction", help="Train split fraction"),
    val_fraction: float = typer.Option(0.15, "--val-fraction", help="Validation split fraction"),
) -> None:
    prepared = prepare_timeseries_dataset(
        input_path=input_path,
        output_dir=output_dir,
        entity_column=entity_column,
        timestamp_column=timestamp_column,
        observation_categorical_columns=_split_csv_values(observation_categorical_columns),
        action_numeric_columns=_split_csv_values(action_numeric_columns),
        action_categorical_columns=_split_csv_values(action_categorical_columns),
        auxiliary_numeric_target_columns=_split_csv_values(auxiliary_numeric_targets),
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    typer.echo(f"Wrote {prepared.schema.row_count} rows to {prepared.root}")


@app.command("prepare-vei-runs")
def prepare_vei_runs_command(
    workspace_root: str = typer.Option(..., "--workspace-root", help="VEI workspace root"),
    output_dir: str = typer.Option(..., "--output", help="Prepared dataset directory"),
    run_id: list[str] = RUN_ID_OPTION,
    seed: int = typer.Option(7, "--seed", help="Split seed"),
    train_fraction: float = typer.Option(0.7, "--train-fraction", help="Train split fraction"),
    val_fraction: float = typer.Option(0.15, "--val-fraction", help="Validation split fraction"),
) -> None:
    prepared = prepare_vei_runs_dataset(
        workspace_root=workspace_root,
        output_dir=output_dir,
        run_ids=run_id or None,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    typer.echo(f"Wrote {prepared.schema.row_count} VEI run rows to {prepared.root}")


@app.command("prepare-vei-context")
def prepare_vei_context_command(
    snapshot_glob: str = typer.Option(
        ..., "--snapshot-glob", help="Glob for context snapshot files"
    ),
    output_dir: str = typer.Option(..., "--output", help="Prepared dataset directory"),
    diff_glob: str = typer.Option("", "--diff-glob", help="Optional glob for context diff files"),
    seed: int = typer.Option(7, "--seed", help="Split seed"),
    train_fraction: float = typer.Option(0.7, "--train-fraction", help="Train split fraction"),
    val_fraction: float = typer.Option(0.15, "--val-fraction", help="Validation split fraction"),
) -> None:
    snapshot_paths = sorted(glob.glob(snapshot_glob))
    if not snapshot_paths:
        raise typer.BadParameter(f"no snapshots matched: {snapshot_glob}")
    diff_paths = sorted(glob.glob(diff_glob)) if diff_glob else []
    prepared = prepare_vei_context_dataset(
        snapshot_paths=snapshot_paths,
        diff_paths=diff_paths,
        output_dir=output_dir,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    typer.echo(f"Wrote {prepared.schema.row_count} VEI context rows to {prepared.root}")


@app.command("train")
def train_command(
    dataset_root: str = typer.Option(..., "--dataset", help="Prepared dataset directory"),
    output_dir: str = typer.Option(..., "--output", help="Training output directory"),
    epochs: int = typer.Option(8, "--epochs", help="Epoch count"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    device: str = typer.Option("cpu", "--device", help="Training device"),
) -> None:
    artifacts = train_model(
        dataset_root=dataset_root,
        output_dir=output_dir,
        model_config=ModelConfig(),
        train_config=TrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        ),
    )
    typer.echo(f"Model saved to {artifacts.model_path}")


@app.command("evaluate")
def evaluate_command(
    dataset_root: str = typer.Option(..., "--dataset", help="Prepared dataset directory"),
    checkpoint_path: str = typer.Option(..., "--checkpoint", help="Saved model checkpoint"),
    device: str = typer.Option("cpu", "--device", help="Evaluation device"),
) -> None:
    result = evaluate_model(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@app.command("probe")
def probe_command(
    dataset_root: str = typer.Option(..., "--dataset", help="Prepared dataset directory"),
    checkpoint_path: str = typer.Option(..., "--checkpoint", help="Saved model checkpoint"),
    target_column: str = typer.Option(..., "--target-column", help="Auxiliary target column"),
    device: str = typer.Option("cpu", "--device", help="Probe device"),
) -> None:
    result = fit_linear_probe(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        target_column=target_column,
        device=device,
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@app.command("decode-summary")
def decode_summary_command(
    dataset_root: str = typer.Option(..., "--dataset", help="Prepared dataset directory"),
    checkpoint_path: str = typer.Option(..., "--checkpoint", help="Saved model checkpoint"),
    columns: str = typer.Option("", "--columns", help="Comma-separated numeric columns to decode"),
    device: str = typer.Option("cpu", "--device", help="Decoder device"),
) -> None:
    result = fit_summary_decoder(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        columns=_split_csv_values(columns),
        device=device,
    )
    typer.echo(json.dumps(result, indent=2))


def _split_csv_values(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]
