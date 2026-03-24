from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from structured_jepa.cli import app
from structured_jepa.timeseries import prepare_timeseries_dataset
from structured_jepa.training import train_model
from structured_jepa.vei_runs import prepare_vei_runs_dataset

runner = CliRunner()


def test_benchmark_timeseries_command_writes_report_and_all_variants(tmp_path: Path) -> None:
    input_path = _write_benchmark_timeseries_csv(tmp_path)
    prepared = prepare_timeseries_dataset(
        input_path=input_path,
        output_dir=tmp_path / "dataset",
        entity_column="entity_id",
        timestamp_column="event_ts",
        observation_categorical_columns=["team", "segment"],
        action_numeric_columns=["control"],
        action_categorical_columns=["campaign"],
        auxiliary_numeric_target_columns=["target_backlog"],
        seed=7,
    )

    result = runner.invoke(
        app,
        [
            "benchmark-timeseries",
            "--dataset",
            str(prepared.root),
            "--output",
            str(tmp_path / "benchmark"),
            "--epochs",
            "5",
            "--batch-size",
            "10",
        ],
    )

    assert result.exit_code == 0, result.stdout
    benchmark_root = tmp_path / "benchmark"
    assert (benchmark_root / "metrics.json").exists()
    assert (benchmark_root / "summary.md").exists()
    assert (benchmark_root / "prediction_quality.svg").exists()
    assert (benchmark_root / "surprise_separation.svg").exists()

    metrics = json.loads((benchmark_root / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["variants"]) == {"flat", "tokenized", "flat_no_actions", "persistence"}
    assert metrics["action_aware_beats_persistence"] is True


def test_benchmark_vei_demo_command_writes_readable_bundle_with_multiple_columns(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    run_id = _write_long_playable_run_fixture(workspace_root)
    prepared = prepare_vei_runs_dataset(
        workspace_root=workspace_root,
        output_dir=tmp_path / "dataset",
        run_ids=[run_id],
    )
    artifacts = train_model(
        dataset_root=prepared.root,
        output_dir=tmp_path / "model",
    )

    result = runner.invoke(
        app,
        [
            "benchmark-vei-demo",
            "--dataset",
            str(prepared.root),
            "--checkpoint",
            artifacts.model_path,
            "--output",
            str(tmp_path / "demo"),
            "--max-steps",
            "3",
        ],
    )

    assert result.exit_code == 0, result.stdout
    demo_root = tmp_path / "demo"
    assert (demo_root / "demo_steps.json").exists()
    assert (demo_root / "decoded_summary.json").exists()
    assert (demo_root / "summary.md").exists()

    steps = json.loads((demo_root / "demo_steps.json").read_text(encoding="utf-8"))
    decoded = json.loads((demo_root / "decoded_summary.json").read_text(encoding="utf-8"))
    assert len(steps) == 3
    assert len(decoded["selected_columns"]) > 1
    first = steps[0]
    assert "action" in first
    assert "current_state_summary" in first
    assert "predicted_next_state_summary" in first
    assert "actual_next_state_summary" in first
    assert "surprise_score" in first
    assert len(first["predicted_next_state_summary"]) > 1
    assert set(first["predicted_next_state_summary"]) == set(first["actual_next_state_summary"])


def test_write_brief_command_uses_generated_artifacts(tmp_path: Path) -> None:
    input_path = _write_benchmark_timeseries_csv(tmp_path)
    prepared = prepare_timeseries_dataset(
        input_path=input_path,
        output_dir=tmp_path / "dataset",
        entity_column="entity_id",
        timestamp_column="event_ts",
        observation_categorical_columns=["team", "segment"],
        action_numeric_columns=["control"],
        action_categorical_columns=["campaign"],
        auxiliary_numeric_target_columns=["target_backlog"],
        seed=7,
    )
    benchmark_result = runner.invoke(
        app,
        [
            "benchmark-timeseries",
            "--dataset",
            str(prepared.root),
            "--output",
            str(tmp_path / "benchmark"),
            "--epochs",
            "4",
            "--batch-size",
            "10",
        ],
    )
    assert benchmark_result.exit_code == 0, benchmark_result.stdout

    brief_path = tmp_path / "show_yann.md"
    result = runner.invoke(
        app,
        [
            "write-brief",
            "--benchmark-dir",
            str(tmp_path / "benchmark"),
            "--output",
            str(brief_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    brief = brief_path.read_text(encoding="utf-8")
    assert "What this is" in brief
    assert "Evidence it learns useful dynamics" in brief
    assert "What it does not claim yet" in brief


def _write_benchmark_timeseries_csv(tmp_path: Path) -> Path:
    rows = []
    for episode_index in range(12):
        team = ["sales", "support", "infra"][episode_index % 3]
        segment = "enterprise" if episode_index % 2 else "smb"
        team_bias = {"sales": 2.0, "support": 5.0, "infra": 8.0}[team]
        segment_bias = {"enterprise": 4.0, "smb": -1.0}[segment]
        for step_index in range(28):
            control = float((step_index + episode_index) % 5 in {0, 1})
            campaign = ["none", "reroute", "boost"][((step_index // 2) + episode_index) % 3]
            campaign_lift = {"none": 0.0, "reroute": 2.5, "boost": 5.0}[campaign]
            queue_depth = (
                team_bias + segment_bias + step_index * 0.3 + control * 3.5 + campaign_lift
            )
            backlog = queue_depth * 1.4 + (2.0 if segment == "enterprise" else -0.5)
            rows.append(
                {
                    "entity_id": f"acct-{episode_index}",
                    "event_ts": f"2024-07-{(step_index % 9) + 1:02d}T00:00:00Z",
                    "queue_depth": queue_depth,
                    "backlog": backlog,
                    "team": team,
                    "segment": segment,
                    "control": control,
                    "campaign": campaign,
                    "target_backlog": backlog + 1.5,
                }
            )
    input_path = tmp_path / "benchmark.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    return input_path


def _write_long_playable_run_fixture(workspace_root: Path) -> str:
    run_id = "playable-run-demo"
    run_root = workspace_root / "runs" / run_id
    snapshot_root = run_root / "state" / "player" / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "runner": "playable",
                "scenario_name": "tenant_opening_conflict",
                "branch": "player",
            }
        ),
        encoding="utf-8",
    )
    (run_root / "mission_state.json").write_text(
        json.dumps(
            {
                "objective_variant": "opening_conflict",
                "mission": {
                    "mission_name": "tenant_opening_conflict",
                    "branch_labels": ["initial", "midpoint", "resolved"],
                },
                "scorecard": {"overall_score": 0.82},
            }
        ),
        encoding="utf-8",
    )

    event_lines = []
    for step_index in range(1, 25):
        event_lines.append(
            json.dumps(
                {
                    "time_ms": step_index * 1000 + 250,
                    "kind": "tool_call",
                    "resolved_tool": "slack.send_message"
                    if step_index % 2
                    else "jira.transition_issue",
                    "graph_domain": "communications" if step_index % 2 else "work_management",
                    "graph_action": "notify" if step_index % 2 else "update",
                    "object_refs": [
                        "slack:channel/general",
                        f"ticket:case-{step_index}",
                    ],
                }
            )
        )
    (run_root / "events.jsonl").write_text("\n".join(event_lines), encoding="utf-8")

    for step_index in range(24):
        snapshot_payload = {
            "branch": "player",
            "clock_ms": (step_index + 1) * 1000,
            "data": {
                "pending_events": [
                    {"target": "communications"} for _ in range(max(0, 3 - (step_index % 4)))
                ],
                "components": {
                    "identity_graph": {
                        "users": {
                            "user-1": {"status": "active"},
                            "user-2": {"status": "active" if step_index > 8 else "pending"},
                            "user-3": {"status": "suspended" if step_index > 18 else "active"},
                        }
                    },
                    "communications": {
                        "tickets": {
                            "ticket-1": {"status": "closed" if step_index > 12 else "open"},
                            "ticket-2": {"status": "closed" if step_index > 20 else "open"},
                        },
                    },
                },
            },
        }
        (snapshot_root / f"{step_index + 1:04d}.json").write_text(
            json.dumps(snapshot_payload),
            encoding="utf-8",
        )
    return run_id
