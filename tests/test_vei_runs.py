from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from structured_jepa.storage import load_processed_dataset
from structured_jepa.vei_runs import prepare_vei_runs_dataset
from structured_jepa.vei_surface_features import summarize_snapshot_surface_features


def _enable_vei_imports() -> bool:
    repo_root_env = os.environ.get("VEI_REPO_ROOT")
    if repo_root_env:
        repo_root = Path(repo_root_env)
        if repo_root.exists():
            repo_path = str(repo_root)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
    return importlib.util.find_spec("vei") is not None


@pytest.mark.skipif(not _enable_vei_imports(), reason="VEI package not available")
def test_prepare_vei_runs_dataset_from_real_workflow_run(tmp_path: Path) -> None:
    from vei.run.api import launch_workspace_run
    from vei.workspace.api import create_workspace_from_template

    workspace_root = tmp_path / "workspace"
    create_workspace_from_template(
        root=workspace_root,
        source_kind="example",
        source_ref="acquired_user_cutover",
    )
    manifest = launch_workspace_run(workspace_root, runner="workflow")

    prepared = prepare_vei_runs_dataset(
        workspace_root=workspace_root,
        output_dir=tmp_path / "dataset",
        run_ids=[manifest.run_id],
    )
    loaded = load_processed_dataset(prepared.root)

    assert loaded.schema.dataset_kind == "vei_runs"
    assert len(loaded.frame) > 0
    assert "obs_num__graph_count__identity_graph__users" in loaded.frame.columns or any(
        column.startswith("obs_num__graph_count__") for column in loaded.frame.columns
    )
    assert any(column.startswith("obs_num__orientation_count__") for column in loaded.frame.columns)
    assert any(value != "__none__" for value in loaded.frame["action_name"].astype(str))


def test_prepare_vei_runs_dataset_from_playable_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "playable"
    run_id = _write_playable_run_fixture(workspace_root)

    prepared = prepare_vei_runs_dataset(
        workspace_root=workspace_root,
        output_dir=tmp_path / "dataset",
        run_ids=[run_id],
    )
    loaded = load_processed_dataset(prepared.root)

    assert len(loaded.frame) > 0
    assert "meta__mission_name" in loaded.frame.columns
    assert "meta__snapshot_id" in loaded.frame.columns
    assert "meta__surface_panel_count" in loaded.frame.columns
    assert "obs_num__surface_panel_count__all" in loaded.frame.columns
    assert "obs_num__surface_signal__slack_unread" in loaded.frame.columns
    assert "obs_cat__surface_primary_panel" in loaded.frame.columns
    assert any(
        column.startswith("obs_num__interval_event_count__") for column in loaded.frame.columns
    )
    assert loaded.schema.notes["workspace_root"] == str(workspace_root.resolve())
    assert loaded.frame["obs_cat__surface_primary_panel"].astype(str).ne("__missing__").any()


def test_prepare_vei_runs_dataset_can_use_public_api_helpers_without_snapshot_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    run_id = "run-from-public-api"
    run_root = workspace_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "runner": "workflow",
                "scenario_name": "api_path_demo",
                "branch": "main",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "structured_jepa.vei_runs.load_run_timeline",
        lambda **_: (
            [
                {
                    "time_ms": 1200,
                    "kind": "workflow_step",
                    "resolved_tool": "jira.transition_issue",
                    "graph_domain": "work_management",
                    "graph_action": "update",
                    "object_refs": ["ticket:case-1"],
                }
            ],
            True,
        ),
    )
    monkeypatch.setattr(
        "structured_jepa.vei_runs.load_run_snapshots",
        lambda **_: (
            [
                {
                    "snapshot_id": 11,
                    "label": "before",
                    "clock_ms": 1000,
                    "data": {
                        "pending_events": [{"target": "work_management"}],
                        "components": {
                            "slack": {
                                "channels": {
                                    "general": {
                                        "unread": 2,
                                        "messages": [
                                            {"ts": "1000.1", "user": "ops", "text": "Heads up"}
                                        ],
                                    }
                                }
                            },
                            "tickets": {
                                "tickets": {
                                    "ticket-1": {
                                        "ticket_id": "ticket-1",
                                        "status": "open",
                                        "title": "Launch issue",
                                    }
                                }
                            },
                            "servicedesk": {
                                "requests": {
                                    "req-1": {
                                        "request_id": "req-1",
                                        "status": "pending_approval",
                                        "approvals": [{"status": "PENDING"}],
                                    }
                                }
                            },
                            "identity_graph": {"users": {"user-1": {"status": "active"}}},
                        },
                    },
                },
                {
                    "snapshot_id": 12,
                    "label": "after",
                    "clock_ms": 2000,
                    "data": {
                        "pending_events": [],
                        "components": {
                            "slack": {
                                "channels": {
                                    "general": {
                                        "unread": 0,
                                        "messages": [
                                            {"ts": "2000.1", "user": "ops", "text": "Resolved"}
                                        ],
                                    }
                                }
                            },
                            "tickets": {
                                "tickets": {
                                    "ticket-1": {
                                        "ticket_id": "ticket-1",
                                        "status": "closed",
                                        "title": "Launch issue",
                                    }
                                }
                            },
                            "servicedesk": {
                                "requests": {
                                    "req-1": {
                                        "request_id": "req-1",
                                        "status": "approved",
                                        "approvals": [{"status": "APPROVED"}],
                                    }
                                }
                            },
                            "identity_graph": {"users": {"user-1": {"status": "active"}}},
                        },
                    },
                },
            ],
            True,
        ),
    )
    monkeypatch.setattr(
        "structured_jepa.vei_runs.load_run_surface_summary",
        lambda **_: (
            {
                "company_name": "Acme Holdings",
                "vertical_name": "workspace",
                "current_tension": "Closing a customer issue before launch.",
                "panel_titles": ["Work Tracker", "Team Chat"],
                "panel_count": 2,
                "item_count": 5,
                "ok_count": 1,
                "attention_count": 1,
                "warning_count": 0,
                "critical_count": 0,
            },
            True,
        ),
    )

    prepared = prepare_vei_runs_dataset(
        workspace_root=workspace_root,
        output_dir=tmp_path / "dataset",
        run_ids=[run_id],
    )
    loaded = load_processed_dataset(prepared.root)

    assert len(loaded.frame) == 2
    assert loaded.schema.notes["vei_public_snapshot_api_used"] is True
    assert list(loaded.frame["meta__snapshot_id"]) == [11, 12]
    assert loaded.frame["meta__surface_company_name"].tolist() == [
        "Acme Holdings",
        "Acme Holdings",
    ]
    assert "obs_num__surface_signal__slack_unread" in loaded.frame.columns
    assert "obs_cat__surface_primary_status" in loaded.frame.columns
    assert loaded.frame["action_name"].tolist()[0] == "jira.transition_issue"
    assert loaded.frame["action_name"].tolist()[1] == "__none__"


def test_surface_feature_summary_extracts_panel_counts_and_priority() -> None:
    numeric, categorical = summarize_snapshot_surface_features(
        {
            "components": {
                "slack": {
                    "channels": {
                        "general": {
                            "unread": 3,
                            "messages": [
                                {"ts": "1.0", "user": "ops", "text": "Update 1"},
                                {"ts": "2.0", "user": "ops", "text": "Update 2"},
                            ],
                        }
                    }
                },
                "tickets": {
                    "tickets": {
                        "ticket-1": {"ticket_id": "ticket-1", "status": "open"},
                        "ticket-2": {"ticket_id": "ticket-2", "status": "closed"},
                    }
                },
                "servicedesk": {
                    "requests": {
                        "req-1": {
                            "request_id": "req-1",
                            "status": "pending_approval",
                            "approvals": [{"status": "PENDING"}],
                        }
                    }
                },
                "inventory_ops": {
                    "capacity_pools": {
                        "pool-1": {
                            "total_units": 100,
                            "reserved_units": 96,
                        }
                    },
                    "quotes": {"quote-1": {"status": "open"}},
                },
            }
        }
    )

    assert numeric["surface_panel_count__all"] == 4.0
    assert numeric["surface_signal__slack_unread"] == 3.0
    assert numeric["surface_signal__approvals_pending"] == 2.0
    assert numeric["surface_signal__inventory_low_headroom_critical"] == 1.0
    assert categorical["surface_primary_panel"] == "vertical_inventory"
    assert categorical["surface_primary_status"] == "critical"


def test_surface_feature_summary_supports_service_ops_vertical() -> None:
    numeric, categorical = summarize_snapshot_surface_features(
        {
            "components": {
                "service_ops": {
                    "work_orders": {
                        "wo-1": {"status": "dispatched"},
                        "wo-2": {"status": "pending"},
                    },
                    "appointments": {
                        "appt-1": {"dispatch_status": "assigned"},
                        "appt-2": {"dispatch_status": "pending"},
                    },
                    "billing_cases": {
                        "bill-1": {"hold": True, "dispute_status": "open"},
                    },
                    "exceptions": {
                        "exc-1": {"severity": "high", "status": "open"},
                    },
                }
            }
        }
    )

    assert numeric["surface_item_count__service_ops_work_orders"] == 2.0
    assert numeric["surface_item_count__service_ops_appointments"] == 2.0
    assert numeric["surface_signal__service_ops_assigned_appointments"] == 1.0
    assert numeric["surface_signal__service_ops_holds_or_disputes"] == 1.0
    assert numeric["surface_signal__service_ops_critical_exceptions"] == 1.0
    assert categorical["surface_primary_panel"] == "vertical_service_ops"
    assert categorical["surface_primary_status"] == "critical"


def _write_playable_run_fixture(workspace_root: Path) -> str:
    run_id = "playable-run-1"
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
                    "branch_labels": ["initial", "resolved"],
                },
                "scorecard": {"overall_score": 0.75},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "time_ms": 1500,
                        "kind": "tool_call",
                        "resolved_tool": "slack.send_message",
                        "graph_domain": "communications",
                        "graph_action": "notify",
                        "object_refs": ["slack:channel/general", "user:tenant"],
                    }
                ),
                json.dumps(
                    {
                        "time_ms": 1800,
                        "kind": "status_update",
                        "object_refs": ["ticket:lease-42"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    snapshot_payloads = [
        {
            "branch": "player",
            "clock_ms": 1000,
            "data": {
                "pending_events": [{"target": "communications"}],
                "components": {
                    "slack": {
                        "channels": {
                            "general": {
                                "unread": 2,
                                "messages": [
                                    {
                                        "ts": "1000.1",
                                        "user": "agent",
                                        "text": "Tenant raised a conflict",
                                    }
                                ],
                            }
                        }
                    },
                    "tickets": {
                        "tickets": {
                            "ticket-1": {
                                "ticket_id": "ticket-1",
                                "status": "open",
                                "title": "Lease handoff",
                            }
                        }
                    },
                    "servicedesk": {
                        "requests": {
                            "req-1": {
                                "request_id": "req-1",
                                "status": "pending_approval",
                                "approvals": [{"status": "PENDING"}],
                            }
                        }
                    },
                    "identity_graph": {
                        "users": {
                            "user-1": {"status": "active"},
                            "user-2": {"status": "pending"},
                        }
                    },
                },
            },
        },
        {
            "branch": "player",
            "clock_ms": 2000,
            "data": {
                "pending_events": [],
                "components": {
                    "slack": {
                        "channels": {
                            "general": {
                                "unread": 0,
                                "messages": [
                                    {
                                        "ts": "2000.1",
                                        "user": "agent",
                                        "text": "Conflict resolved",
                                    }
                                ],
                            }
                        }
                    },
                    "tickets": {
                        "tickets": {
                            "ticket-1": {
                                "ticket_id": "ticket-1",
                                "status": "closed",
                                "title": "Lease handoff",
                            }
                        }
                    },
                    "servicedesk": {
                        "requests": {
                            "req-1": {
                                "request_id": "req-1",
                                "status": "approved",
                                "approvals": [{"status": "APPROVED"}],
                            }
                        }
                    },
                    "identity_graph": {
                        "users": {
                            "user-1": {"status": "active"},
                            "user-2": {"status": "active"},
                        }
                    },
                },
            },
        },
    ]
    for index, payload in enumerate(snapshot_payloads, start=1):
        (snapshot_root / f"{index:04d}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    return run_id
