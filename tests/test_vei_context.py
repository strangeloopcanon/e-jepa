from __future__ import annotations

import json
from pathlib import Path

from structured_jepa.storage import load_processed_dataset
from structured_jepa.vei_context import prepare_vei_context_dataset


def test_prepare_vei_context_dataset_extracts_provider_features(tmp_path: Path) -> None:
    snapshots = [
        {
            "organization_name": "Acme",
            "organization_domain": "acme.example.com",
            "captured_at": "2024-01-01T00:00:00Z",
            "sources": [
                {
                    "provider": "slack",
                    "status": "ok",
                    "data": {
                        "channels": [
                            {"channel": "#ops", "unread": 2, "messages": [{"text": "a"}]},
                        ],
                        "users": [
                            {"id": "U1", "deleted": False, "is_bot": False},
                            {"id": "U2", "deleted": True, "is_bot": False},
                        ],
                    },
                },
                {
                    "provider": "jira",
                    "status": "ok",
                    "data": {
                        "issues": [
                            {"status": "open", "priority": "high", "issue_type": "bug"},
                            {"status": "closed", "priority": "low", "issue_type": "task"},
                        ],
                        "projects": [{"key": "ACME"}],
                    },
                },
                {
                    "provider": "google",
                    "status": "ok",
                    "data": {
                        "users": [
                            {"id": "G1", "suspended": False, "is_admin": True, "org_unit": "/Eng"}
                        ],
                        "documents": [
                            {"doc_id": "D1", "shared": True},
                            {"doc_id": "D2", "shared": False},
                        ],
                    },
                },
                {
                    "provider": "okta",
                    "status": "ok",
                    "data": {
                        "users": [
                            {"id": "O1", "status": "ACTIVE"},
                            {"id": "O2", "status": "SUSPENDED"},
                        ],
                        "groups": [{"id": "G1"}],
                        "applications": [{"id": "A1"}],
                    },
                },
            ],
        },
        {
            "organization_name": "Acme",
            "organization_domain": "acme.example.com",
            "captured_at": "2024-01-02T00:00:00Z",
            "sources": [
                {
                    "provider": "slack",
                    "status": "ok",
                    "data": {
                        "channels": [
                            {
                                "channel": "#ops",
                                "unread": 1,
                                "messages": [{"text": "a"}, {"text": "b"}],
                            },
                        ],
                        "users": [{"id": "U1", "deleted": False, "is_bot": False}],
                    },
                },
                {
                    "provider": "jira",
                    "status": "ok",
                    "data": {
                        "issues": [{"status": "open", "priority": "high", "issue_type": "bug"}],
                        "projects": [{"key": "ACME"}],
                    },
                },
                {"provider": "google", "status": "ok", "data": {"users": [], "documents": []}},
                {
                    "provider": "okta",
                    "status": "ok",
                    "data": {"users": [], "groups": [], "applications": []},
                },
            ],
        },
    ]
    diffs = [
        {
            "after_captured_at": "2024-01-02T00:00:00Z",
            "entries": [
                {"kind": "changed", "domain": "issues", "item_id": "ACME-1"},
                {"kind": "removed", "domain": "users", "item_id": "O2"},
            ],
        }
    ]
    snapshot_paths = []
    diff_paths = []
    for index, payload in enumerate(snapshots, start=1):
        path = tmp_path / f"snapshot_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        snapshot_paths.append(path)
    for index, payload in enumerate(diffs, start=1):
        path = tmp_path / f"diff_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        diff_paths.append(path)

    prepared = prepare_vei_context_dataset(
        snapshot_paths=snapshot_paths,
        diff_paths=diff_paths,
        output_dir=tmp_path / "dataset",
    )
    loaded = load_processed_dataset(prepared.root)

    assert loaded.schema.dataset_kind == "vei_context"
    assert "obs_num__slack_unread_total" in loaded.frame.columns
    assert "obs_num__jira_status__open" in loaded.frame.columns
    assert "obs_num__okta_status__active" in loaded.frame.columns
    assert "aux_num__diff_changed_total" in loaded.frame.columns
    assert len(loaded.frame) == 2
