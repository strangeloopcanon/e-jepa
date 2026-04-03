from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from structured_jepa import vei_support


def test_load_run_timeline_from_vei_api_can_use_sdk_fallback(
    monkeypatch,
) -> None:
    def fake_import(*candidates: tuple[str, str]):
        if ("vei.sdk", "load_run_events_entry") in candidates:
            return lambda workspace_root, run_id: [
                {"kind": "workflow_step", "resolved_tool": "jira.transition_issue", "time_ms": 10}
            ]
        return None

    monkeypatch.setattr(vei_support, "_import_vei_callable", fake_import)

    events = vei_support._load_run_timeline_from_vei_api(Path("/tmp/workspace"), "run-1")

    assert events is not None
    assert events[0]["resolved_tool"] == "jira.transition_issue"


def test_load_run_snapshots_from_vei_api_can_read_snapshot_paths_from_refs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot_path = tmp_path / "runs" / "run-1" / "state" / "main" / "snapshots" / "0001.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "snapshot_id": 1,
                "clock_ms": 1000,
                "data": {"pending_events": [{"target": "work_management"}]},
            }
        ),
        encoding="utf-8",
    )

    def fake_import(*candidates: tuple[str, str]):
        if ("vei.run.api", "list_run_snapshots") in candidates:
            return lambda workspace_root, run_id: [
                SimpleNamespace(
                    snapshot_id=1,
                    branch="main",
                    label="first",
                    time_ms=1000,
                    path=snapshot_path.relative_to(tmp_path),
                )
            ]
        return None

    monkeypatch.setattr(vei_support, "_import_vei_callable", fake_import)

    snapshots = vei_support._load_run_snapshots_from_vei_api(tmp_path, "run-1")

    assert snapshots is not None
    assert snapshots[0]["snapshot_id"] == 1
    assert snapshots[0]["clock_ms"] == 1000


def test_load_snapshot_diff_from_vei_api_can_use_keyword_only_sdk_entry(
    monkeypatch,
) -> None:
    def sdk_diff(root, run_id, *, snapshot_from, snapshot_to):
        return {
            "from": snapshot_from,
            "to": snapshot_to,
            "added": {},
            "removed": {},
            "changed": {"pending_events": {"from": 1, "to": 0}},
        }

    def fake_import(*candidates: tuple[str, str]):
        if ("vei.sdk", "diff_run_snapshots_entry") in candidates:
            return sdk_diff
        return None

    monkeypatch.setattr(vei_support, "_import_vei_callable", fake_import)

    diff_payload = vei_support._load_snapshot_diff_from_vei_api(
        workspace_root=Path("/tmp/workspace"),
        run_id="run-1",
        snapshot_from=1,
        snapshot_to=2,
    )

    assert diff_payload is not None
    assert diff_payload["changed"]["pending_events"]["to"] == 0
