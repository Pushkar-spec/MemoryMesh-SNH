"""Unit and integration tests for GET /api/eval routes.

Governed by docs/SPECIFICATION.md Section 7.2.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


def test_get_latest_eval_summary(client: TestClient) -> None:
    """Test GET /api/eval/latest returns EvalSummary from results.json."""
    mock_results = Path("eval/results.json")
    if mock_results.is_file():
        res = client.get("/api/eval/latest")
        assert res.status_code == 200
        data = res.json()
        assert "run_id" in data
        assert "mode" in data
        assert "systems" in data
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)
        assert "scenarios" in data
        assert "questions" in data


def test_get_latest_eval_not_found(client: TestClient) -> None:
    """Test GET /api/eval/latest returns 404 when results file is missing."""
    with patch(
        "memorymesh.api.routers.eval.get_results_path",
        return_value=Path("nonexistent.json"),
    ):
        res = client.get("/api/eval/latest")
        assert res.status_code == 404
        assert res.json()["error"]["code"] == "not_found"


def test_get_latest_eval_report(client: TestClient) -> None:
    """Test GET /api/eval/latest/report returns Markdown content."""
    mock_report = Path("eval/report.md")
    if mock_report.is_file():
        res = client.get("/api/eval/latest/report")
        assert res.status_code == 200
        assert "text/markdown" in res.headers["content-type"]
        assert len(res.text) > 0


def test_get_latest_eval_report_not_found(client: TestClient) -> None:
    """Test GET /api/eval/latest/report returns 404 when report file is missing."""
    with patch("memorymesh.api.routers.eval.get_report_path", return_value=Path("nonexistent.md")):
        res = client.get("/api/eval/latest/report")
        assert res.status_code == 404
        assert res.json()["error"]["code"] == "not_found"


def test_get_latest_eval_malformed_json(client: TestClient, tmp_path: Path) -> None:
    """Test GET /api/eval/latest returns 500 when results file is malformed JSON."""
    bad_file = tmp_path / "bad_results.json"
    bad_file.write_text("{ this is not valid json }", encoding="utf-8")
    with patch("memorymesh.api.routers.eval.get_results_path", return_value=bad_file):
        res = client.get("/api/eval/latest")
        assert res.status_code == 500
        data = res.json()
        assert data["error"]["code"] == "internal"
        # Ensure no filesystem path leaked
        assert str(bad_file) not in data["error"]["message"]


def test_get_latest_eval_report_unreadable(client: TestClient, tmp_path: Path) -> None:
    """Test GET /api/eval/latest/report returns 500 when report file read fails."""
    report_file = tmp_path / "report.md"
    report_file.write_text("Valid markdown", encoding="utf-8")
    with (
        patch("memorymesh.api.routers.eval.get_report_path", return_value=report_file),
        patch.object(Path, "read_text", side_effect=OSError("Disk I/O error")),
    ):
        res = client.get("/api/eval/latest/report")
        assert res.status_code == 500
        data = res.json()
        assert data["error"]["code"] == "internal"
