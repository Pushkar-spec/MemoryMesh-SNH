"""Evaluation benchmarks and report API routes.

Governed by docs/SPECIFICATION.md Section 7.2.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response, status

from memorymesh.errors import MemoryNotFoundError
from memorymesh.models import EvalSummary

router = APIRouter(prefix="/eval", tags=["eval"])

EVAL_DIR = Path("eval")


def get_results_path() -> Path:
    """Resolve filesystem path to evaluation results JSON."""
    return EVAL_DIR / "results.json"


def get_report_path() -> Path:
    """Resolve filesystem path to evaluation Markdown report."""
    return EVAL_DIR / "report.md"


@router.get(
    "/latest",
    response_model=EvalSummary,
    status_code=status.HTTP_200_OK,
    summary="Get latest evaluation benchmark summary",
)
def get_latest_eval() -> EvalSummary:
    """Retrieve summary metrics and system comparison table from latest evaluation run."""
    results_file = get_results_path()
    if not results_file.is_file():
        raise MemoryNotFoundError("No evaluation results found.")

    try:
        content = results_file.read_text(encoding="utf-8")
        data = json.loads(content)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read evaluation results artifact.",
        ) from None

    raw_metrics = data.get("metrics", {})
    systems: list[str] = []
    metrics_table: dict[str, dict[str, float | None]] = {}

    for sys_key, sys_data in raw_metrics.items():
        if not isinstance(sys_data, dict):
            continue
        sys_id = str(sys_data.get("system_id") or sys_key)
        systems.append(sys_id)

        for metric_name, val in sys_data.items():
            if metric_name in ("system_id", "system_name", "retrieval_method"):
                continue
            if isinstance(val, (int, float)):
                metrics_table.setdefault(metric_name, {})[sys_id] = float(val)
            elif val is None:
                metrics_table.setdefault(metric_name, {})[sys_id] = None

    ts_raw = str(data.get("timestamp", ""))
    created_at = datetime.now(UTC)
    if ts_raw:
        try:
            created_at = datetime.strptime(ts_raw, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
        except Exception:
            pass

    return EvalSummary(
        run_id=data.get("run_id") or ts_raw or "latest",
        created_at=created_at,
        mode=data.get("mode", "all"),
        systems=systems,
        metrics=metrics_table,
        scenarios=int(data.get("scenario_count", len(data.get("scenarios", [])))),
        questions=int(data.get("question_count", 0)),
    )


@router.get(
    "/latest/report",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    summary="Get latest evaluation Markdown report",
)
def get_latest_eval_report() -> Response:
    """Retrieve raw Markdown content of the latest evaluation report."""
    report_file = get_report_path()
    if not report_file.is_file():
        raise MemoryNotFoundError("No evaluation report found.")

    try:
        content = report_file.read_text(encoding="utf-8")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read evaluation report artifact.",
        ) from None
    return Response(content=content, media_type="text/markdown")
