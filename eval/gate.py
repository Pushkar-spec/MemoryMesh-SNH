"""G2 Evaluation Gate implementation for MemoryMesh (P8.4).

Governed by:
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P8.4, Execution Gate G2)
- docs/ROADMAP_AND_TASKS.md (G2 Criteria, T8.4)
- docs/EVALUATION_AND_DEMO.md (Evaluation Objectives & Output)

Acceptance Criteria for G2 Gate:
1. G2-1: Artifact Generation (report.md and results.json generated and verified)
2. G2-2: Multi-Mode Evaluation (Structured, Engine-only, E2E, Baseline B0)
3. G2-3: Scenario Catalog Coverage (Target >= 40 sc, >= 100 q, all 10 categories)
4. G2-4: Write-Time Resolution Claim (MM outperforms Naive RAG on updates)
5. G2-5: Decision Auditability & Provenance (Audit logging; B0 is n/a)
6. G2-6: Designed-To-Lose Honesty (At least 3 designed-to-lose cases)
7. G2-7: Deterministic Reproducibility (Evaluation results repeatable across runs)
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from eval.metrics import EvaluationMetrics, QuestionEvalResult
from memorymesh.retrieval.retrieve import build_context_key, match_context_scope


def parse_iso_datetime(val: Any) -> datetime | None:
    """Safely parse an ISO datetime string or datetime instance into a UTC datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None or val.utcoffset() is None:
            return val.replace(tzinfo=UTC)
        return val.astimezone(UTC)
    s = str(val).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None or dt.utcoffset() is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


REQUIRED_REPORT_SECTIONS: list[str] = [
    "## A. Executive Summary",
    "## B. Evaluation Methodology",
    "## C. Quantitative Results",
    "## D. Scenario-Group Analysis",
    "## E. Failure Analysis",
    "## F. Limitations",
    "## G. Conclusion",
    "## H. G2 Evaluation Gate Verification",
]

REQUIRED_RESULTS_JSON_KEYS: list[str] = [
    "metrics",
    "scenarios",
    "g2_gate",
]

REQUIRED_CATEGORIES: dict[str, tuple[str, ...]] = {
    "simple_update": ("update-", "replace-"),
    "corroboration": ("corroborate-",),
    "low_trust_reject": ("reject-",),
    "dispute": ("dispute-",),
    "multi_valued": ("multivalue-",),
    "context_isolation": ("context-",),
    "out_of_order": ("ordering-",),
    "dependency_propagation": ("dependency-",),
    "historical_time_travel": ("temporal-",),
    "designed_to_lose": ("lose-",),
}

CANONICAL_MODES: tuple[str, ...] = ("structured", "engine-only", "e2e", "baseline")

CANONICAL_G2_CRITERIA: dict[str, str] = {
    "G2-1": "Artifact Generation",
    "G2-2": "Multi-Mode Comparative Evaluation",
    "G2-3": "Scenario Catalog Coverage",
    "G2-4": "Write-Time Resolution Superiority",
    "G2-5": "Decision Auditability and Provenance",
    "G2-6": "Designed-To-Lose Honesty & Failure Accounting",
    "G2-7": "Deterministic Reproducibility",
}


def validate_evaluation_metric(
    val: Any,
    name: str,
    *,
    allow_none: bool = False,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> tuple[bool, float | None, str | None]:
    """Validate metric is numeric, non-boolean, finite, and in [min_val, max_val].

    Returns:
        tuple of (is_valid: bool, numeric_value: float | None, error_message: str | None)
    """
    if val is None:
        if allow_none:
            return True, None, None
        return False, None, f"{name} is missing (None)"

    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False, None, f"{name} must be numeric, got {type(val).__name__} ({val!r})"

    try:
        f_val = float(val)
    except (ValueError, TypeError):
        return False, None, f"{name} cannot be converted to float: {val!r}"

    if not math.isfinite(f_val):
        return False, None, f"{name} must be finite, got {f_val}"

    if f_val < min_val or f_val > max_val:
        return False, None, f"{name} must be in range [{min_val}, {max_val}], got {f_val}"

    return True, f_val, None


def find_system_for_mode(m_name: str, keys: Any) -> str | None:
    """Find matching system key for canonical mode name."""
    matchers = {
        "structured": lambda k: "struct" in k.lower() or "mm (structured)" in k.lower(),
        "engine-only": lambda k: "engine-only" in k.lower() or "engine_only" in k.lower(),
        "e2e": lambda k: "e2e" in k.lower(),
        "baseline": lambda k: "b0" in k.lower() or "baseline" in k.lower(),
    }
    matcher = matchers.get(m_name)
    if not matcher:
        return None
    for k in keys:
        if matcher(k):
            return str(k)
    return None


def map_canonical_modes(system_keys: Any) -> dict[str, str]:
    """Map canonical mode names to the actual system keys present."""
    mapping: dict[str, str] = {}
    for m in CANONICAL_MODES:
        matched = find_system_for_mode(m, system_keys)
        if matched is not None:
            mapping[m] = matched
    return mapping


def serialize_markdown_table_cell(text: str) -> str:
    """Serialize arbitrary text for safe inclusion in a Markdown table cell.

    Escapes literal backslashes, pipe characters, and opening angle brackets (<)
    so they do not break table columns or conflict with newline markers.
    Actual newlines (\\r\\n, \\n, \\r) are encoded as '<br>' to preserve single-line
    row structure.

    Serialization contract:
    - Literal backslashes ('\\') are escaped as '\\\\'.
    - Literal pipe delimiters ('|') are escaped as '\\|'.
    - Literal opening angle brackets ('<') are escaped as '\\<' to distinguish
      literal HTML-like text (such as '<br>', '<br/>', '<br />', '<tag>') from
      newline-generated line break markers.
    - Actual line endings ('\\r\\n', '\\n', and '\\r') are serialized as '<br>'.
    - Deserialization restores all generated newline markers as '\\n'.
    - Therefore, round-trip equality is defined against newline-normalized input,
      not the original byte-for-byte string.
    """
    if not text:
        return ""
    # 1. Escape backslashes first so subsequent escapes are not re-escaped
    escaped = text.replace("\\", "\\\\")
    # 2. Escape table column delimiters
    escaped = escaped.replace("|", "\\|")
    # 3. Escape literal angle brackets to distinguish literal HTML tags from newline <br>
    escaped = escaped.replace("<", "\\<")
    # 4. Convert actual line breaks to <br> tags
    return escaped.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")


def deserialize_markdown_table_cell(cell: str) -> str:
    """Deserialize a Markdown table cell back into its original text.

    Distinguishes literal HTML text (escaped as '\\<...') from newline-generated
    '<br>', '<br/>', and '<br />' markers. Unescapes '\\|' to '|', '\\\\' to '\\',
    and '\\<' to '<'.

    Deserialization contract:
    - Escaped sequences are unescaped: '\\|' -> '|', '\\\\' -> '\\', and '\\<' -> '<'.
    - Literal HTML-like text escaped with a leading backslash (e.g. '\\<br>', '\\<br/>',
      '\\<br />') is unescaped to its literal tag representation without conversion.
    - Unescaped newline markers ('<br>', '<br/>', '<br />', case-insensitive) are
      restored as newline characters ('\\n').
    - Line endings in the original text ('\\r\\n', '\\r', '\\n') are normalized to '\\n'.
    - Therefore, round-trip equality is defined against newline-normalized input,
      not the original byte-for-byte string.
    """
    if not cell:
        return ""
    res: list[str] = []
    i = 0
    n = len(cell)
    while i < n:
        char = cell[i]
        if char == "\\":
            if i + 1 < n:
                res.append(cell[i + 1])
                i += 2
            else:
                res.append("\\")
                i += 1
        elif char == "<":
            sub_lower = cell[i:].lower()
            if sub_lower.startswith("<br />"):
                res.append("\n")
                i += 6
            elif sub_lower.startswith("<br/>"):
                res.append("\n")
                i += 5
            elif sub_lower.startswith("<br>"):
                res.append("\n")
                i += 4
            else:
                res.append("<")
                i += 1
        else:
            res.append(char)
            i += 1
    return "".join(res)


def split_markdown_table_row(line: str) -> list[str]:
    """Split a Markdown table row into stripped cell strings, respecting escaped pipes (\\|).

    Ignores leading and trailing empty cells created by outer table borders '|'.
    Properly handles sequences of backslashes immediately before delimiters:
    an odd number of backslashes before a pipe escapes the pipe (treated as cell content),
    while an even number of backslashes escapes preceding backslashes (so the pipe remains
    an active column delimiter).
    """
    stripped = line.strip()
    if not stripped:
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in stripped:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            current.append(char)
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    last_cell = "".join(current).strip()
    if last_cell:
        cells.append(last_cell)
    return cells


class GateStatus(StrEnum):
    """Evaluation gate criterion status."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class ModeExecutionStatus(StrEnum):
    """Execution status for an evaluation mode."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_EXECUTED = "NOT_EXECUTED"


class GateCriterionResult(BaseModel):
    """Result and evidence for an individual G2 acceptance criterion."""

    criterion_id: str
    title: str
    target_rule: str
    status: GateStatus
    evidence: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class GateEvaluationResult(BaseModel):
    """Aggregate evaluation result for the G2 Gate."""

    gate_name: str = "G2"
    timestamp: str
    overall_status: GateStatus
    criteria: list[GateCriterionResult]
    summary: str


class QuestionOutcomeSignature(BaseModel):
    """Deterministic per-question evaluated outcome signature for reproducibility comparison."""

    system_id: str
    scenario_id: str
    question_id: str
    question_type: str
    actual_value: Any | None = None
    actual_values: list[Any] | None = None
    actual_answer_text: str | None = None
    valid_at: str | None = None
    known_at: str | None = None
    now: str | None = None
    is_current_correct: bool | None = None
    is_as_of_correct: bool | None = None
    is_history_correct: bool | None = None
    is_stale: bool | None = None
    conflict_flagged: bool = False
    abstention_correct: bool | None = None
    abstained: bool = False
    needs_review_flagged: bool = False
    dependents: list[tuple[str, str]] = Field(default_factory=list)
    impact_recall: float | None = None
    impact_precision: float | None = None
    provenance_correct: bool | None = None
    cited_memory_ids: list[str] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    decision_records: list[dict[str, Any]] = Field(default_factory=list)
    dependent_records: list[dict[str, Any]] = Field(default_factory=list)
    active_memory_records: list[dict[str, Any]] = Field(default_factory=list)
    memory_records: list[dict[str, Any]] = Field(default_factory=list)
    repository_retrieval: dict[str, Any] | None = None
    retrieval_method: str | None = None


class NormalizedRunResult(BaseModel):
    """Normalized evaluation run snapshot for deterministic reproducibility verification."""

    metrics_summary: dict[str, dict[str, Any]]
    question_outcomes: dict[str, QuestionOutcomeSignature] = Field(default_factory=dict)
    scenario_ids: list[str]
    retrieval_method: str
    modes_evaluated: list[str]
    failure_keys: list[str]
    total_failures: int
    total_questions: int
    duplicate_keys: list[str] = Field(default_factory=list)


def normalize_run_result(
    *,
    metrics_by_system: dict[str, EvaluationMetrics],
    scenario_ids: list[str],
    retrieval_method: str = "fastembed",
    failures: list[dict[str, Any]] | None = None,
    traces_by_system: dict[str, list[QuestionEvalResult]] | None = None,
) -> NormalizedRunResult:
    """Extract a normalized, deterministic representation of an evaluation run."""
    norm_metrics: dict[str, dict[str, Any]] = {}
    for sys_id, m in sorted(metrics_by_system.items()):
        # Compare exact counts and ratios; do not apply lossy rounding
        norm_metrics[sys_id] = {
            "current_accuracy": m.current_accuracy,
            "as_of_accuracy": m.as_of_accuracy,
            "stale_answer_rate": m.stale_answer_rate,
            "conflict_recall": m.conflict_recall,
            "impact_recall": m.impact_recall,
            "provenance_correctness": m.provenance_correctness,
            "abstention_accuracy": m.abstention_accuracy,
            "engine_decision_accuracy": m.engine_decision_accuracy,
            "total_questions": m.total_questions,
            "total_messages": m.total_messages,
        }

    norm_failures: list[str] = []
    if failures:
        for f in failures:
            key = f"{f.get('system')}:{f.get('scenario_id')}:{f.get('question_id')}"
            norm_failures.append(key)

    q_outcomes: dict[str, QuestionOutcomeSignature] = {}
    duplicate_keys: list[str] = []
    seen_q_keys: set[str] = set()

    if traces_by_system:
        for sys_id, traces in sorted(traces_by_system.items()):
            for t in traces:
                q_key = f"{sys_id}:{t.scenario_id}:{t.question_id}"
                if q_key in seen_q_keys:
                    duplicate_keys.append(q_key)
                seen_q_keys.add(q_key)

                ans = t.actual if isinstance(t.actual, dict) else {}
                cited_mems = [str(m) for m in ans.get("cited_memory_ids", [])]
                cited_srcs = [str(s) for s in ans.get("cited_sources", [])]
                rules = [str(r) for r in ans.get("rules", [])]
                raw_deps = ans.get("dependents", [])
                norm_deps: list[tuple[str, str]] = []
                for d in raw_deps:
                    if isinstance(d, (list, tuple)) and len(d) >= 2:
                        norm_deps.append((str(d[0]), str(d[1])))

                raw_dec_recs = ans.get("decision_records", [])
                norm_dec_recs: list[dict[str, Any]] = []
                for r in raw_dec_recs:
                    if isinstance(r, dict):
                        norm_dec_recs.append(
                            {
                                "rule_id": str(r.get("rule_id", "")),
                                "action": str(r.get("action", "")),
                                "claim_value": (
                                    str(r.get("claim_value"))
                                    if r.get("claim_value") is not None
                                    else None
                                ),
                                "claim_subject": (
                                    str(r.get("claim_subject"))
                                    if r.get("claim_subject") is not None
                                    else None
                                ),
                                "claim_attribute": (
                                    str(r.get("claim_attribute"))
                                    if r.get("claim_attribute") is not None
                                    else None
                                ),
                            }
                        )

                raw_dep_recs = ans.get("dependent_records", [])
                norm_dep_recs: list[dict[str, Any]] = []
                for r in raw_dep_recs:
                    if isinstance(r, dict):
                        norm_dep_recs.append(
                            {
                                "id": str(r.get("id", "")),
                                "entity": str(r.get("entity", "")),
                                "attribute": str(r.get("attribute", "")),
                                "needs_review": bool(r.get("needs_review")),
                            }
                        )

                raw_act_mems = ans.get("active_memory_records", [])
                norm_act_mems: list[dict[str, Any]] = []
                for r in raw_act_mems:
                    if isinstance(r, dict):
                        norm_ctx = (
                            {str(k): str(v) for k, v in sorted(r.get("context", {}).items())}
                            if isinstance(r.get("context"), dict)
                            else {}
                        )
                        ctx_key = str(r.get("context_key") or build_context_key(norm_ctx))
                        norm_act_mems.append(
                            {
                                "id": str(r.get("id", "")),
                                "entity": str(r.get("entity", "")),
                                "attribute": str(r.get("attribute", "")),
                                "value": str(r.get("value", "")),
                                "status": str(r.get("status", "")),
                                "is_active": bool(r.get("is_active")),
                                "context": norm_ctx,
                                "context_key": ctx_key,
                                "valid_from": (
                                    str(r.get("valid_from"))
                                    if r.get("valid_from") is not None
                                    else None
                                ),
                                "valid_to": (
                                    str(r.get("valid_to"))
                                    if r.get("valid_to") is not None
                                    else None
                                ),
                                "recorded_at": (
                                    str(r.get("recorded_at"))
                                    if r.get("recorded_at") is not None
                                    else None
                                ),
                                "closed_at": (
                                    str(r.get("closed_at"))
                                    if r.get("closed_at") is not None
                                    else None
                                ),
                            }
                        )

                raw_all_mems = ans.get("memory_records", [])
                norm_all_mems: list[dict[str, Any]] = []
                for r in raw_all_mems:
                    if isinstance(r, dict):
                        norm_ctx = (
                            {str(k): str(v) for k, v in sorted(r.get("context", {}).items())}
                            if isinstance(r.get("context"), dict)
                            else {}
                        )
                        ctx_key = str(r.get("context_key") or build_context_key(norm_ctx))
                        norm_all_mems.append(
                            {
                                "id": str(r.get("id", "")),
                                "entity": str(r.get("entity", "")),
                                "attribute": str(r.get("attribute", "")),
                                "value": str(r.get("value", "")),
                                "status": str(r.get("status", "")),
                                "is_active": bool(r.get("is_active")),
                                "context": norm_ctx,
                                "context_key": ctx_key,
                                "valid_from": (
                                    str(r.get("valid_from"))
                                    if r.get("valid_from") is not None
                                    else None
                                ),
                                "valid_to": (
                                    str(r.get("valid_to"))
                                    if r.get("valid_to") is not None
                                    else None
                                ),
                                "recorded_at": (
                                    str(r.get("recorded_at"))
                                    if r.get("recorded_at") is not None
                                    else None
                                ),
                                "closed_at": (
                                    str(r.get("closed_at"))
                                    if r.get("closed_at") is not None
                                    else None
                                ),
                            }
                        )

                raw_ret_ev = ans.get("repository_retrieval")
                norm_ret_ev: dict[str, Any] | None = None
                if isinstance(raw_ret_ev, dict):
                    raw_ret_mems = raw_ret_ev.get("retrieved_memories", [])
                    norm_ret_mems: list[dict[str, Any]] = []
                    for r in raw_ret_mems:
                        if isinstance(r, dict):
                            norm_ctx = (
                                {str(k): str(v) for k, v in sorted(r.get("context", {}).items())}
                                if isinstance(r.get("context"), dict)
                                else {}
                            )
                            ctx_key = str(r.get("context_key") or build_context_key(norm_ctx))
                            norm_ret_mems.append(
                                {
                                    "id": str(r.get("id", "")),
                                    "entity": str(r.get("entity", "")),
                                    "attribute": str(r.get("attribute", "")),
                                    "value": str(r.get("value", "")),
                                    "status": str(r.get("status", "")),
                                    "is_active": bool(r.get("is_active")),
                                    "context": norm_ctx,
                                    "context_key": ctx_key,
                                    "valid_from": (
                                        str(r.get("valid_from"))
                                        if r.get("valid_from") is not None
                                        else None
                                    ),
                                    "valid_to": (
                                        str(r.get("valid_to"))
                                        if r.get("valid_to") is not None
                                        else None
                                    ),
                                    "recorded_at": (
                                        str(r.get("recorded_at"))
                                        if r.get("recorded_at") is not None
                                        else None
                                    ),
                                    "closed_at": (
                                        str(r.get("closed_at"))
                                        if r.get("closed_at") is not None
                                        else None
                                    ),
                                }
                            )

                    raw_slot_mems = raw_ret_ev.get("slot_memories", [])
                    norm_slot_mems: list[dict[str, Any]] = []
                    for r in raw_slot_mems:
                        if isinstance(r, dict):
                            norm_ctx = (
                                {str(k): str(v) for k, v in sorted(r.get("context", {}).items())}
                                if isinstance(r.get("context"), dict)
                                else {}
                            )
                            ctx_key = str(r.get("context_key") or build_context_key(norm_ctx))
                            norm_slot_mems.append(
                                {
                                    "id": str(r.get("id", "")),
                                    "entity": str(r.get("entity", "")),
                                    "attribute": str(r.get("attribute", "")),
                                    "value": str(r.get("value", "")),
                                    "status": str(r.get("status", "")),
                                    "is_active": bool(r.get("is_active")),
                                    "context": norm_ctx,
                                    "context_key": ctx_key,
                                    "valid_from": (
                                        str(r.get("valid_from"))
                                        if r.get("valid_from") is not None
                                        else None
                                    ),
                                    "valid_to": (
                                        str(r.get("valid_to"))
                                        if r.get("valid_to") is not None
                                        else None
                                    ),
                                    "recorded_at": (
                                        str(r.get("recorded_at"))
                                        if r.get("recorded_at") is not None
                                        else None
                                    ),
                                    "closed_at": (
                                        str(r.get("closed_at"))
                                        if r.get("closed_at") is not None
                                        else None
                                    ),
                                }
                            )

                    ev_ctx = (
                        {str(k): str(v) for k, v in sorted(raw_ret_ev.get("context", {}).items())}
                        if isinstance(raw_ret_ev.get("context"), dict)
                        else {}
                    )

                    norm_ret_ev = {
                        "performed": bool(raw_ret_ev.get("performed")),
                        "entity": (
                            str(raw_ret_ev.get("entity"))
                            if raw_ret_ev.get("entity") is not None
                            else None
                        ),
                        "attribute": (
                            str(raw_ret_ev.get("attribute"))
                            if raw_ret_ev.get("attribute") is not None
                            else None
                        ),
                        "context": ev_ctx,
                        "valid_at": (
                            str(raw_ret_ev.get("valid_at"))
                            if raw_ret_ev.get("valid_at") is not None
                            else None
                        ),
                        "known_at": (
                            str(raw_ret_ev.get("known_at"))
                            if raw_ret_ev.get("known_at") is not None
                            else None
                        ),
                        "retrieved_memories": sorted(
                            norm_ret_mems,
                            key=lambda x: (
                                str(x.get("id")),
                                str(x.get("valid_from")),
                            ),
                        ),
                        "slot_memories": sorted(
                            norm_slot_mems,
                            key=lambda x: (
                                str(x.get("id")),
                                str(x.get("valid_from")),
                            ),
                        ),
                        "error": (
                            str(raw_ret_ev.get("error"))
                            if raw_ret_ev.get("error") is not None
                            else None
                        ),
                    }

                sig = QuestionOutcomeSignature(
                    system_id=sys_id,
                    scenario_id=t.scenario_id,
                    question_id=t.question_id,
                    question_type=t.question_type,
                    actual_value=ans.get("value"),
                    actual_values=ans.get("values"),
                    actual_answer_text=ans.get("answer_text"),
                    valid_at=str(ans.get("valid_at")) if ans.get("valid_at") is not None else None,
                    known_at=str(ans.get("known_at")) if ans.get("known_at") is not None else None,
                    now=str(ans.get("now")) if ans.get("now") is not None else None,
                    is_current_correct=t.is_current_correct,
                    is_as_of_correct=t.is_as_of_correct,
                    is_history_correct=t.is_history_correct,
                    is_stale=t.is_stale,
                    conflict_flagged=t.conflict_flagged,
                    abstention_correct=t.abstention_correct,
                    abstained=bool(ans.get("abstained", False)),
                    needs_review_flagged=bool(ans.get("needs_review_flagged", False)),
                    dependents=sorted(norm_deps),
                    impact_recall=t.impact_recall,
                    impact_precision=t.impact_precision,
                    provenance_correct=t.provenance_correct,
                    cited_memory_ids=sorted(cited_mems),
                    cited_sources=sorted(cited_srcs),
                    rules=sorted(rules),
                    decision_records=sorted(norm_dec_recs, key=lambda x: str(x.get("rule_id"))),
                    dependent_records=sorted(norm_dep_recs, key=lambda x: str(x.get("id"))),
                    active_memory_records=sorted(
                        norm_act_mems,
                        key=lambda x: (
                            str(x.get("id")),
                            str(x.get("entity")),
                            str(x.get("attribute")),
                            str(x.get("value")),
                            str(x.get("context_key")),
                            str(x.get("valid_from")),
                        ),
                    ),
                    memory_records=sorted(
                        norm_all_mems,
                        key=lambda x: (
                            str(x.get("id")),
                            str(x.get("entity")),
                            str(x.get("attribute")),
                            str(x.get("value")),
                            str(x.get("context_key")),
                            str(x.get("valid_from")),
                        ),
                    ),
                    repository_retrieval=norm_ret_ev,
                    retrieval_method=t.retrieval_method or retrieval_method,
                )
                q_outcomes[q_key] = sig

                if not failures:
                    if (
                        t.is_current_correct is False
                        or t.is_as_of_correct is False
                        or t.is_history_correct is False
                        or t.is_stale is True
                        or t.abstention_correct is False
                        or t.provenance_correct is False
                    ):
                        norm_failures.append(q_key)

    tot_q = (
        len(q_outcomes)
        if q_outcomes
        else sum(m.total_questions for m in metrics_by_system.values())
    )

    return NormalizedRunResult(
        metrics_summary=norm_metrics,
        question_outcomes=q_outcomes,
        scenario_ids=sorted(scenario_ids),
        retrieval_method=str(retrieval_method),
        modes_evaluated=sorted(metrics_by_system.keys()),
        failure_keys=sorted(set(norm_failures)),
        total_failures=len(set(norm_failures)),
        total_questions=tot_q,
        duplicate_keys=sorted(duplicate_keys),
    )


# Explicit canonical invariant question mapping for designed-to-lose scenarios
CANONICAL_DTL_INVARIANT_QUESTIONS: dict[str, str] = {
    "lose-001": "q1",  # Safe abstention on unrecorded attribute
    "lose-002": "q1",  # Conflict flagging on unresolvable peer dispute
    "lose-003": "q1",  # Historical safe abstention before entity existence
    "lose-004": "q1",  # Rumor rejection defense (retaining 3.12, forbidding 3.9)
    "lose-005": "q1",  # Impact query flagging broken upstream dependency
}


def find_dtl_invariant_question_id(
    scenario_id: str, scenario_def: dict[str, Any] | None
) -> str | None:
    """Identifies the question ID that tests the designed-to-lose invariant.

    Derives the question ID by inspecting scenario questions for invariant-testing
    expectations (abstention, conflict_flagged, forbidden rumor, impact dependents).
    If scenario_def and questions exist, strictly derives from questions (returning
    None if none match). Falls back to CANONICAL_DTL_INVARIANT_QUESTIONS only when
    scenario_def or questions is unavailable.
    """
    if scenario_def is not None and "questions" in scenario_def:
        questions = scenario_def.get("questions")
        if not isinstance(questions, list):
            return None
        for q in questions:
            if not isinstance(q, dict):
                continue
            qid = str(q.get("id", ""))
            exp = q.get("expect", {}) if isinstance(q.get("expect"), dict) else {}
            q_type = str(q.get("type", ""))

            if scenario_id in ("lose-001", "lose-003") and exp.get("abstain") is True:
                return qid
            if scenario_id == "lose-002" and exp.get("conflict_flagged") is True:
                return qid
            if scenario_id == "lose-004" and exp.get("forbidden"):
                return qid
            if scenario_id == "lose-005" and (exp.get("dependents") or q_type == "impact"):
                return qid

        return None

    return CANONICAL_DTL_INVARIANT_QUESTIONS.get(scenario_id)


def select_dtl_invariant_trace(
    scenario_id: str,
    scenario_def: dict[str, Any] | None,
    sc_traces: list[QuestionEvalResult],
) -> tuple[QuestionEvalResult | None, str | None]:
    """Selects the exact question trace testing the designed-to-lose invariant.

    Returns:
        tuple of (trace: QuestionEvalResult | None, error_message: str | None)
    """
    if not sc_traces:
        return None, f"No execution traces recorded for scenario '{scenario_id}'"

    inv_qid = find_dtl_invariant_question_id(scenario_id, scenario_def)
    if not inv_qid:
        return None, f"Could not determine invariant-testing question ID for '{scenario_id}'"

    matching = [t for t in sc_traces if t.question_id == inv_qid]
    if not matching:
        return None, f"Invariant question '{inv_qid}' missing from traces for '{scenario_id}'"
    if len(matching) > 1:
        return (
            None,
            f"Duplicate traces found for invariant question '{inv_qid}' in '{scenario_id}' "
            f"({len(matching)} traces)",
        )

    return matching[0], None


def verify_designed_to_lose_outcome(
    scenario_id: str,
    scenario_def: dict[str, Any] | None,
    trace: QuestionEvalResult | None,
) -> tuple[bool, str, str]:
    """Verify an observed trace against the expected invariant for a designed-to-lose scenario.

    Returns:
        tuple of (is_verified: bool, observed_outcome_text: str, accounting_status_text: str)
    """
    if trace is None:
        return False, "No execution traces recorded", "**`UNEXECUTED`**"

    ans = trace.actual if isinstance(trace.actual, dict) else {}
    sid = scenario_id

    # Check for query execution errors
    if ans.get("error") is not None:
        err_msg = str(ans.get("error"))
        return False, f"Query execution error occurred: {err_msg}", "Execution Error (Fail)"

    # 1. lose-001 & lose-003: Strict explicit abstention verification
    if sid in ("lose-001", "lose-003"):
        is_abstained = ans.get("abstained") is True
        val = ans.get("value")

        # Reject ambiguous non-abstaining null return
        if not is_abstained and val is None:
            return (
                False,
                "Ambiguous non-abstaining null return (abstained=False, value=None)",
                "Observed Failure",
            )

        # Reject non-abstaining answer to unsupported query
        if not is_abstained:
            return (
                False,
                f"Failed to abstain: non-abstaining answer returned (`value={val}`)",
                "Observed Failure",
            )

        # Reject contradictory state: abstained=True but returned a value
        if val is not None:
            return (
                False,
                f"Contradictory state: abstained=True but returned non-null value (`value={val}`)",
                "Observed Failure",
            )

        label = (
            "Safe historical abstention verified"
            if sid == "lose-003"
            else "Safe abstention verified"
        )
        return (
            True,
            f"{label} (`value=None`, `abstained=True`)",
            "Honest Abstention (Pass)",
        )

    # 2. lose-002: Peer dispute conflict flagging
    elif sid == "lose-002":
        is_conf = bool(ans.get("conflict_flagged") or trace.conflict_flagged)
        vals = ans.get("values")
        if not is_conf:
            return (
                False,
                f"Conflict not flagged for unresolvable dispute (`value={ans.get('value')}`)",
                "Observed Failure",
            )
        exp_vals = {"friday", "monday"}
        act_vals = {str(v).lower() for v in vals} if isinstance(vals, (list, tuple)) else set()
        if not exp_vals.issubset(act_vals):
            return (
                False,
                f"Conflict flagged but conflicting values missing: expected {sorted(exp_vals)}, "
                f"got {sorted(act_vals)}",
                "Observed Failure",
            )
        return True, f"Conflict flagged (`values={vals}`)", "Honest Dispute Flag (Pass)"

    # 3. lose-004: Claim-specific rejection verification
    elif sid == "lose-004":
        dec_records = ans.get("decision_records")

        # Decision records must exist and be non-empty (no fallback to aggregate rules)
        if not dec_records or not isinstance(dec_records, list):
            return (
                False,
                "Decision evidence missing: no decision records available for lose-004",
                "Observed Failure",
            )

        # Derive expected claim parameters, current value, and context from scenario definition
        exp_subject = "Server A"
        exp_attribute = "python_version"
        exp_claim_val = "3.9"
        exp_current_val: str | None = None
        exp_context: dict[str, str] = {}
        has_explicit_context = False

        if trace is not None and isinstance(trace.expected, dict):
            if trace.expected.get("value") is not None:
                exp_current_val = str(trace.expected.get("value"))
            if trace.expected.get("forbidden"):
                for f in trace.expected.get("forbidden", []):
                    exp_claim_val = str(f)

        if scenario_def:
            exp_decs = scenario_def.get("expected_decisions", {})
            struct_claims = scenario_def.get("structured_claims", {})
            for mid, dec in exp_decs.items():
                if str(dec).upper() == "R-REJECT":
                    for c in struct_claims.get(mid, []):
                        if isinstance(c, dict):
                            exp_subject = str(c.get("subject", exp_subject))
                            exp_attribute = str(c.get("attribute", exp_attribute))
                            exp_claim_val = str(c.get("value", exp_claim_val))
                            if isinstance(c.get("context"), dict) and c.get("context"):
                                exp_context = dict(c.get("context"))
                                has_explicit_context = True
                            break
                elif str(dec).upper() in ("R-INSERT", "R-REPLACE"):
                    for c in struct_claims.get(mid, []):
                        if isinstance(c, dict):
                            c_subj = str(c.get("subject", "")).lower()
                            c_attr = str(c.get("attribute", "")).lower()
                            if c_subj == exp_subject.lower() and c_attr == exp_attribute.lower():
                                if exp_current_val is None and c.get("value") is not None:
                                    exp_current_val = str(c.get("value"))
                                if (
                                    not has_explicit_context
                                    and isinstance(c.get("context"), dict)
                                    and c.get("context")
                                ):
                                    exp_context = dict(c.get("context"))
                                    has_explicit_context = True

            questions = scenario_def.get("questions")
            if isinstance(questions, list):
                for q in questions:
                    if not isinstance(q, dict):
                        continue
                    plan = q.get("plan", {}) if isinstance(q.get("plan"), dict) else {}
                    if plan.get("entity"):
                        exp_subject = str(plan.get("entity"))
                    if plan.get("attribute"):
                        exp_attribute = str(plan.get("attribute"))
                    q_ctx = plan.get("context") if "context" in plan else q.get("context")
                    if isinstance(q_ctx, dict) and q_ctx:
                        exp_context = dict(q_ctx)
                        has_explicit_context = True

                    q_exp = q.get("expect", {}) if isinstance(q.get("expect"), dict) else {}
                    if q_exp.get("value") is not None and (
                        exp_current_val is None
                        or q.get("id") == getattr(trace, "question_id", None)
                    ):
                        exp_current_val = str(q_exp.get("value"))
                    if q_exp.get("forbidden"):
                        for f in q_exp.get("forbidden", []):
                            exp_claim_val = str(f)

        if exp_current_val is None:
            return (
                False,
                "Could not derive expected current value from scenario data for lose-004",
                "Observed Failure",
            )

        target_rejection_found = False
        rejection_reason = ""
        for rec in dec_records:
            if not isinstance(rec, dict):
                continue
            c_val = str(rec.get("claim_value") or rec.get("value") or "")
            c_subj = str(rec.get("claim_subject") or rec.get("subject") or "")
            c_attr = str(rec.get("claim_attribute") or rec.get("attribute") or "")
            action = str(rec.get("action", "")).upper()
            rule_id = str(rec.get("rule_id", "")).upper()

            if (
                c_val == exp_claim_val
                and c_subj.lower() == exp_subject.lower()
                and c_attr.lower() == exp_attribute.lower()
            ):
                if action == "REJECT" and rule_id == "R-REJECT":
                    target_rejection_found = True
                    rejection_reason = str(rec.get("explanation", ""))
                    break

        if not target_rejection_found:
            return (
                False,
                f"Claim-specific rejection missing: claim '{exp_claim_val}' "
                f"({exp_subject}.{exp_attribute}) was not rejected with action=REJECT and "
                "rule=R-REJECT in decision records",
                "Observed Failure",
            )

        # Verify rumor was not adopted as the answer value
        observed_val = ans.get("value")
        if str(observed_val) == exp_claim_val:
            return (
                False,
                f"Rumor accepted despite rejection record: observed value is '{observed_val}'",
                "Observed Failure",
            )

        # Verify actual repository retrieval evidence for lose-004
        retrieval_ev = ans.get("repository_retrieval")
        if not retrieval_ev or not isinstance(retrieval_ev, dict):
            return (
                False,
                "Active-memory evidence missing: repository retrieval evidence "
                "not available for lose-004",
                "Observed Failure",
            )

        if retrieval_ev.get("performed") is not True:
            err_msg = retrieval_ev.get("error") or "repository retrieval was not performed"
            return (
                False,
                f"Active-memory evidence missing: repository retrieval error - {err_msg}",
                "Observed Failure",
            )

        if retrieval_ev.get("error") is not None:
            err_msg = str(retrieval_ev.get("error"))
            return (
                False,
                f"Active-memory evidence missing: repository retrieval reported error - {err_msg}",
                "Observed Failure",
            )

        ret_ent = retrieval_ev.get("entity")
        if not ret_ent or str(ret_ent).strip().lower() != exp_subject.lower():
            return (
                False,
                f"Retrieval metadata mismatch: expected entity '{exp_subject}', got '{ret_ent}'",
                "Observed Failure",
            )

        ret_attr = retrieval_ev.get("attribute")
        if not ret_attr or str(ret_attr).strip().lower() != exp_attribute.lower():
            return (
                False,
                f"Retrieval metadata mismatch: expected attribute '{exp_attribute}', "
                f"got '{ret_attr}'",
                "Observed Failure",
            )

        ret_valid_at = retrieval_ev.get("valid_at")
        ret_valid_dt = parse_iso_datetime(ret_valid_at)
        if not ret_valid_at or ret_valid_dt is None:
            return (
                False,
                f"Retrieval metadata missing or unparseable: valid_at='{ret_valid_at}'",
                "Observed Failure",
            )
        q_valid_at = ans.get("valid_at")
        if q_valid_at is None or str(q_valid_at).strip() == "":
            return (
                False,
                "Query metadata missing: answer lacks valid_at timestamp",
                "Observed Failure",
            )
        q_valid_dt = parse_iso_datetime(q_valid_at)
        if q_valid_dt is None:
            return (
                False,
                f"Query metadata invalid: unable to parse answer valid_at='{q_valid_at}'",
                "Observed Failure",
            )
        if q_valid_dt != ret_valid_dt:
            return (
                False,
                f"Retrieval metadata mismatch: valid_at '{ret_valid_at}' != query '{q_valid_at}'",
                "Observed Failure",
            )

        ret_known_at = retrieval_ev.get("known_at")
        ret_known_dt = parse_iso_datetime(ret_known_at)
        if not ret_known_at or ret_known_dt is None:
            return (
                False,
                f"Retrieval metadata missing or unparseable: known_at='{ret_known_at}'",
                "Observed Failure",
            )
        q_known_at = ans.get("known_at")
        if q_known_at is None or str(q_known_at).strip() == "":
            return (
                False,
                "Query metadata missing: answer lacks known_at timestamp",
                "Observed Failure",
            )
        q_known_dt = parse_iso_datetime(q_known_at)
        if q_known_dt is None:
            return (
                False,
                f"Query metadata invalid: unable to parse answer known_at='{q_known_at}'",
                "Observed Failure",
            )
        if q_known_dt != ret_known_dt:
            return (
                False,
                f"Retrieval metadata mismatch: known_at '{ret_known_at}' != query '{q_known_at}'",
                "Observed Failure",
            )

        if "context" not in retrieval_ev or retrieval_ev.get("context") is None:
            return (
                False,
                "Required context evidence missing: repository retrieval lacks context",
                "Observed Failure",
            )
        ret_ctx = retrieval_ev.get("context")
        if not isinstance(ret_ctx, dict):
            return (
                False,
                f"Retrieval context invalid: expected dictionary, got {type(ret_ctx).__name__}",
                "Observed Failure",
            )
        if has_explicit_context and exp_context:
            if not match_context_scope(ret_ctx, exp_context):
                return (
                    False,
                    f"Retrieval metadata mismatch: retrieval context {ret_ctx} "
                    f"does not match scenario context {exp_context}",
                    "Observed Failure",
                )
        elif ret_ctx:
            return (
                False,
                f"Retrieval metadata mismatch: unscoped scenario has unexpected context {ret_ctx}",
                "Observed Failure",
            )

        retrieved_mems = retrieval_ev.get("retrieved_memories")
        if not isinstance(retrieved_mems, list):
            return (
                False,
                "Active-memory evidence invalid: retrieved_memories must be a list",
                "Observed Failure",
            )

        retrieved_values: set[str] = set()
        for r in retrieved_mems:
            if not isinstance(r, dict):
                return (
                    False,
                    "Active-memory evidence invalid: retrieved record is not a dict",
                    "Observed Failure",
                )
            r_id = r.get("id")
            if r_id is None or str(r_id).strip() == "":
                return (
                    False,
                    "Active-memory evidence invalid: retrieved record lacks valid ID",
                    "Observed Failure",
                )
            r_ent = str(r.get("entity", "")).strip().lower()
            if r_ent != exp_subject.lower():
                return (
                    False,
                    f"Retrieved memory id={r_id} entity mismatch: expected '{exp_subject}', "
                    f"got '{r.get('entity')}'",
                    "Observed Failure",
                )
            r_attr = str(r.get("attribute", "")).strip().lower()
            if r_attr != exp_attribute.lower():
                return (
                    False,
                    f"Retrieved memory id={r_id} attribute mismatch: expected '{exp_attribute}', "
                    f"got '{r.get('attribute')}'",
                    "Observed Failure",
                )
            r_status = str(r.get("status", "")).strip().lower()
            r_is_active = r.get("is_active")
            if r_status != "active" or r_is_active is not True:
                return (
                    False,
                    f"Retrieved memory id={r_id} has non-active status '{r.get('status')}' "
                    f"(is_active={r_is_active})",
                    "Observed Failure",
                )
            r_val = r.get("value")
            if r_val is not None and str(r_val).strip() != "":
                retrieved_values.add(str(r_val).strip())

        if exp_current_val not in retrieved_values:
            return (
                False,
                f"Expected current value '{exp_current_val}' is not active in memory "
                f"(active values: {sorted(retrieved_values)})",
                "Observed Failure",
            )

        if exp_claim_val in retrieved_values:
            return (
                False,
                f"Rejected value '{exp_claim_val}' is present in active memory "
                f"(active values: {sorted(retrieved_values)})",
                "Observed Failure",
            )

        if retrieved_values != {exp_current_val}:
            return (
                False,
                f"Unexpected active memory values for '{exp_subject}.{exp_attribute}': "
                f"{sorted(retrieved_values)}",
                "Observed Failure",
            )

        if str(observed_val) != exp_current_val:
            return (
                False,
                f"Incorrect active memory state: expected '{exp_current_val}', "
                f"got '{observed_val}'",
                "Observed Failure",
            )

        msg = (
            f"R-REJECT enforced on claim '{exp_claim_val}'; rumor blocked (`value={observed_val}`)"
        )
        if rejection_reason:
            msg += f" - {rejection_reason}"
        return (
            True,
            msg,
            "Honest Defense (Pass)",
        )

    # 4. lose-005: Expected dependent verification
    elif sid == "lose-005":
        dep_records = ans.get("dependent_records")
        if not dep_records or not isinstance(dep_records, list):
            return (
                False,
                "Per-dependent evidence missing: no dependent records available for lose-005",
                "Observed Failure",
            )

        # Derive expected dependents from scenario definition if available
        exp_deps: set[tuple[str, str]] = set()
        if scenario_def:
            for q in scenario_def.get("questions", []):
                for d in q.get("expect", {}).get("dependents", []):
                    if isinstance(d, (list, tuple)) and len(d) >= 2:
                        exp_deps.add((str(d[0]), str(d[1])))
        if not exp_deps:
            exp_deps = {("Library X", "compatibility_note")}

        # Check that each expected dependent has a matching record with needs_review is True
        for exp_ent, exp_attr in sorted(exp_deps):
            matching_records = [
                rec
                for rec in dep_records
                if isinstance(rec, dict)
                and str(rec.get("entity", "")).lower() == exp_ent.lower()
                and str(rec.get("attribute", "")).lower() == exp_attr.lower()
            ]
            if not matching_records:
                return (
                    False,
                    f"Expected dependent '{exp_ent}.{exp_attr}' missing from dependent records",
                    "Observed Failure",
                )
            for rec in matching_records:
                if "needs_review" not in rec:
                    return (
                        False,
                        f"Dependent record '{exp_ent}.{exp_attr}' missing explicit "
                        "'needs_review' field",
                        "Observed Failure",
                    )
                if rec.get("needs_review") is not True:
                    return (
                        False,
                        f"Dependent record '{exp_ent}.{exp_attr}' does not have needs_review=True "
                        f"(got {rec.get('needs_review')!r})",
                        "Observed Failure",
                    )

        return (
            True,
            f"Downstream impact flagged for review: expected dependent {sorted(exp_deps)} "
            "verified with per-dependent needs_review=True",
            "Honest Review Flag (Pass)",
        )

    # Generic fallback
    else:
        if ans.get("abstained") is True and ans.get("value") is None:
            return (
                True,
                f"Safe abstention (`value={ans.get('value')}`)",
                "Honest Abstention (Pass)",
            )
        if ans.get("conflict_flagged"):
            return (
                True,
                f"Conflict flagged (`values={ans.get('values')}`)",
                "Honest Dispute Flag (Pass)",
            )
        if ans.get("needs_review_flagged") and len(ans.get("dependents", [])) > 0:
            return (
                True,
                "Downstream impact flagged for review",
                "Honest Review Flag (Pass)",
            )
        if trace.is_current_correct or trace.is_as_of_correct:
            return (
                True,
                f"Defense verified (`value={ans.get('value')}`)",
                "Honest Defense (Pass)",
            )
        return (
            False,
            f"Boundary invariant not verified (`value={ans.get('value')}`)",
            "Observed Failure",
        )


def verify_persisted_artifacts(
    report_path: Path | None,
    results_path: Path | None,
    expected_gate_result: GateEvaluationResult | None = None,
    *,
    ignore_g2_1_status: bool = False,
    require_exact_evidence: bool = True,
) -> tuple[bool, list[str], list[str]]:
    """Verify persisted report.md and results.json match canonical specs and in-memory results.

    Returns:
        tuple of (is_valid: bool, success_evidence: list[str], failure_reasons: list[str])
    """
    evidence: list[str] = []
    failures: list[str] = []

    if report_path is None or results_path is None:
        return (
            False,
            [],
            ["Report or results JSON path not supplied (in-memory verification disallowed)"],
        )

    # 1. Results JSON verification
    if not results_path.is_file():
        failures.append(f"Results JSON file does not exist: {results_path}")
    elif results_path.stat().st_size == 0:
        failures.append(f"Results JSON file is empty (0 bytes): {results_path.name}")
    else:
        try:
            results_text = results_path.read_text(encoding="utf-8")
            parsed_json = json.loads(results_text)
            missing_keys = [k for k in REQUIRED_RESULTS_JSON_KEYS if k not in parsed_json]
            if missing_keys:
                failures.append(f"Results JSON missing required key(s): {', '.join(missing_keys)}")

            if "g2_gate" not in parsed_json or not isinstance(parsed_json["g2_gate"], dict):
                failures.append("Results JSON missing final 'g2_gate' object")
            else:
                json_gate = parsed_json["g2_gate"]
                json_overall = str(json_gate.get("overall_status", ""))
                json_criteria = json_gate.get("criteria", [])

                if not isinstance(json_criteria, list):
                    failures.append("Results JSON 'g2_gate.criteria' must be a list")
                else:
                    json_c_ids = [
                        str(c.get("criterion_id", "")) for c in json_criteria if isinstance(c, dict)
                    ]
                    seen_ids: set[str] = set()
                    dup_ids: set[str] = set()
                    for cid in json_c_ids:
                        if cid in seen_ids:
                            dup_ids.add(cid)
                        seen_ids.add(cid)

                    if dup_ids:
                        failures.append(
                            "Results JSON contains duplicate criterion ID(s): "
                            f"{sorted(list(dup_ids))}"
                        )

                    missing_cids = [cid for cid in CANONICAL_G2_CRITERIA if cid not in seen_ids]
                    if missing_cids:
                        failures.append(
                            f"Results JSON missing canonical criterion ID(s): {missing_cids}"
                        )

                    json_c_map = {
                        str(c.get("criterion_id", "")): c
                        for c in json_criteria
                        if isinstance(c, dict)
                    }
                    for cid, canonical_title in CANONICAL_G2_CRITERIA.items():
                        c_dict = json_c_map.get(cid)
                        if not c_dict:
                            continue
                        act_title = str(c_dict.get("title", ""))
                        if act_title != canonical_title:
                            failures.append(
                                f"Results JSON criterion '{cid}' title mismatch: "
                                f"expected '{canonical_title}', got '{act_title}'"
                            )

                    if expected_gate_result is not None:
                        if not ignore_g2_1_status:
                            if json_overall != expected_gate_result.overall_status.value:
                                failures.append(
                                    f"Results JSON overall status mismatch: expected "
                                    f"'{expected_gate_result.overall_status.value}', "
                                    f"got '{json_overall}'"
                                )
                            if expected_gate_result.summary:
                                json_summary = str(json_gate.get("summary", ""))
                                if json_summary != expected_gate_result.summary:
                                    failures.append(
                                        f"Results JSON overall summary mismatch: expected "
                                        f"'{expected_gate_result.summary}', got '{json_summary}'"
                                    )

                        exp_c_map = {c.criterion_id: c for c in expected_gate_result.criteria}
                        for cid, exp_c in exp_c_map.items():
                            if cid == "G2-1" and ignore_g2_1_status:
                                continue
                            c_dict = json_c_map.get(cid)
                            if c_dict:
                                act_st = str(c_dict.get("status", ""))
                                if act_st != exp_c.status.value:
                                    failures.append(
                                        f"Results JSON criterion '{cid}' status mismatch: "
                                        f"expected '{exp_c.status.value}', got '{act_st}'"
                                    )
                                act_ev = str(c_dict.get("evidence", ""))
                                if require_exact_evidence and act_ev != exp_c.evidence:
                                    failures.append(
                                        f"Results JSON criterion '{cid}' evidence mismatch: "
                                        f"expected '{exp_c.evidence}', got '{act_ev}'"
                                    )

                evidence.append(
                    f"Results JSON verified: {results_path.name} "
                    f"({results_path.stat().st_size} bytes, "
                    "valid JSON with required keys and canonical g2_gate criteria)"
                )
        except json.JSONDecodeError as e:
            failures.append(f"Results JSON failed to parse: {e}")
        except Exception as e:
            failures.append(f"Failed to read results JSON: {e}")

    # 2. Markdown Report verification
    if not report_path.is_file():
        failures.append(f"Report file does not exist: {report_path}")
    elif report_path.stat().st_size == 0:
        failures.append(f"Report file is empty (0 bytes): {report_path.name}")
    else:
        try:
            report_text = report_path.read_text(encoding="utf-8")
            missing_secs = [sec for sec in REQUIRED_REPORT_SECTIONS if sec not in report_text]
            if missing_secs:
                failures.append(f"Report missing required section(s): {', '.join(missing_secs)}")

            if "## H. G2 Evaluation Gate Verification" not in report_text:
                failures.append("Report missing Section H (## H. G2 Evaluation Gate Verification)")
            else:
                sec_h = report_text.split("## H. G2 Evaluation Gate Verification", 1)[1]

                table_lines = [
                    line.strip() for line in sec_h.splitlines() if line.strip().startswith("| `G2-")
                ]
                table_cids = []
                table_cells_map: dict[str, list[str]] = {}
                for line in table_lines:
                    cells = split_markdown_table_row(line)
                    if cells:
                        raw_cid = cells[0].strip("`")
                        table_cids.append(raw_cid)
                        table_cells_map[raw_cid] = cells
                        if len(cells) != 5:
                            failures.append(
                                f"Report Section H table row malformed for criterion '{raw_cid}' "
                                f"(expected 5 columns, got {len(cells)})"
                            )

                seen_t_ids: set[str] = set()
                dup_t_ids: set[str] = set()
                for cid in table_cids:
                    if cid in seen_t_ids:
                        dup_t_ids.add(cid)
                    seen_t_ids.add(cid)

                if dup_t_ids:
                    failures.append(
                        "Report Section H table contains duplicate criterion ID(s): "
                        f"{sorted(list(dup_t_ids))}"
                    )

                missing_t_cids = [cid for cid in CANONICAL_G2_CRITERIA if cid not in seen_t_ids]
                if missing_t_cids:
                    failures.append(
                        "Report Section H table missing canonical criterion ID(s): "
                        f"{missing_t_cids}"
                    )

                for cid, canonical_title in CANONICAL_G2_CRITERIA.items():
                    row_cells = table_cells_map.get(cid)
                    if row_cells:
                        if len(row_cells) >= 2:
                            row_title = deserialize_markdown_table_cell(row_cells[1])
                            if row_title != canonical_title:
                                failures.append(
                                    f"Report Section H criterion '{cid}' title mismatch "
                                    f"(expected '{canonical_title}', got '{row_title}')"
                                )
                        else:
                            failures.append(
                                f"Report Section H criterion '{cid}' missing title column"
                            )

                if expected_gate_result is not None:
                    if not ignore_g2_1_status:
                        exp_overall_badge = f"**`{expected_gate_result.overall_status.value}`**"
                        exp_determination_header = (
                            f"### Overall Gate Determination: {exp_overall_badge}"
                        )
                        if exp_determination_header not in sec_h:
                            failures.append(
                                "Report Section H overall status mismatch: "
                                f"expected '{exp_determination_header}'"
                            )
                        if (
                            expected_gate_result.summary
                            and expected_gate_result.summary not in sec_h
                        ):
                            failures.append(
                                "Report Section H overall summary mismatch: "
                                f"expected '{expected_gate_result.summary}'"
                            )

                    exp_c_map = {c.criterion_id: c for c in expected_gate_result.criteria}
                    for cid, exp_c in exp_c_map.items():
                        if cid == "G2-1" and ignore_g2_1_status:
                            continue
                        row_cells = table_cells_map.get(cid)
                        if row_cells:
                            exp_badge = f"**`{exp_c.status.value}`**"
                            if len(row_cells) >= 4:
                                row_st = row_cells[3]
                                if exp_badge not in row_st:
                                    failures.append(
                                        f"Report Section H criterion '{cid}' status mismatch: "
                                        f"expected {exp_badge}"
                                    )
                            else:
                                failures.append(
                                    f"Report Section H criterion '{cid}' missing status column"
                                )
                            if require_exact_evidence:
                                if len(row_cells) >= 5:
                                    row_ev = deserialize_markdown_table_cell(row_cells[4])
                                else:
                                    row_ev = ""
                                if row_ev != exp_c.evidence:
                                    failures.append(
                                        f"Report Section H criterion '{cid}' evidence mismatch: "
                                        f"expected '{exp_c.evidence}', got '{row_ev}'"
                                    )

                evidence.append(
                    f"Report verified: {report_path.name} ({report_path.stat().st_size} bytes, "
                    f"all {len(REQUIRED_REPORT_SECTIONS)} sections and Section H canonical "
                    "criteria table present)"
                )
        except Exception as e:
            failures.append(f"Failed to read report file: {e}")

    return len(failures) == 0, evidence, failures


def evaluate_g2_gate(
    *,
    metrics_by_system: dict[str, EvaluationMetrics],
    executed_scenarios: list[dict[str, Any]] | None = None,
    scenario_ids: list[str] | None = None,
    categories_present: set[str] | None = None,
    total_questions: int | None = None,
    report_path: Path | None = None,
    results_path: Path | None = None,
    timestamp: str = "",
    traces_by_system: dict[str, list[QuestionEvalResult]] | None = None,
    retrieval_method: str = "fastembed",
    failures: list[dict[str, Any]] | None = None,
    reproducibility_run: NormalizedRunResult | dict[str, Any] | None = None,
    mode_execution_statuses: dict[str, ModeExecutionStatus] | None = None,
    verify_final_artifacts: bool = False,
) -> GateEvaluationResult:
    """Evaluate all formal G2 criteria against evaluation outcomes."""
    if executed_scenarios is not None:
        actual_scenario_ids = [str(sc.get("id", "")) for sc in executed_scenarios]
    else:
        actual_scenario_ids = list(scenario_ids or [])

    # Criterion G2-2: Multi-Mode Comparative Evaluation
    mode_statuses: dict[str, ModeExecutionStatus] = {}
    if mode_execution_statuses is not None:
        for m_name in CANONICAL_MODES:
            if m_name in mode_execution_statuses:
                mode_statuses[m_name] = mode_execution_statuses[m_name]
            else:
                mode_statuses[m_name] = ModeExecutionStatus.NOT_EXECUTED
    else:
        for m_name in CANONICAL_MODES:
            sys_m = find_system_for_mode(m_name, metrics_by_system.keys())
            sys_t = (
                find_system_for_mode(m_name, traces_by_system.keys()) if traces_by_system else None
            )
            has_metric = sys_m is not None
            has_traces = sys_t is not None and bool(traces_by_system.get(sys_t))
            if has_metric and has_traces:
                mode_statuses[m_name] = ModeExecutionStatus.COMPLETED
            else:
                mode_statuses[m_name] = ModeExecutionStatus.NOT_EXECUTED

    failed_modes = [m for m, st in mode_statuses.items() if st == ModeExecutionStatus.FAILED]
    not_exec_modes = [
        m for m, st in mode_statuses.items() if st == ModeExecutionStatus.NOT_EXECUTED
    ]
    completed_modes = [m for m, st in mode_statuses.items() if st == ModeExecutionStatus.COMPLETED]
    inconsistent_modes: list[str] = []

    for m in completed_modes:
        sys_m = find_system_for_mode(m, metrics_by_system.keys())
        sys_t = find_system_for_mode(m, traces_by_system.keys()) if traces_by_system else None
        if sys_m is None:
            inconsistent_modes.append(f"mode '{m}' marked completed but missing metric evidence")
        if traces_by_system is None or sys_t is None or not traces_by_system.get(sys_t):
            inconsistent_modes.append(f"mode '{m}' marked completed but missing trace evidence")

    if failed_modes or inconsistent_modes:
        c2_status = GateStatus.FAIL
        reasons = []
        if failed_modes:
            reasons.append(f"mode execution failure: {', '.join(failed_modes)} FAILED")
        if inconsistent_modes:
            reasons.append(f"execution inconsistencies: {'; '.join(inconsistent_modes)}")
        c2_evidence = f"{'; '.join(reasons)}. Completed: {', '.join(completed_modes) or 'none'}."
    elif not_exec_modes:
        c2_status = GateStatus.INCONCLUSIVE
        c2_evidence = (
            f"Required evaluation mode(s) not executed: {', '.join(not_exec_modes)}. "
            f"Completed ({len(completed_modes)}/4): {', '.join(completed_modes)}. "
            "All 4 canonical modes must be executed."
        )
    else:
        c2_status = GateStatus.PASS
        c2_evidence = (
            "Evaluated across all 4 canonical modes: structured [COMPLETED], "
            "engine-only [COMPLETED], e2e (simulated/partial) [COMPLETED], "
            "naive RAG baseline (B0) [COMPLETED]."
        )

    c2 = GateCriterionResult(
        criterion_id="G2-2",
        title="Multi-Mode Comparative Evaluation",
        target_rule="'make eval' produces table in both modes | label e2e as partial",
        status=c2_status,
        evidence=c2_evidence,
        metrics={
            "mode_statuses": {k: v.value for k, v in mode_statuses.items()},
            "completed": completed_modes,
            "failed": failed_modes,
            "not_executed": not_exec_modes,
            "inconsistencies": inconsistent_modes,
        },
    )

    # Criterion G2-3: Scenario Catalog Coverage
    if executed_scenarios is None or traces_by_system is None or not traces_by_system:
        c3_status = GateStatus.INCONCLUSIVE
        c3_evidence = (
            "Question execution traces or scenario definitions unavailable for empirical "
            "coverage verification. Verification requires executed scenario definitions and "
            "per-mode question traces."
        )
        c3 = GateCriterionResult(
            criterion_id="G2-3",
            title="Scenario Catalog Coverage",
            target_rule="Include all applicable evaluation scenarios (~40 sc, ~100 questions)",
            status=c3_status,
            evidence=c3_evidence,
            metrics={
                "scenario_count": len(executed_scenarios) if executed_scenarios else 0,
                "total_questions": 0,
            },
        )
    else:
        coverage_deficits: list[str] = []
        verified_category_map: dict[str, list[str]] = {cat: [] for cat in REQUIRED_CATEGORIES}
        mismatched_scenarios: list[str] = []

        actual_scenario_ids = [str(sc.get("id", "")) for sc in executed_scenarios]
        sc_question_map: dict[str, set[str]] = {}
        expected_q_pairs: list[tuple[str, str]] = []

        for sc in executed_scenarios:
            sid = str(sc.get("id", ""))
            decl_cat = str(sc.get("category", "")).strip()

            q_ids = set()
            for q in sc.get("questions", []):
                qid = str(q.get("id", ""))
                expected_q_pairs.append((sid, qid))
                q_ids.add(qid)
            sc_question_map[sid] = q_ids

            if decl_cat not in REQUIRED_CATEGORIES:
                if decl_cat != "hero":
                    coverage_deficits.append(
                        f"Scenario '{sid}' has unrecognized category '{decl_cat}'"
                    )
                continue

            expected_prefixes = REQUIRED_CATEGORIES[decl_cat]
            if not any(sid.startswith(p) for p in expected_prefixes):
                mismatched_scenarios.append(
                    f"'{sid}' declared as '{decl_cat}' but expected prefix in {expected_prefixes}"
                )
                continue

            verified_category_map[decl_cat].append(sid)

        expected_q_set = set(expected_q_pairs)
        sc_count = len(actual_scenario_ids)
        actual_total_q = len(expected_q_set)

        missing_traces: list[str] = []
        duplicate_traces: list[str] = []
        unexpected_traces: list[str] = []
        incomplete_scenarios: list[str] = []

        # Require trace evidence for ALL 4 canonical modes
        for m_name in CANONICAL_MODES:
            sys_t = find_system_for_mode(m_name, traces_by_system.keys())
            if sys_t is None or not traces_by_system.get(sys_t):
                coverage_deficits.append(f"canonical mode '{m_name}' missing from execution traces")
                continue

            traces = traces_by_system[sys_t]
            actual_pairs = [(t.scenario_id, t.question_id) for t in traces]

            seen_pairs: set[tuple[str, str]] = set()
            sys_dups: set[tuple[str, str]] = set()
            for pair in actual_pairs:
                if pair in seen_pairs:
                    sys_dups.add(pair)
                seen_pairs.add(pair)
            if sys_dups:
                duplicate_traces.append(f"{m_name} ({sys_t}): {sorted(list(sys_dups))}")

            actual_pair_set = set(actual_pairs)
            missing = expected_q_set - actual_pair_set
            if missing:
                missing_traces.append(
                    f"{m_name} ({sys_t}): {sorted(list(missing))[:3]} ({len(missing)} missing)"
                )

            unexpected = actual_pair_set - expected_q_set
            if unexpected:
                unexpected_traces.append(f"{m_name} ({sys_t}): {sorted(list(unexpected))}")

            sys_sc_q_map: dict[str, set[str]] = {}
            for sid, qid in actual_pairs:
                sys_sc_q_map.setdefault(sid, set()).add(qid)

            for sid, exp_q_ids in sc_question_map.items():
                act_q_ids = sys_sc_q_map.get(sid, set())
                if exp_q_ids and act_q_ids != exp_q_ids:
                    incomplete_scenarios.append(
                        f"{m_name}:{sid} ({len(act_q_ids)}/{len(exp_q_ids)} questions)"
                    )

        if sc_count < 40:
            coverage_deficits.append(f"scenario count {sc_count} < canonical threshold 40")
        if actual_total_q < 100:
            coverage_deficits.append(f"total questions {actual_total_q} < canonical threshold 100")
        if mismatched_scenarios:
            coverage_deficits.append(
                f"category/ID prefix mismatches: {'; '.join(mismatched_scenarios)}"
            )
        if duplicate_traces:
            coverage_deficits.append(
                f"duplicate question traces detected: {'; '.join(duplicate_traces)}"
            )
        if missing_traces:
            coverage_deficits.append(f"missing question traces: {'; '.join(missing_traces)}")
        if unexpected_traces:
            coverage_deficits.append(f"unexpected question traces: {'; '.join(unexpected_traces)}")
        if incomplete_scenarios:
            coverage_deficits.append(
                "scenarios with incomplete question execution: "
                f"{'; '.join(incomplete_scenarios[:5])}"
            )

        empty_cats = [cat for cat, sids in verified_category_map.items() if not sids]
        if empty_cats:
            coverage_deficits.append(
                f"missing required categories with no executed scenarios: {sorted(empty_cats)}"
            )

        if coverage_deficits:
            c3_status = GateStatus.FAIL
            c3_evidence = f"Incomplete scenario/question coverage: {'; '.join(coverage_deficits)}."
        else:
            c3_status = GateStatus.PASS
            c3_evidence = (
                "Full canonical catalog coverage derived from execution across all 4 modes "
                f"({len(traces_by_system)} systems): {sc_count} scenarios (target >= 40) across "
                f"{len(verified_category_map)} categories with {actual_total_q} declared "
                "questions (target >= 100) all verified executed with zero missing, duplicate, "
                "or mismatched question traces."
            )

        c3 = GateCriterionResult(
            criterion_id="G2-3",
            title="Scenario Catalog Coverage",
            target_rule="Include all applicable evaluation scenarios (~40 sc, ~100 questions)",
            status=c3_status,
            evidence=c3_evidence,
            metrics={
                "scenario_count": sc_count,
                "total_questions": actual_total_q,
                "verified_category_counts": {k: len(v) for k, v in verified_category_map.items()},
                "empty_categories": empty_cats,
                "mismatches": mismatched_scenarios,
                "duplicate_traces": duplicate_traces,
                "missing_traces": missing_traces,
                "incomplete_scenarios": incomplete_scenarios,
            },
        )

    # Criterion G2-4: Write-Time Resolution Superiority
    mm_metric = None
    for k, v in metrics_by_system.items():
        if "struct" in k.lower() or "mm (full)" in k.lower():
            mm_metric = v
            break

    b0_metric = None
    for k, v in metrics_by_system.items():
        if "b0" in k.lower() or "baseline" in k.lower():
            b0_metric = v
            break

    if mm_metric is None or b0_metric is None:
        c4_status = GateStatus.INCONCLUSIVE
        c4_evidence = "Cannot evaluate core thesis: MM and B0 metrics must both be present."
    else:
        metric_checks = [
            ("MM current_accuracy", mm_metric.current_accuracy),
            ("B0 current_accuracy", b0_metric.current_accuracy),
            ("MM stale_answer_rate", mm_metric.stale_answer_rate),
            ("B0 stale_answer_rate", b0_metric.stale_answer_rate),
        ]
        missing_m: list[str] = []
        invalid_m: list[str] = []
        val_map: dict[str, float] = {}

        for name, raw_v in metric_checks:
            if raw_v is None:
                missing_m.append(name)
            else:
                ok, num_v, err = validate_evaluation_metric(raw_v, name)
                if not ok:
                    invalid_m.append(err or f"{name} invalid")
                else:
                    val_map[name] = num_v  # type: ignore[assignment]

        if invalid_m:
            c4_status = GateStatus.FAIL
            c4_evidence = f"Evaluation metric validation failed: {'; '.join(invalid_m)}."
        elif missing_m:
            c4_status = GateStatus.INCONCLUSIVE
            c4_evidence = (
                f"Cannot evaluate core thesis: required comparison metric(s) missing: "
                f"{', '.join(missing_m)}."
            )
        else:
            mm_curr = val_map["MM current_accuracy"]
            b0_curr = val_map["B0 current_accuracy"]
            mm_stale = val_map["MM stale_answer_rate"]
            b0_stale = val_map["B0 stale_answer_rate"]

            if mm_curr > b0_curr and mm_stale < b0_stale:
                c4_status = GateStatus.PASS
                c4_evidence = (
                    f"MemoryMesh write-time resolution outperforms Naive RAG (B0): "
                    f"Current Acc is {mm_curr * 100:.1f}% vs {b0_curr * 100:.1f}%; "
                    f"Stale Rate is {mm_stale * 100:.1f}% vs {b0_stale * 100:.1f}%."
                )
            else:
                c4_status = GateStatus.FAIL
                c4_evidence = (
                    f"MM failed to outperform B0 on core metrics: "
                    f"MM curr={mm_curr:.2f}, B0 curr={b0_curr:.2f}, "
                    f"MM stale={mm_stale:.2f}, B0 stale={b0_stale:.2f}."
                )

    c4 = GateCriterionResult(
        criterion_id="G2-4",
        title="Write-Time Resolution Superiority",
        target_rule="Write-time resolution beats raw message retrieval at answer time.",
        status=c4_status,
        evidence=c4_evidence,
        metrics={
            "mm_current_accuracy": mm_metric.current_accuracy if mm_metric else None,
            "b0_current_accuracy": b0_metric.current_accuracy if b0_metric else None,
            "mm_stale_answer_rate": mm_metric.stale_answer_rate if mm_metric else None,
            "b0_stale_answer_rate": b0_metric.stale_answer_rate if b0_metric else None,
        },
    )

    # Criterion G2-5: Decision Auditability and Provenance
    if mm_metric and b0_metric:
        b0_errors: list[str] = []
        if b0_metric.engine_decision_accuracy is not None:
            b0_errors.append(
                f"decision metric ({b0_metric.engine_decision_accuracy!r}) must be None (n/a)"
            )
        if b0_metric.provenance_correctness is not None:
            b0_errors.append(
                f"provenance metric ({b0_metric.provenance_correctness!r}) must be None (n/a)"
            )

        if b0_errors:
            c5_status = GateStatus.FAIL
            c5_evidence = (
                f"Baseline B0 incorrectly reported auditability metrics: {'; '.join(b0_errors)}. "
                "Baseline B0 must report 'n/a' (None) as it lacks graph/rule engine."
            )
        else:
            mm_val_checks = [
                ("MM engine_decision_accuracy", mm_metric.engine_decision_accuracy),
                ("MM provenance_correctness", mm_metric.provenance_correctness),
            ]
            missing_mm: list[str] = []
            invalid_mm: list[str] = []
            val_map_mm: dict[str, float] = {}

            for name, raw_v in mm_val_checks:
                if raw_v is None:
                    missing_mm.append(name)
                else:
                    ok, num_v, err = validate_evaluation_metric(raw_v, name)
                    if not ok:
                        invalid_mm.append(err or f"{name} invalid")
                    else:
                        val_map_mm[name] = num_v  # type: ignore[assignment]

            if invalid_mm:
                c5_status = GateStatus.FAIL
                c5_evidence = (
                    f"MemoryMesh auditability metric validation failed: {'; '.join(invalid_mm)}."
                )
            elif missing_mm:
                c5_status = GateStatus.INCONCLUSIVE
                c5_evidence = (
                    f"MemoryMesh auditability evaluation incomplete: "
                    f"{', '.join(missing_mm)} is None. "
                    "Empirical measurement is strictly required for G2-5."
                )
            else:
                mm_dec_acc = val_map_mm["MM engine_decision_accuracy"]
                mm_prov = val_map_mm["MM provenance_correctness"]

                if mm_dec_acc < 0.90 or mm_prov < 0.80:
                    c5_status = GateStatus.FAIL
                    c5_evidence = (
                        f"MemoryMesh audit metrics below canonical thresholds: "
                        f"decision accuracy={mm_dec_acc * 100:.1f}% (threshold >= 90.0%), "
                        f"provenance correctness={mm_prov * 100:.1f}% (threshold >= 80.0%)."
                    )
                else:
                    c5_status = GateStatus.PASS
                    c5_evidence = (
                        "MemoryMesh decisions and provenance are fully auditable with evidence: "
                        f"Decision Accuracy: {mm_dec_acc * 100:.1f}% (>=90.0%), "
                        f"Provenance Correctness: {mm_prov * 100:.1f}% (>=80.0%). "
                        "Baseline B0 correctly reports 'n/a' (no graph or rule logging)."
                    )
    else:
        c5_status = GateStatus.INCONCLUSIVE
        c5_evidence = "Missing system metrics for MemoryMesh or Baseline B0 for auditability check."

    c5 = GateCriterionResult(
        criterion_id="G2-5",
        title="Decision Auditability and Provenance",
        target_rule="Decisions must be auditable; citations reflect rules/sources; B0 is n/a.",
        status=c5_status,
        evidence=c5_evidence,
        metrics={
            "mm_engine_decision_accuracy": (
                mm_metric.engine_decision_accuracy if mm_metric else None
            ),
            "mm_provenance_correctness": (mm_metric.provenance_correctness if mm_metric else None),
            "b0_engine_decision_accuracy": (
                b0_metric.engine_decision_accuracy if b0_metric else None
            ),
            "b0_provenance_correctness": (b0_metric.provenance_correctness if b0_metric else None),
        },
    )

    # Criterion G2-6: Designed-To-Lose Honesty & Failure Accounting
    if executed_scenarios is not None:
        dtl_scenarios = [
            str(s.get("id"))
            for s in executed_scenarios
            if str(s.get("id", "")).startswith("lose-")
            or str(s.get("category", "")) == "designed_to_lose"
        ]
    else:
        dtl_scenarios = [s for s in actual_scenario_ids if s.startswith("lose-")]

    dtl_scenarios = sorted(set(dtl_scenarios))
    c6_deficits: list[str] = []

    if len(dtl_scenarios) < 3:
        c6_deficits.append(f"found {len(dtl_scenarios)} designed-to-lose scenarios (target >= 3)")

    # Explicitly isolate and select MemoryMesh structured evaluation traces for invariant checks
    mm_traces_dtl: list[QuestionEvalResult] = []
    if traces_by_system:
        for k, v in traces_by_system.items():
            if "struct" in k.lower():
                mm_traces_dtl = v
                break
        if not mm_traces_dtl:
            for k, v in traces_by_system.items():
                if "b0" not in k.lower() and "baseline" not in k.lower():
                    mm_traces_dtl = v
                    break

    sc_def_map = {str(s.get("id")): s for s in executed_scenarios} if executed_scenarios else {}
    verified_dtl_records: dict[str, dict[str, Any]] = {}

    if not mm_traces_dtl and traces_by_system:
        c6_deficits.append(
            "MemoryMesh evaluation traces unavailable for designed-to-lose invariant verification"
        )
    elif traces_by_system:
        for sid in dtl_scenarios:
            sc_traces = [t for t in mm_traces_dtl if t.scenario_id == sid]
            sc_def = sc_def_map.get(sid)

            target_trace, trace_err = select_dtl_invariant_trace(sid, sc_def, sc_traces)
            if trace_err or target_trace is None:
                c6_deficits.append(
                    f"'{sid}' invariant question trace selection failed: {trace_err}"
                )
                verified_dtl_records[sid] = {
                    "verified": False,
                    "outcome_text": trace_err or "Trace selection failed",
                    "acct_status": "**`UNEXECUTED`**",
                }
                continue

            is_ok, outcome_text, acct_status = verify_designed_to_lose_outcome(
                sid, sc_def, target_trace
            )
            verified_dtl_records[sid] = {
                "verified": is_ok,
                "outcome_text": outcome_text,
                "acct_status": acct_status,
            }

            if not is_ok:
                c6_deficits.append(f"'{sid}' failed invariant verification: {outcome_text}")

    # 2. Verify report representation in Section E
    if report_path is not None and report_path.is_file():
        try:
            report_text = report_path.read_text(encoding="utf-8")
            sec_e_text = ""
            if "## E. Failure Analysis" in report_text:
                after_e = report_text.split("## E. Failure Analysis", 1)[1]
                sec_e_text = after_e.split("## F. Limitations", 1)[0]

            missing_from_e = [sid for sid in dtl_scenarios if sid not in sec_e_text]
            if missing_from_e:
                c6_deficits.append(
                    f"designed-to-lose scenarios not represented in Section E (Failure Analysis): "
                    f"{missing_from_e}"
                )
            else:
                for line in sec_e_text.splitlines():
                    for sid in dtl_scenarios:
                        if f"`{sid}`" in line:
                            v_rec = verified_dtl_records.get(sid, {})
                            is_verified = v_rec.get("verified", False)

                            if not is_verified and "Pass" in line:
                                c6_deficits.append(
                                    f"designed-to-lose scenario '{sid}' claimed Pass in Section E "
                                    f"but failed actual invariant: {line.strip()}"
                                )
                            if (
                                "Observed Failure" in line
                                or "UNEXECUTED" in line
                                or "Execution Error" in line
                            ):
                                c6_deficits.append(
                                    f"designed-to-lose scenario '{sid}' has failing or "
                                    f"unexecuted status in Section E: {line.strip()}"
                                )
        except Exception as e:
            c6_deficits.append(f"Failed to read report for Section E verification: {e}")

    if c6_deficits:
        c6_status = GateStatus.FAIL
        c6_evidence = f"Designed-to-lose accounting verification failed: {'; '.join(c6_deficits)}."
    else:
        c6_status = GateStatus.PASS
        c6_evidence = (
            f"Designed-to-lose cases ({len(dtl_scenarios)} scenarios: {', '.join(dtl_scenarios)}) "
            "empirically executed with verified invariants (abstentions, dispute flags, "
            "R-REJECT decisions, and dependency reviews) and accounted for in Section E "
            "(Failure Analysis) without suppressing boundary conditions or limitations."
        )

    c6 = GateCriterionResult(
        criterion_id="G2-6",
        title="Designed-To-Lose Honesty & Failure Accounting",
        target_rule="Include at least 3 designed-to-lose cases and report them honestly.",
        status=c6_status,
        evidence=c6_evidence,
        metrics={
            "designed_to_lose_count": len(dtl_scenarios),
            "scenarios": dtl_scenarios,
            "deficits": c6_deficits,
        },
    )

    # Criterion G2-7: Deterministic Reproducibility
    current_run = normalize_run_result(
        metrics_by_system=metrics_by_system,
        scenario_ids=actual_scenario_ids,
        retrieval_method=retrieval_method,
        failures=failures,
        traces_by_system=traces_by_system,
    )

    if reproducibility_run is None:
        c7_status = GateStatus.INCONCLUSIVE
        c7_evidence = (
            "Deterministic reproducibility verification run unavailable for comparative "
            "validation. A second independent execution run is required for G2-7 verification."
        )
        c7_metrics = {"current_run": current_run.model_dump()}
    else:
        if isinstance(reproducibility_run, dict):
            if "metrics_summary" in reproducibility_run:
                rep_run = NormalizedRunResult.model_validate(reproducibility_run)
            else:
                rep_run = normalize_run_result(**reproducibility_run)
        else:
            rep_run = reproducibility_run

        discrepancies: list[str] = []

        if current_run.duplicate_keys:
            discrepancies.append(
                f"Run 1 contains duplicate question traces (duplicate question identities): "
                f"{sorted(set(current_run.duplicate_keys))}"
            )
        if rep_run.duplicate_keys:
            discrepancies.append(
                f"Run 2 contains duplicate question traces (duplicate question identities): "
                f"{sorted(set(rep_run.duplicate_keys))}"
            )

        # Independently enforce that both runs contain all four canonical modes
        run1_modes = map_canonical_modes(current_run.modes_evaluated)
        run2_modes = map_canonical_modes(rep_run.modes_evaluated)

        missing_modes_run1 = [m for m in CANONICAL_MODES if m not in run1_modes]
        missing_modes_run2 = [m for m in CANONICAL_MODES if m not in run2_modes]

        if missing_modes_run1:
            discrepancies.append(
                f"Run 1 missing canonical evaluation mode(s): {missing_modes_run1}"
            )
        if missing_modes_run2:
            discrepancies.append(
                f"Run 2 missing canonical evaluation mode(s): {missing_modes_run2}"
            )

        if not current_run.question_outcomes or not rep_run.question_outcomes:
            discrepancies.append(
                "One or both runs have empty question outcomes; per-question verification "
                "impossible"
            )
        elif executed_scenarios:
            expected_q_pairs = [
                (str(sc.get("id", "")), str(q.get("id", "")))
                for sc in executed_scenarios
                for q in sc.get("questions", [])
            ]
            expected_q_pair_set = set(expected_q_pairs)

            # Check every canonical mode has all questions in both runs
            for m_name in CANONICAL_MODES:
                sys1 = run1_modes.get(m_name)
                sys2 = run2_modes.get(m_name)

                if sys1:
                    actual_pairs_1: set[tuple[str, str]] = set()
                    for k in current_run.question_outcomes:
                        if k.startswith(f"{sys1}:"):
                            parts = k.split(":", 2)
                            if len(parts) == 3:
                                actual_pairs_1.add((parts[1], parts[2]))

                    missing_1 = expected_q_pair_set - actual_pairs_1
                    if missing_1:
                        discrepancies.append(
                            f"Run 1 mode '{m_name}' ({sys1}) missing question(s): "
                            f"{sorted(list(missing_1))}"
                        )
                    unexpected_1 = actual_pairs_1 - expected_q_pair_set
                    if unexpected_1:
                        discrepancies.append(
                            f"Run 1 mode '{m_name}' ({sys1}) unexpected question(s): "
                            f"{sorted(list(unexpected_1))}"
                        )

                if sys2:
                    actual_pairs_2: set[tuple[str, str]] = set()
                    for k in rep_run.question_outcomes:
                        if k.startswith(f"{sys2}:"):
                            parts = k.split(":", 2)
                            if len(parts) == 3:
                                actual_pairs_2.add((parts[1], parts[2]))

                    missing_2 = expected_q_pair_set - actual_pairs_2
                    if missing_2:
                        discrepancies.append(
                            f"Run 2 mode '{m_name}' ({sys2}) missing question(s): "
                            f"{sorted(list(missing_2))}"
                        )
                    unexpected_2 = actual_pairs_2 - expected_q_pair_set
                    if unexpected_2:
                        discrepancies.append(
                            f"Run 2 mode '{m_name}' ({sys2}) unexpected question(s): "
                            f"{sorted(list(unexpected_2))}"
                        )

        if current_run.retrieval_method != rep_run.retrieval_method:
            discrepancies.append(
                f"Retrieval method mismatch: '{current_run.retrieval_method}' vs "
                f"'{rep_run.retrieval_method}'"
            )
        if current_run.modes_evaluated != rep_run.modes_evaluated:
            discrepancies.append(
                f"Modes evaluated mismatch: {current_run.modes_evaluated} vs "
                f"{rep_run.modes_evaluated}"
            )
        if current_run.scenario_ids != rep_run.scenario_ids:
            discrepancies.append(
                f"Scenario catalog mismatch: {len(current_run.scenario_ids)} vs "
                f"{len(rep_run.scenario_ids)} scenarios"
            )
        if current_run.metrics_summary != rep_run.metrics_summary:
            diff_keys = []
            all_sys = sorted(
                set(current_run.metrics_summary.keys()) | set(rep_run.metrics_summary.keys())
            )
            for sys_id in all_sys:
                m1 = current_run.metrics_summary.get(sys_id, {})
                m2 = rep_run.metrics_summary.get(sys_id, {})
                if m1 != m2:
                    diff_keys.append(f"{sys_id}: {m1} != {m2}")
            discrepancies.append(f"Metrics mismatch: {'; '.join(diff_keys)}")
        if current_run.failure_keys != rep_run.failure_keys:
            discrepancies.append(
                f"Failure distribution mismatch: {len(current_run.failure_keys)} vs "
                f"{len(rep_run.failure_keys)} failures"
            )

        # Per-question outcome comparison:
        all_q_keys = sorted(
            set(current_run.question_outcomes.keys()) | set(rep_run.question_outcomes.keys())
        )
        for q_key in all_q_keys:
            o1 = current_run.question_outcomes.get(q_key)
            o2 = rep_run.question_outcomes.get(q_key)
            if o1 is None or o2 is None:
                discrepancies.append(f"Question outcome key '{q_key}' missing in one run")
            elif o1 != o2:
                diff_fields = []
                for field_name in QuestionOutcomeSignature.model_fields:
                    v1 = getattr(o1, field_name)
                    v2 = getattr(o2, field_name)
                    if v1 != v2:
                        diff_fields.append(f"{field_name} ('{v1}' vs '{v2}')")
                discrepancies.append(
                    f"Question '{q_key}' outcome discrepancy: {', '.join(diff_fields)}"
                )

        if discrepancies:
            c7_status = GateStatus.FAIL
            c7_evidence = (
                f"Deterministic reproducibility conflict detected between independent runs: "
                f"{'; '.join(discrepancies)}."
            )
        else:
            c7_status = GateStatus.PASS
            c7_evidence = (
                f"Dual independent evaluation runs produced 100% identical normalized outcomes "
                f"across all 4 canonical modes ({len(current_run.modes_evaluated)} systems), "
                f"{len(current_run.scenario_ids)} scenarios, and "
                f"{len(current_run.question_outcomes)} individual questions: all metrics, answers, "
                "citations, conflict/abstention flags, and failure distributions match exactly."
            )
        c7_metrics = {
            "current_run": current_run.model_dump(),
            "reproducibility_run": rep_run.model_dump(),
            "discrepancies": discrepancies,
        }

    c7 = GateCriterionResult(
        criterion_id="G2-7",
        title="Deterministic Reproducibility",
        target_rule="Results can be regenerated identically from the documented command.",
        status=c7_status,
        evidence=c7_evidence,
        metrics=c7_metrics,
    )

    # Criterion G2-1: Artifact Generation (Read-Back Verification)
    c1_evidence: list[str] = []
    c1_failures: list[str] = []

    if report_path is None or results_path is None:
        c1_failures.append(
            "Report or results JSON path not supplied (in-memory verification disallowed)"
        )
    else:
        # If verify_final_artifacts is requested, compare against in-memory criteria;
        # otherwise validate artifact existence, JSON schema/keys, report sections, and titles.
        temp_other_criteria = [c2, c3, c4, c5, c6, c7]
        temp_other_statuses = [c.status for c in temp_other_criteria]
        if any(st == GateStatus.FAIL for st in temp_other_statuses):
            temp_expected_overall = GateStatus.FAIL
        elif any(st == GateStatus.INCONCLUSIVE for st in temp_other_statuses):
            temp_expected_overall = GateStatus.INCONCLUSIVE
        else:
            temp_expected_overall = GateStatus.PASS

        temp_gate_result = (
            GateEvaluationResult(
                gate_name="G2",
                timestamp=timestamp,
                overall_status=temp_expected_overall,
                criteria=temp_other_criteria,
                summary="",
            )
            if verify_final_artifacts
            else None
        )

        art_ok, art_ev, art_errs = verify_persisted_artifacts(
            report_path=report_path,
            results_path=results_path,
            expected_gate_result=temp_gate_result,
            ignore_g2_1_status=True,
        )
        if not art_ok:
            c1_failures.extend(art_errs)
        else:
            c1_evidence.extend(art_ev)

    if c1_failures:
        c1_status = GateStatus.FAIL
        c1_evidence_text = f"Artifact verification failed: {'; '.join(c1_failures)}"
    else:
        c1_status = GateStatus.PASS
        c1_evidence_text = f"Artifacts verified on disk: {'; '.join(c1_evidence)}"

    c1 = GateCriterionResult(
        criterion_id="G2-1",
        title="Artifact Generation",
        target_rule="Generate report.md and results.json; complete the G2 evaluation gate.",
        status=c1_status,
        evidence=c1_evidence_text,
        metrics={"report_path": str(report_path), "results_path": str(results_path)},
    )

    criteria = [c1, c2, c3, c4, c5, c6, c7]

    # Overall Gate Determination
    if any(c.status == GateStatus.FAIL for c in criteria):
        overall_status = GateStatus.FAIL
        summary = "Gate G2 FAILED: One or more acceptance criteria failed."
    elif any(c.status == GateStatus.INCONCLUSIVE for c in criteria):
        overall_status = GateStatus.INCONCLUSIVE
        summary = "Gate G2 INCONCLUSIVE: Evaluation data incomplete to confirm all criteria."
    else:
        overall_status = GateStatus.PASS
        summary = "Gate G2 PASSED: All 7 acceptance criteria satisfied with empirical evidence."

    return GateEvaluationResult(
        gate_name="G2",
        timestamp=timestamp,
        overall_status=overall_status,
        criteria=criteria,
        summary=summary,
    )
