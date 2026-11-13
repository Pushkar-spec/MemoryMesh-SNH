"""Evaluation metrics and comparison table generation for MemoryMesh (P8.1).

Governed by docs/EVALUATION_AND_DEMO.md Section 4 and Section 5.
Computes deterministic evaluation metrics across systems:
- B0 (naive RAG baseline)
- MM (MemoryMesh full)
- A1-A5 (MemoryMesh feature ablations)
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


def _normalize_str(val: Any) -> str:
    """Normalize string for robust case-insensitive comparison."""
    if val is None:
        return ""
    return str(val).strip().lower()


class QuestionEvalResult(BaseModel):
    """Detailed evaluation result for a single question."""

    scenario_id: str
    question_id: str
    question_type: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    is_current_correct: bool | None = None
    is_as_of_correct: bool | None = None
    is_history_correct: bool | None = None
    is_stale: bool | None = None
    conflict_expected: bool = False
    conflict_flagged: bool = False
    impact_recall: float | None = None
    impact_precision: float | None = None
    provenance_correct: bool | None = None
    abstention_correct: bool | None = None
    latency_seconds: float = 0.0
    llm_calls: int = 0
    retrieval_method: str | None = None


class EvaluationMetrics(BaseModel):
    """Aggregated evaluation metrics for a system under test."""

    system_id: str
    system_name: str
    retrieval_method: str | None = None
    current_accuracy: float | None = None
    as_of_accuracy: float | None = None
    history_accuracy: float | None = None
    stale_answer_rate: float | None = None
    conflict_recall: float | None = None
    conflict_false_alarm_rate: float | None = None
    impact_recall: float | None = None
    impact_precision: float | None = None
    provenance_correctness: float | None = None
    abstention_accuracy: float | None = None
    engine_decision_accuracy: float | None = None
    avg_llm_calls_per_question: float = 0.0
    avg_latency_per_question: float = 0.0
    total_questions: int = 0
    total_messages: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


def evaluate_question(
    scenario_id: str,
    question: dict[str, Any],
    answer_dict: dict[str, Any],
    *,
    is_baseline: bool = False,
    latency_seconds: float = 0.0,
    llm_calls: int = 0,
    retrieval_method: str | None = None,
) -> QuestionEvalResult:
    """Grade an individual question output against its expected specification."""
    q_id = str(question.get("id", ""))
    q_type = str(question.get("type", "current")).lower()
    expect = question.get("expect", {})

    res = QuestionEvalResult(
        scenario_id=scenario_id,
        question_id=q_id,
        question_type=q_type,
        expected=expect,
        actual=answer_dict,
        latency_seconds=latency_seconds,
        llm_calls=llm_calls,
        retrieval_method=retrieval_method,
    )

    actual_val = _normalize_str(answer_dict.get("value"))
    actual_vals = [_normalize_str(v) for v in answer_dict.get("values", [])]
    if actual_val and not actual_vals:
        actual_vals = [actual_val]
    actual_text = str(answer_dict.get("answer_text", "")).lower()

    # 1. Current accuracy
    if q_type == "current":
        if "value" in expect:
            exp_val = _normalize_str(expect["value"])
            res.is_current_correct = (actual_val == exp_val) or (exp_val in actual_vals)
        elif "values" in expect:
            exp_vals = sorted([_normalize_str(v) for v in expect["values"]])
            res.is_current_correct = sorted(actual_vals) == exp_vals

    # 2. As-of accuracy
    if q_type in ("as_of", "time_travel"):
        if "value" in expect:
            exp_val = _normalize_str(expect["value"])
            res.is_as_of_correct = (actual_val == exp_val) or (exp_val in actual_vals)
        elif "values" in expect:
            exp_vals = sorted([_normalize_str(v) for v in expect["values"]])
            res.is_as_of_correct = sorted(actual_vals) == exp_vals

    # 3. History accuracy (order-sensitive)
    if q_type == "history":
        exp_vals = [_normalize_str(v) for v in expect.get("values", [])]
        res.is_history_correct = actual_vals == exp_vals

    # 4. Stale-answer rate (check for forbidden values)
    forbidden = [_normalize_str(f) for f in expect.get("forbidden", [])]
    if forbidden:
        is_stale = False
        for f in forbidden:
            if f in actual_vals or f in actual_val:
                is_stale = True
                break
            # Word-boundary check in natural-language text
            pattern = rf"\b{re.escape(f)}\b"
            if re.search(pattern, actual_text):
                is_stale = True
                break
        res.is_stale = is_stale

    # 5. Conflicts
    res.conflict_expected = bool(expect.get("conflict_flagged", False))
    res.conflict_flagged = bool(answer_dict.get("conflict_flagged", False))

    # 6. Abstention
    if "abstain" in expect:
        res.abstention_correct = bool(answer_dict.get("abstained", False)) == bool(
            expect["abstain"]
        )

    # 7. Impact recall & precision
    if q_type == "impact":
        if is_baseline:
            res.impact_recall = None
            res.impact_precision = None
        else:
            exp_deps = {
                (dep[0].strip().lower(), dep[1].strip().lower())
                for dep in expect.get("dependents", [])
                if len(dep) >= 2
            }
            actual_deps_raw = answer_dict.get("dependents", [])
            actual_deps = {
                (d[0].strip().lower(), d[1].strip().lower()) for d in actual_deps_raw if len(d) >= 2
            }
            if exp_deps:
                tp = len(exp_deps & actual_deps)
                res.impact_recall = tp / len(exp_deps)
                res.impact_precision = tp / len(actual_deps) if actual_deps else 0.0
            else:
                res.impact_recall = 1.0 if not actual_deps else 0.0
                res.impact_precision = 1.0 if not actual_deps else 0.0

            # Also check review flag if expected
            if "needs_review_flagged" in expect:
                flag_expected = bool(expect["needs_review_flagged"])
                flag_actual = bool(answer_dict.get("needs_review_flagged", False))
                if flag_expected and not flag_actual:
                    res.impact_recall = 0.0

    # 8. Provenance correctness
    if q_type == "why":
        if is_baseline:
            res.provenance_correct = None
        else:
            exp_src = _normalize_str(expect.get("cited_source"))
            exp_rule = _normalize_str(expect.get("rule_id"))
            exp_str = _normalize_str(answer_dict.get("explanation", ""))
            actual_srcs = [_normalize_str(s) for s in answer_dict.get("cited_sources", [])]
            actual_rules = [_normalize_str(r) for r in answer_dict.get("rules", [])]

            src_match = True
            if exp_src:
                src_match = exp_src in actual_srcs or exp_src in exp_str or exp_src in actual_text

            rule_match = True
            if exp_rule:
                rule_match = (
                    exp_rule in actual_rules or exp_rule in exp_str or exp_rule in actual_text
                )

            res.provenance_correct = src_match and rule_match

    return res


def compute_metrics(
    system_id: str,
    system_name: str,
    question_results: list[QuestionEvalResult],
    decision_matches: list[bool] | None = None,
    total_messages: int = 0,
    metadata: dict[str, Any] | None = None,
    retrieval_method: str | None = None,
) -> EvaluationMetrics:
    """Aggregate per-question results into summary metrics."""
    is_baseline = system_id.upper() == "B0"

    # Current
    curr_results = [
        r.is_current_correct for r in question_results if r.is_current_correct is not None
    ]
    current_acc = sum(curr_results) / len(curr_results) if curr_results else None

    # As-of
    as_of_results = [r.is_as_of_correct for r in question_results if r.is_as_of_correct is not None]
    as_of_acc = sum(as_of_results) / len(as_of_results) if as_of_results else None

    # History
    hist_results = [
        r.is_history_correct for r in question_results if r.is_history_correct is not None
    ]
    history_acc = sum(hist_results) / len(hist_results) if hist_results else None

    # Stale answer rate (percentage of questions with forbidden values that were violated)
    stale_results = [r.is_stale for r in question_results if r.is_stale is not None]
    stale_rate = sum(stale_results) / len(stale_results) if stale_results else None

    # Conflicts
    exp_conflicts = [r for r in question_results if r.conflict_expected]
    exp_non_conflicts = [r for r in question_results if not r.conflict_expected]
    conflict_recall = (
        sum(1 for r in exp_conflicts if r.conflict_flagged) / len(exp_conflicts)
        if exp_conflicts
        else None
    )
    conflict_far = (
        sum(1 for r in exp_non_conflicts if r.conflict_flagged) / len(exp_non_conflicts)
        if exp_non_conflicts
        else None
    )

    # Impact
    if is_baseline:
        impact_rec = None
        impact_prec = None
    else:
        imp_recs = [r.impact_recall for r in question_results if r.impact_recall is not None]
        imp_precs = [r.impact_precision for r in question_results if r.impact_precision is not None]
        impact_rec = sum(imp_recs) / len(imp_recs) if imp_recs else None
        impact_prec = sum(imp_precs) / len(imp_precs) if imp_precs else None

    # Provenance
    if is_baseline:
        prov_acc = None
    else:
        prov_results = [
            r.provenance_correct for r in question_results if r.provenance_correct is not None
        ]
        prov_acc = sum(prov_results) / len(prov_results) if prov_results else None

    # Abstention
    abst_results = [
        r.abstention_correct for r in question_results if r.abstention_correct is not None
    ]
    abst_acc = sum(abst_results) / len(abst_results) if abst_results else None

    # Engine decisions
    if is_baseline or decision_matches is None or not decision_matches:
        dec_acc = None
    else:
        dec_acc = sum(decision_matches) / len(decision_matches)

    # Cost
    total_q = len(question_results)
    avg_calls = sum(r.llm_calls for r in question_results) / total_q if total_q > 0 else 0.0
    avg_lat = sum(r.latency_seconds for r in question_results) / total_q if total_q > 0 else 0.0

    return EvaluationMetrics(
        system_id=system_id,
        system_name=system_name,
        retrieval_method=retrieval_method,
        current_accuracy=current_acc,
        as_of_accuracy=as_of_acc,
        history_accuracy=history_acc,
        stale_answer_rate=stale_rate,
        conflict_recall=conflict_recall,
        conflict_false_alarm_rate=conflict_far,
        impact_recall=impact_rec,
        impact_precision=impact_prec,
        provenance_correctness=prov_acc,
        abstention_accuracy=abst_acc,
        engine_decision_accuracy=dec_acc,
        avg_llm_calls_per_question=avg_calls,
        avg_latency_per_question=avg_lat,
        total_questions=total_q,
        total_messages=total_messages,
        metadata=metadata or {},
    )


def _fmt(val: float | None, is_pct: bool = True) -> str:
    """Format metric value as percentage or n/a."""
    if val is None:
        return "n/a"
    if is_pct:
        return f"{val * 100:.1f}%"
    return f"{val:.2f}"


def format_comparison_table(metrics_by_system: dict[str, EvaluationMetrics]) -> str:
    """Generate GitHub-flavored Markdown comparison table across systems."""
    systems = list(metrics_by_system.keys())
    if not systems:
        return "_No evaluation metrics recorded._\n"

    headers = ["Metric"] + [f"{s} ({metrics_by_system[s].system_name})" for s in systems]
    lines: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] + [":---:"] * len(systems)) + " |",
    ]

    metric_rows = [
        ("Current accuracy", lambda m: _fmt(m.current_accuracy)),
        ("As-of accuracy", lambda m: _fmt(m.as_of_accuracy)),
        ("History accuracy", lambda m: _fmt(m.history_accuracy)),
        ("Stale-answer rate", lambda m: _fmt(m.stale_answer_rate)),
        ("Conflict recall", lambda m: _fmt(m.conflict_recall)),
        ("Conflict false-alarm rate", lambda m: _fmt(m.conflict_false_alarm_rate)),
        ("Impact recall", lambda m: _fmt(m.impact_recall)),
        ("Impact precision", lambda m: _fmt(m.impact_precision)),
        ("Provenance correctness", lambda m: _fmt(m.provenance_correctness)),
        ("Abstention accuracy", lambda m: _fmt(m.abstention_accuracy)),
        ("Engine decision accuracy", lambda m: _fmt(m.engine_decision_accuracy)),
        ("Avg LLM calls / query", lambda m: _fmt(m.avg_llm_calls_per_question, is_pct=False)),
        ("Avg Latency (s) / query", lambda m: _fmt(m.avg_latency_per_question, is_pct=False)),
    ]

    for label, fn in metric_rows:
        row = [label] + [fn(metrics_by_system[s]) for s in systems]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(
        "\\* _B0 metrics for Impact, Provenance, and Decision accuracy are reported as 'n/a' "
        "(no rule IDs or graph traversal)._"
    )
    return "\n".join(lines) + "\n"


def format_ascii_table(metrics_by_system: dict[str, EvaluationMetrics]) -> str:
    """Generate fixed-width plain text summary table for console output."""
    systems = list(metrics_by_system.keys())
    if not systems:
        return "No evaluation metrics recorded.\n"

    col_widths = [26] + [max(14, len(s) + 6) for s in systems]

    header_cols = ["Metric"] + [s for s in systems]
    header_line = "".join(f"{h:<{w}}" for h, w in zip(header_cols, col_widths, strict=False))
    sep_line = "-" * len(header_line)

    lines = [sep_line, header_line, sep_line]

    metric_rows = [
        ("Current accuracy", lambda m: _fmt(m.current_accuracy)),
        ("As-of accuracy", lambda m: _fmt(m.as_of_accuracy)),
        ("History accuracy", lambda m: _fmt(m.history_accuracy)),
        ("Stale-answer rate", lambda m: _fmt(m.stale_answer_rate)),
        ("Conflict recall", lambda m: _fmt(m.conflict_recall)),
        ("Conflict false-alarm rate", lambda m: _fmt(m.conflict_false_alarm_rate)),
        ("Impact recall", lambda m: _fmt(m.impact_recall)),
        ("Impact precision", lambda m: _fmt(m.impact_precision)),
        ("Provenance correctness", lambda m: _fmt(m.provenance_correctness)),
        ("Abstention accuracy", lambda m: _fmt(m.abstention_accuracy)),
        ("Engine decision accuracy", lambda m: _fmt(m.engine_decision_accuracy)),
        ("Avg LLM calls / q", lambda m: _fmt(m.avg_llm_calls_per_question, is_pct=False)),
        ("Avg Latency (s) / q", lambda m: _fmt(m.avg_latency_per_question, is_pct=False)),
    ]

    for label, fn in metric_rows:
        cols = [label] + [fn(metrics_by_system[s]) for s in systems]
        lines.append("".join(f"{c:<{w}}" for c, w in zip(cols, col_widths, strict=False)))

    lines.append(sep_line)
    lines.append(
        "* B0 metrics for Impact, Provenance, and Decision accuracy are reported as 'n/a'."
    )
    return "\n".join(lines) + "\n"
