"""Tests for P8.4 Evaluation Report and G2 Gate verification.

Governed by:
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P8.4, Execution Gate G2)
- docs/ROADMAP_AND_TASKS.md (T8.4, G2 Gate)
- docs/EVALUATION_AND_DEMO.md

Verifies:
1. G2-1: Artifact generation integrity (report.md, results.json, section presence, json schema).
2. G2-2: Mode execution statuses (distinguish COMPLETED, FAILED, NOT_EXECUTED for all 4 modes).
3. G2-3: Derived category coverage (scenario ID prefix, declared category, and execution agreement).
4. G2-4: Core write-time resolution thesis (missing metrics -> INCONCLUSIVE).
5. G2-5: Decision auditability & provenance (MM >= 0.90/0.80, None disallowed, B0 is n/a).
6. G2-6: Designed-to-lose accounting (verification of execution and representation in Section E).
7. G2-7: Deterministic reproducibility (per-question answer, citation, and flag comparison).
8. Overall gate aggregation (PASS only when all 7 pass; FAIL overrides INCONCLUSIVE).
9. Report generation structure and metric preservation.
10. Provider isolation and zero expected-answer leakage.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.gate import (
    CANONICAL_G2_CRITERIA,
    REQUIRED_REPORT_SECTIONS,
    GateCriterionResult,
    GateEvaluationResult,
    GateStatus,
    ModeExecutionStatus,
    deserialize_markdown_table_cell,
    evaluate_g2_gate,
    find_dtl_invariant_question_id,
    normalize_run_result,
    select_dtl_invariant_trace,
    serialize_markdown_table_cell,
    split_markdown_table_row,
    validate_evaluation_metric,
    verify_designed_to_lose_outcome,
    verify_persisted_artifacts,
)
from eval.metrics import EvaluationMetrics, QuestionEvalResult
from eval.providers import (
    DeterministicEvaluationBaselineProvider,
    DeterministicEvaluationE2EProvider,
)
from eval.report_generator import generate_evaluation_report
from eval.run_eval import (
    finalize_evaluation_artifacts,
    load_scenarios,
    run_system_on_scenarios,
)
from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.llm.client import LLMClient
from memorymesh.models import Claim, IngestMessage, MemoryStatus, Source
from memorymesh.retrieval.retrieve import retrieve_valid_memories
from memorymesh.service import MemoryService


def _canonical_metrics() -> dict[str, EvaluationMetrics]:
    """Helper returning canonical passing metrics across all 4 modes."""
    return {
        "MM (structured)": EvaluationMetrics(
            system_id="MM",
            system_name="MemoryMesh (structured)",
            current_accuracy=1.0,
            as_of_accuracy=1.0,
            stale_answer_rate=0.0,
            conflict_recall=1.0,
            impact_recall=1.0,
            provenance_correctness=1.0,
            abstention_accuracy=1.0,
            engine_decision_accuracy=1.0,
            total_questions=111,
        ),
        "MM (engine-only)": EvaluationMetrics(
            system_id="MM",
            system_name="MemoryMesh (engine-only)",
            current_accuracy=1.0,
            as_of_accuracy=1.0,
            stale_answer_rate=0.0,
            conflict_recall=1.0,
            impact_recall=1.0,
            provenance_correctness=1.0,
            abstention_accuracy=1.0,
            engine_decision_accuracy=1.0,
            total_questions=111,
        ),
        "MM (e2e)": EvaluationMetrics(
            system_id="MM",
            system_name="MemoryMesh (e2e)",
            current_accuracy=0.85,
            as_of_accuracy=0.90,
            stale_answer_rate=0.02,
            conflict_recall=1.0,
            impact_recall=1.0,
            provenance_correctness=0.94,
            abstention_accuracy=0.80,
            engine_decision_accuracy=1.0,
            total_questions=111,
        ),
        "B0 (naive RAG)": EvaluationMetrics(
            system_id="B0",
            system_name="Naive RAG",
            current_accuracy=0.58,
            as_of_accuracy=0.36,
            stale_answer_rate=0.35,
            conflict_recall=0.80,
            impact_recall=None,
            provenance_correctness=None,
            abstention_accuracy=0.20,
            engine_decision_accuracy=None,
            total_questions=111,
        ),
    }


def _canonical_scenarios_and_categories() -> tuple[list[dict[str, Any]], set[str]]:
    """Helper returning canonical 51 scenario definitions and 10 categories."""
    categories = {
        "simple_update",
        "corroboration",
        "low_trust_reject",
        "dispute",
        "multi_valued",
        "context_isolation",
        "out_of_order",
        "dependency_propagation",
        "historical_time_travel",
        "designed_to_lose",
    }
    sc_defs: list[dict[str, Any]] = []

    cat_specs = [
        ("simple_update", "update", 6),
        ("corroboration", "corroborate", 5),
        ("low_trust_reject", "reject", 5),
        ("dispute", "dispute", 4),
        ("multi_valued", "multivalue", 4),
        ("context_isolation", "context", 6),
        ("out_of_order", "ordering", 5),
        ("dependency_propagation", "dependency", 5),
        ("historical_time_travel", "temporal", 5),
        ("designed_to_lose", "lose", 5),
    ]

    for cat, prefix, count in cat_specs:
        for i in range(1, count + 1):
            sid = f"{prefix}-{i:03d}"
            if sid == "lose-001":
                q1_exp = {"abstain": True, "value": None}
            elif sid == "lose-002":
                q1_exp = {"conflict_flagged": True, "values": ["friday", "monday"]}
            elif sid == "lose-003":
                q1_exp = {"abstain": True, "value": None}
            elif sid == "lose-004":
                q1_exp = {"value": "3.12", "forbidden": ["3.9"]}
            elif sid == "lose-005":
                q1_exp = {
                    "dependents": [("Library X", "compatibility_note")],
                    "needs_review_flagged": True,
                }
            else:
                q1_exp = {"value": "v1"}

            sc_defs.append(
                {
                    "id": sid,
                    "category": cat,
                    "now": "2026-06-25T09:00:00Z",
                    "questions": [
                        {"id": "q1", "text": f"Question 1 for {sid}", "expect": q1_exp},
                        {"id": "q2", "text": f"Question 2 for {sid}", "expect": {"value": "v2"}},
                    ],
                }
            )

    return sc_defs, categories


def _canonical_traces(
    scenarios: list[dict[str, Any]], systems: list[str]
) -> dict[str, list[QuestionEvalResult]]:
    """Helper returning question traces for all scenarios across systems."""
    traces: dict[str, list[QuestionEvalResult]] = {sys_id: [] for sys_id in systems}
    for sc in scenarios:
        sid = sc["id"]
        for q in sc.get("questions", []):
            qid = q["id"]
            for sys_id in systems:
                if sid == "lose-001":
                    act = {
                        "value": None,
                        "abstained": True,
                        "conflict_flagged": False,
                        "needs_review_flagged": False,
                        "rules": ["R-INSERT"],
                        "dependents": [],
                    }
                elif sid == "lose-002":
                    act = {
                        "value": None,
                        "values": ["friday", "monday"],
                        "abstained": False,
                        "conflict_flagged": True,
                        "needs_review_flagged": False,
                        "rules": ["R-INSERT", "R-DISPUTE"],
                        "dependents": [],
                    }
                elif sid == "lose-003":
                    act = {
                        "value": None,
                        "abstained": True,
                        "conflict_flagged": False,
                        "needs_review_flagged": False,
                        "rules": ["R-INSERT"],
                        "dependents": [],
                    }
                elif sid == "lose-004":
                    act = {
                        "value": "3.12",
                        "abstained": False,
                        "conflict_flagged": False,
                        "needs_review_flagged": False,
                        "valid_at": "2026-06-25T09:00:00Z",
                        "known_at": "2026-06-25T09:00:00Z",
                        "now": "2026-06-25T09:00:00Z",
                        "rules": ["R-INSERT", "R-REJECT"],
                        "decision_records": [
                            {
                                "rule_id": "R-INSERT",
                                "action": "INSERT",
                                "claim_value": "3.12",
                                "claim_subject": "Server A",
                                "claim_attribute": "python_version",
                            },
                            {
                                "rule_id": "R-REJECT",
                                "action": "REJECT",
                                "claim_value": "3.9",
                                "claim_subject": "Server A",
                                "claim_attribute": "python_version",
                            },
                        ],
                        "active_memory_records": [
                            {
                                "id": "1",
                                "entity": "Server A",
                                "attribute": "python_version",
                                "value": "3.12",
                                "status": "active",
                                "is_active": True,
                                "context": {},
                                "context_key": "",
                                "valid_from": "2026-01-10T09:00:00Z",
                                "valid_to": None,
                                "recorded_at": "2026-01-10T09:00:00Z",
                                "closed_at": None,
                            }
                        ],
                        "memory_records": [
                            {
                                "id": "1",
                                "entity": "Server A",
                                "attribute": "python_version",
                                "value": "3.12",
                                "status": "active",
                                "is_active": True,
                                "context": {},
                                "context_key": "",
                                "valid_from": "2026-01-10T09:00:00Z",
                                "valid_to": None,
                                "recorded_at": "2026-01-10T09:00:00Z",
                                "closed_at": None,
                            },
                            {
                                "id": "2",
                                "entity": "Server A",
                                "attribute": "python_version",
                                "value": "3.9",
                                "status": "rejected",
                                "is_active": False,
                                "context": {},
                                "context_key": "",
                                "valid_from": "2026-01-12T11:00:00Z",
                                "valid_to": None,
                                "recorded_at": "2026-01-12T11:00:00Z",
                                "closed_at": None,
                            },
                        ],
                        "repository_retrieval": {
                            "performed": True,
                            "entity": "Server A",
                            "attribute": "python_version",
                            "context": {},
                            "valid_at": "2026-06-25T09:00:00Z",
                            "known_at": "2026-06-25T09:00:00Z",
                            "retrieved_memories": [
                                {
                                    "id": "1",
                                    "entity": "Server A",
                                    "attribute": "python_version",
                                    "value": "3.12",
                                    "status": "active",
                                    "is_active": True,
                                    "context": {},
                                    "context_key": "",
                                    "valid_from": "2026-01-10T09:00:00Z",
                                    "valid_to": None,
                                    "recorded_at": "2026-01-10T09:00:00Z",
                                    "closed_at": None,
                                }
                            ],
                            "slot_memories": [
                                {
                                    "id": "1",
                                    "entity": "Server A",
                                    "attribute": "python_version",
                                    "value": "3.12",
                                    "status": "active",
                                    "is_active": True,
                                    "context": {},
                                    "context_key": "",
                                    "valid_from": "2026-01-10T09:00:00Z",
                                    "valid_to": None,
                                    "recorded_at": "2026-01-10T09:00:00Z",
                                    "closed_at": None,
                                },
                                {
                                    "id": "2",
                                    "entity": "Server A",
                                    "attribute": "python_version",
                                    "value": "3.9",
                                    "status": "rejected",
                                    "is_active": False,
                                    "context": {},
                                    "context_key": "",
                                    "valid_from": "2026-01-12T11:00:00Z",
                                    "valid_to": None,
                                    "recorded_at": "2026-01-12T11:00:00Z",
                                    "closed_at": None,
                                },
                            ],
                            "error": None,
                        },
                        "dependents": [],
                        "dependent_records": [],
                    }
                elif sid == "lose-005":
                    act = {
                        "value": None,
                        "abstained": False,
                        "conflict_flagged": False,
                        "needs_review_flagged": True,
                        "rules": ["R-INSERT", "R-REPLACE", "R-PROPAGATE"],
                        "dependents": [("Library X", "compatibility_note")],
                        "dependent_records": [
                            {
                                "id": "2",
                                "entity": "Library X",
                                "attribute": "compatibility_note",
                                "needs_review": True,
                            }
                        ],
                    }
                else:
                    act = {
                        "value": "v1" if qid == "q1" else "v2",
                        "answer_text": "Answer text",
                        "cited_memory_ids": ["m1"],
                        "cited_sources": ["src1"],
                        "rules": ["R-INSERT"],
                        "abstained": False,
                        "conflict_flagged": False,
                        "needs_review_flagged": False,
                        "dependents": [],
                    }

                traces[sys_id].append(
                    QuestionEvalResult(
                        scenario_id=sid,
                        question_id=qid,
                        question_type=q.get("type", "current"),
                        expected=q.get("expect", {}),
                        actual=act,
                        is_current_correct=True,
                        is_stale=False,
                        provenance_correct=True,
                        abstention_correct=True,
                        conflict_flagged=act.get("conflict_flagged", False),
                    )
                )
    return traces


def _write_valid_artifacts(
    tmpdir: Path,
    dtl_sids: list[str] | None = None,
    criterion_statuses: dict[str, str] | None = None,
    overall_status: str = "PASS",
) -> tuple[Path, Path]:
    """Helper writing minimal valid report.md and results.json to tmpdir."""
    tmpdir.mkdir(parents=True, exist_ok=True)
    rep_p = tmpdir / "report.md"
    res_p = tmpdir / "results.json"

    dtl_sids = dtl_sids or ["lose-001", "lose-002", "lose-003", "lose-004", "lose-005"]
    dtl_table = "\n".join(f"| `{sid}` | Invariant | Outcome | Pass |" for sid in dtl_sids)
    sec_e = (
        f"## E. Failure Analysis\n\n### 2. Accounting\n\n"
        f"| Scenario ID | Invariant | Outcome | Status |\n| :--- | :--- | :--- | :--- |\n"
        f"{dtl_table}\n"
    )

    statuses = {
        "G2-1": "PASS",
        "G2-2": "PASS",
        "G2-3": "PASS",
        "G2-4": "PASS",
        "G2-5": "PASS",
        "G2-6": "PASS",
        "G2-7": "PASS",
    }
    if criterion_statuses:
        statuses.update(criterion_statuses)

    canonical_titles = {
        "G2-1": "Artifact Generation",
        "G2-2": "Multi-Mode Comparative Evaluation",
        "G2-3": "Scenario Catalog Coverage",
        "G2-4": "Write-Time Resolution Superiority",
        "G2-5": "Decision Auditability and Provenance",
        "G2-6": "Designed-To-Lose Honesty & Failure Accounting",
        "G2-7": "Deterministic Reproducibility",
    }

    criteria_table_rows = "\n".join(
        f"| `{cid}` | {title} | Rule for {cid} | **`{statuses[cid]}`** | Evidence for {cid} |"
        for cid, title in canonical_titles.items()
    )
    criteria_table = (
        "| Criterion | Title | Target Acceptance Rule | Status | Empirical Evidence |\n"
        "| :---: | :--- | :--- | :---: | :--- |\n"
        f"{criteria_table_rows}\n"
    )

    body_sections = [
        "## A. Executive Summary\nText",
        "## B. Evaluation Methodology\nText",
        "## C. Quantitative Results\nText",
        "## D. Scenario-Group Analysis\nText",
        sec_e,
        "## F. Limitations\nText",
        "## G. Conclusion\nText",
        (
            f"## H. G2 Evaluation Gate Verification\n\n"
            f"### Overall Gate Determination: **`{overall_status}`**\n\n"
            f"Gate G2 {overall_status}\n\n{criteria_table}"
        ),
    ]
    rep_p.write_text("# Evaluation Report\n\n" + "\n\n".join(body_sections), encoding="utf-8")

    json_data = {
        "timestamp": "20261115_000000",
        "metrics": {"MM": {}},
        "scenarios": ["update-001"],
        "g2_gate": {
            "gate_name": "G2",
            "timestamp": "20261115_000000",
            "overall_status": overall_status,
            "criteria": [
                {
                    "criterion_id": cid,
                    "title": title,
                    "target_rule": f"Rule for {cid}",
                    "status": statuses[cid],
                    "evidence": f"Evidence for {cid}",
                }
                for cid, title in canonical_titles.items()
            ],
            "summary": f"Gate G2 {overall_status}",
        },
    }
    res_p.write_text(json.dumps(json_data), encoding="utf-8")
    return rep_p, res_p


def test_p84_g2_1_artifact_generation_integrity() -> None:
    """Verifies G2-1 strictly asserts disk presence, non-emptiness, JSON parsing, and sections."""
    metrics = _canonical_metrics()
    sc_defs, cats = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        rep_p, res_p = _write_valid_artifacts(td)
        rep_run = normalize_run_result(metrics_by_system=metrics, scenario_ids=sc_ids)

        # 1. Valid artifacts -> PASS
        res_pass = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            report_path=rep_p,
            results_path=res_p,
            reproducibility_run=rep_run,
        )
        c1 = next(c for c in res_pass.criteria if c.criterion_id == "G2-1")
        assert c1.status == GateStatus.PASS

        # 2. None paths (in-memory verification disallowed) -> FAIL
        res_none = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            report_path=None,
            results_path=None,
            reproducibility_run=rep_run,
        )
        c1_none = next(c for c in res_none.criteria if c.criterion_id == "G2-1")
        assert c1_none.status == GateStatus.FAIL
        assert "not supplied" in c1_none.evidence

        # 3. Non-existent file -> FAIL
        res_missing = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            report_path=td / "missing_report.md",
            results_path=res_p,
            reproducibility_run=rep_run,
        )
        c1_miss = next(c for c in res_missing.criteria if c.criterion_id == "G2-1")
        assert c1_miss.status == GateStatus.FAIL
        assert "does not exist" in c1_miss.evidence

        # 4. Empty report file (0 bytes) -> FAIL
        empty_rep = td / "empty_report.md"
        empty_rep.write_text("", encoding="utf-8")
        res_empty = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            report_path=empty_rep,
            results_path=res_p,
            reproducibility_run=rep_run,
        )
        c1_empty = next(c for c in res_empty.criteria if c.criterion_id == "G2-1")
        assert c1_empty.status == GateStatus.FAIL
        assert "empty (0 bytes)" in c1_empty.evidence

        # 5. Invalid JSON -> FAIL
        bad_json = td / "corrupt.json"
        bad_json.write_text("{not valid json", encoding="utf-8")
        res_bad_json = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            report_path=rep_p,
            results_path=bad_json,
            reproducibility_run=rep_run,
        )
        c1_bad_json = next(c for c in res_bad_json.criteria if c.criterion_id == "G2-1")
        assert c1_bad_json.status == GateStatus.FAIL
        assert "failed to parse" in c1_bad_json.evidence

        # 6. JSON missing required keys -> FAIL
        missing_keys_json = td / "missing_keys.json"
        missing_keys_json.write_text(json.dumps({"some_key": 123}), encoding="utf-8")
        res_mk = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            report_path=rep_p,
            results_path=missing_keys_json,
            reproducibility_run=rep_run,
        )
        c1_mk = next(c for c in res_mk.criteria if c.criterion_id == "G2-1")
        assert c1_mk.status == GateStatus.FAIL
        assert "missing required key" in c1_mk.evidence

        # 7. Report missing required sections -> FAIL
        incomplete_rep = td / "incomplete.md"
        incomplete_rep.write_text(
            "## A. Executive Summary\n\nSome text but missing rest.", encoding="utf-8"
        )
        res_inc = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            report_path=incomplete_rep,
            results_path=res_p,
            reproducibility_run=rep_run,
        )
        c1_inc = next(c for c in res_inc.criteria if c.criterion_id == "G2-1")
        assert c1_inc.status == GateStatus.FAIL
        assert "missing required section" in c1_inc.evidence


def test_p84_g2_2_mode_execution_statuses_distinction() -> None:
    """Verifies G2-2 distinguishes COMPLETED, FAILED, and NOT_EXECUTED for all 4 modes."""
    all_metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(all_metrics.keys()))

    # 1. All 4 modes COMPLETED -> PASS
    all_completed = {
        "structured": ModeExecutionStatus.COMPLETED,
        "engine-only": ModeExecutionStatus.COMPLETED,
        "e2e": ModeExecutionStatus.COMPLETED,
        "baseline": ModeExecutionStatus.COMPLETED,
    }
    res_pass = evaluate_g2_gate(
        metrics_by_system=all_metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
        mode_execution_statuses=all_completed,
    )
    c2 = next(c for c in res_pass.criteria if c.criterion_id == "G2-2")
    assert c2.status == GateStatus.PASS
    assert "e2e (simulated/partial) [COMPLETED]" in c2.evidence

    # 2. Each mode FAILED -> FAIL
    for failed_mode in ("structured", "engine-only", "e2e", "baseline"):
        statuses = dict(all_completed)
        statuses[failed_mode] = ModeExecutionStatus.FAILED
        res_fail = evaluate_g2_gate(
            metrics_by_system=all_metrics,
            executed_scenarios=sc_defs,
            mode_execution_statuses=statuses,
        )
        c2_fail = next(c for c in res_fail.criteria if c.criterion_id == "G2-2")
        assert c2_fail.status == GateStatus.FAIL
        assert f"{failed_mode} FAILED" in c2_fail.evidence

    # 3. Each mode NOT_EXECUTED -> INCONCLUSIVE
    for not_exec_mode in ("structured", "engine-only", "e2e", "baseline"):
        statuses = dict(all_completed)
        statuses[not_exec_mode] = ModeExecutionStatus.NOT_EXECUTED
        res_not_exec = evaluate_g2_gate(
            metrics_by_system=all_metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            mode_execution_statuses=statuses,
        )
        c2_ne = next(c for c in res_not_exec.criteria if c.criterion_id == "G2-2")
        assert c2_ne.status == GateStatus.INCONCLUSIVE
        assert not_exec_mode in c2_ne.evidence


def test_p84_g2_3_category_coverage_derived_from_actual_scenarios() -> None:
    """Verifies G2-3 derives coverage from executed scenarios, checking prefix agreement."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    # 1. Full canonical execution -> PASS
    res_pass = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
    )
    c3 = next(c for c in res_pass.criteria if c.criterion_id == "G2-3")
    assert c3.status == GateStatus.PASS

    # 2. Scenario ID prefix and declared category mismatch -> FAIL
    mismatched_defs = [dict(s) for s in sc_defs]
    mismatched_defs[0] = {
        "id": "corroborate-999",  # prefix is corroborate-
        "category": "simple_update",  # declared category is simple_update (mismatch!)
        "questions": [{"id": "q1", "expect": {"value": "v1"}}],
    }
    res_mismatch = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=mismatched_defs,
        traces_by_system=traces,
    )
    c3_mm = next(c for c in res_mismatch.criteria if c.criterion_id == "G2-3")
    assert c3_mm.status == GateStatus.FAIL
    assert "category/ID prefix mismatches" in c3_mm.evidence

    # 3. Declared category with no executed scenarios -> FAIL
    no_dispute_defs = [s for s in sc_defs if s["category"] != "dispute"]
    res_no_disp = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=no_dispute_defs,
        traces_by_system=traces,
    )
    c3_nd = next(c for c in res_no_disp.criteria if c.criterion_id == "G2-3")
    assert c3_nd.status == GateStatus.FAIL
    assert "missing required categories" in c3_nd.evidence
    assert "dispute" in c3_nd.evidence

    # 4. Fewer than 40 scenarios -> FAIL
    res_few_sc = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs[:35],
        traces_by_system=traces,
    )
    c3_few_sc = next(c for c in res_few_sc.criteria if c.criterion_id == "G2-3")
    assert c3_few_sc.status == GateStatus.FAIL
    assert "scenario count 35 < canonical threshold 40" in c3_few_sc.evidence

    # 5. Fewer than 100 questions -> FAIL
    single_q_defs = [
        {"id": s["id"], "category": s["category"], "questions": [{"id": "q1"}]}
        for s in sc_defs[:45]
    ]  # 45 questions
    res_few_q = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=single_q_defs,
        traces_by_system=traces,
    )
    c3_few_q = next(c for c in res_few_q.criteria if c.criterion_id == "G2-3")
    assert c3_few_q.status == GateStatus.FAIL
    assert "total questions 45 < canonical threshold 100" in c3_few_q.evidence


def test_p84_g2_4_missing_metrics_must_not_become_zero() -> None:
    """Verifies G2-4 returns INCONCLUSIVE when any metric is missing, without defaulting to zero."""
    sc_defs, _ = _canonical_scenarios_and_categories()

    # 1. All metrics present and superior -> PASS
    m_pass = _canonical_metrics()
    res_pass = evaluate_g2_gate(
        metrics_by_system=m_pass,
        executed_scenarios=sc_defs,
    )
    c4 = next(c for c in res_pass.criteria if c.criterion_id == "G2-4")
    assert c4.status == GateStatus.PASS

    # 2. MM current_accuracy missing (None) -> INCONCLUSIVE
    m_no_mm_curr = _canonical_metrics()
    m_no_mm_curr["MM (structured)"].current_accuracy = None
    res_no_mm_curr = evaluate_g2_gate(
        metrics_by_system=m_no_mm_curr,
        executed_scenarios=sc_defs,
    )
    c4_nmc = next(c for c in res_no_mm_curr.criteria if c.criterion_id == "G2-4")
    assert c4_nmc.status == GateStatus.INCONCLUSIVE
    assert "MM current_accuracy" in c4_nmc.evidence

    # 3. B0 current_accuracy missing (None) -> INCONCLUSIVE
    m_no_b0_curr = _canonical_metrics()
    m_no_b0_curr["B0 (naive RAG)"].current_accuracy = None
    res_no_b0_curr = evaluate_g2_gate(
        metrics_by_system=m_no_b0_curr,
        executed_scenarios=sc_defs,
    )
    c4_nbc = next(c for c in res_no_b0_curr.criteria if c.criterion_id == "G2-4")
    assert c4_nbc.status == GateStatus.INCONCLUSIVE
    assert "B0 current_accuracy" in c4_nbc.evidence

    # 4. MM stale_answer_rate missing (None) -> INCONCLUSIVE
    m_no_mm_stale = _canonical_metrics()
    m_no_mm_stale["MM (structured)"].stale_answer_rate = None
    res_no_mm_stale = evaluate_g2_gate(
        metrics_by_system=m_no_mm_stale,
        executed_scenarios=sc_defs,
    )
    c4_nms = next(c for c in res_no_mm_stale.criteria if c.criterion_id == "G2-4")
    assert c4_nms.status == GateStatus.INCONCLUSIVE
    assert "MM stale_answer_rate" in c4_nms.evidence

    # 5. B0 stale_answer_rate missing (None) -> INCONCLUSIVE
    m_no_b0_stale = _canonical_metrics()
    m_no_b0_stale["B0 (naive RAG)"].stale_answer_rate = None
    res_no_b0_stale = evaluate_g2_gate(
        metrics_by_system=m_no_b0_stale,
        executed_scenarios=sc_defs,
    )
    c4_nbs = next(c for c in res_no_b0_stale.criteria if c.criterion_id == "G2-4")
    assert c4_nbs.status == GateStatus.INCONCLUSIVE
    assert "B0 stale_answer_rate" in c4_nbs.evidence

    # 6. Superiority fails (MM curr <= B0 curr) -> FAIL
    m_fail = _canonical_metrics()
    m_fail["MM (structured)"].current_accuracy = 0.50
    m_fail["B0 (naive RAG)"].current_accuracy = 0.60
    res_fail = evaluate_g2_gate(
        metrics_by_system=m_fail,
        executed_scenarios=sc_defs,
    )
    c4_fail = next(c for c in res_fail.criteria if c.criterion_id == "G2-4")
    assert c4_fail.status == GateStatus.FAIL


def test_p84_g2_5_auditability_and_provenance_strictness() -> None:
    """Verifies G2-5 rejects None for MM metrics, enforces thresholds, and asserts B0 is n/a."""
    sc_defs, _ = _canonical_scenarios_and_categories()

    # 1. MM meeting thresholds (>=0.90, >=0.80) and B0 is None -> PASS
    metrics = _canonical_metrics()
    res_pass = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
    )
    c5 = next(c for c in res_pass.criteria if c.criterion_id == "G2-5")
    assert c5.status == GateStatus.PASS

    # 2. MM decision accuracy is None -> INCONCLUSIVE (do NOT accept None)
    m_none_dec = _canonical_metrics()
    m_none_dec["MM (structured)"].engine_decision_accuracy = None
    res_nd = evaluate_g2_gate(
        metrics_by_system=m_none_dec,
        executed_scenarios=sc_defs,
    )
    c5_nd = next(c for c in res_nd.criteria if c.criterion_id == "G2-5")
    assert c5_nd.status == GateStatus.INCONCLUSIVE
    assert "engine_decision_accuracy is None" in c5_nd.evidence

    # 3. MM provenance correctness is None -> INCONCLUSIVE (do NOT accept None)
    m_none_prov = _canonical_metrics()
    m_none_prov["MM (structured)"].provenance_correctness = None
    res_np = evaluate_g2_gate(
        metrics_by_system=m_none_prov,
        executed_scenarios=sc_defs,
    )
    c5_np = next(c for c in res_np.criteria if c.criterion_id == "G2-5")
    assert c5_np.status == GateStatus.INCONCLUSIVE
    assert "provenance_correctness is None" in c5_np.evidence

    # 4. MM decision accuracy below threshold (<0.90) -> FAIL
    m_low_dec = _canonical_metrics()
    m_low_dec["MM (structured)"].engine_decision_accuracy = 0.88
    res_ld = evaluate_g2_gate(
        metrics_by_system=m_low_dec,
        executed_scenarios=sc_defs,
    )
    c5_ld = next(c for c in res_ld.criteria if c.criterion_id == "G2-5")
    assert c5_ld.status == GateStatus.FAIL

    # 5. MM provenance correctness below threshold (<0.80) -> FAIL
    m_low_prov = _canonical_metrics()
    m_low_prov["MM (structured)"].provenance_correctness = 0.75
    res_lp = evaluate_g2_gate(
        metrics_by_system=m_low_prov,
        executed_scenarios=sc_defs,
    )
    c5_lp = next(c for c in res_lp.criteria if c.criterion_id == "G2-5")
    assert c5_lp.status == GateStatus.FAIL

    # 6. Baseline B0 incorrectly reports numerical audit metrics -> FAIL
    m_b0_leaked = _canonical_metrics()
    m_b0_leaked["B0 (naive RAG)"].engine_decision_accuracy = 0.95
    res_b0_leak = evaluate_g2_gate(
        metrics_by_system=m_b0_leaked,
        executed_scenarios=sc_defs,
    )
    c5_leak = next(c for c in res_b0_leak.criteria if c.criterion_id == "G2-5")
    assert c5_leak.status == GateStatus.FAIL
    assert "Baseline B0 incorrectly reported" in c5_leak.evidence


def test_p84_g2_6_designed_to_lose_actual_failure_accounting() -> None:
    """Verifies G2-6 requires execution outcomes and Section E representation for lose-*."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        rep_p, res_p = _write_valid_artifacts(td)

        # 1. All designed-to-lose scenarios executed and represented in Section E -> PASS
        res_pass = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res_pass.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.PASS

        # 2. Designed-to-lose scenario missing from execution traces -> FAIL
        traces_missing = {
            sys_id: [t for t in trs if t.scenario_id != "lose-003"]
            for sys_id, trs in traces.items()
        }
        res_missing_exec = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces_missing,
            report_path=rep_p,
            results_path=res_p,
        )
        c6_me = next(c for c in res_missing_exec.criteria if c.criterion_id == "G2-6")
        assert c6_me.status == GateStatus.FAIL
        assert "lose-003" in c6_me.evidence
        assert (
            "no execution traces" in c6_me.evidence.lower()
            or "missing" in c6_me.evidence.lower()
            or "unexecuted" in c6_me.evidence.lower()
        )

        # 3. Designed-to-lose scenario executed but absent from report Section E -> FAIL
        rep_p_incomplete, _ = _write_valid_artifacts(
            td / "inc",
            dtl_sids=["lose-001", "lose-002", "lose-004", "lose-005"],  # missing lose-003
        )
        res_missing_rep = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p_incomplete,
            results_path=res_p,
        )
        c6_mr = next(c for c in res_missing_rep.criteria if c.criterion_id == "G2-6")
        assert c6_mr.status == GateStatus.FAIL
        assert "not represented in Section E (Failure Analysis): ['lose-003']" in c6_mr.evidence

        # 4. Expected abstention outcome represented honestly without artificial failure -> PASS
        traces_abstained = {
            sys_id: [
                (
                    QuestionEvalResult(
                        scenario_id=t.scenario_id,
                        question_id=t.question_id,
                        question_type="current",
                        expected={"abstain": True},
                        actual={"abstained": True, "value": None},
                        abstention_correct=True,
                    )
                    if t.scenario_id in ("lose-001", "lose-003")
                    else t
                )
                for t in trs
            ]
            for sys_id, trs in traces.items()
        }
        res_abst = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces_abstained,
            report_path=rep_p,
            results_path=res_p,
        )
        c6_abst = next(c for c in res_abst.criteria if c.criterion_id == "G2-6")
        assert c6_abst.status == GateStatus.PASS


def test_p84_g2_7_compare_actual_per_question_outcomes() -> None:
    """Verifies G2-7 detects differences in answers, citations, conflict flags, and abstentions."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces1 = _canonical_traces(sc_defs, list(metrics.keys()))

    # 1. Identical Run 2 -> PASS
    norm_run2_match = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        retrieval_method="fastembed",
        traces_by_system=traces1,
    )
    res_pass = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        retrieval_method="fastembed",
        traces_by_system=traces1,
        reproducibility_run=norm_run2_match,
    )
    c7 = next(c for c in res_pass.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.PASS
    assert "100% identical normalized outcomes" in c7.evidence

    # 2. Missing Run 2 -> INCONCLUSIVE
    res_none = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        reproducibility_run=None,
    )
    c7_none = next(c for c in res_none.criteria if c.criterion_id == "G2-7")
    assert c7_none.status == GateStatus.INCONCLUSIVE

    # 3. Different actual answer detected even with identical aggregate metrics -> FAIL
    traces_diff_ans = {
        sys_id: [
            (
                QuestionEvalResult(
                    scenario_id=t.scenario_id,
                    question_id=t.question_id,
                    question_type=t.question_type,
                    expected=t.expected,
                    actual={
                        **t.actual,
                        "value": "different_value_3.11",  # Changed answer value!
                    },
                    is_current_correct=t.is_current_correct,
                    is_stale=t.is_stale,
                    provenance_correct=t.provenance_correct,
                    abstention_correct=t.abstention_correct,
                )
                if t.scenario_id == "update-001" and t.question_id == "q1"
                else t
            )
            for t in trs
        ]
        for sys_id, trs in traces1.items()
    }
    norm_diff_ans = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        retrieval_method="fastembed",
        traces_by_system=traces_diff_ans,
    )
    res_diff_ans = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        retrieval_method="fastembed",
        traces_by_system=traces1,
        reproducibility_run=norm_diff_ans,
    )
    c7_da = next(c for c in res_diff_ans.criteria if c.criterion_id == "G2-7")
    assert c7_da.status == GateStatus.FAIL
    assert "outcome discrepancy" in c7_da.evidence
    assert "actual_value" in c7_da.evidence

    # 4. Different citation/provenance identifiers detected -> FAIL
    traces_diff_cit = {
        sys_id: [
            (
                QuestionEvalResult(
                    scenario_id=t.scenario_id,
                    question_id=t.question_id,
                    question_type=t.question_type,
                    expected=t.expected,
                    actual={
                        **t.actual,
                        "cited_memory_ids": ["m999_different"],  # Changed citation!
                    },
                    is_current_correct=t.is_current_correct,
                    is_stale=t.is_stale,
                    provenance_correct=t.provenance_correct,
                    abstention_correct=t.abstention_correct,
                )
                if t.scenario_id == "update-001" and t.question_id == "q1"
                else t
            )
            for t in trs
        ]
        for sys_id, trs in traces1.items()
    }
    norm_diff_cit = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        retrieval_method="fastembed",
        traces_by_system=traces_diff_cit,
    )
    res_diff_cit = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        retrieval_method="fastembed",
        traces_by_system=traces1,
        reproducibility_run=norm_diff_cit,
    )
    c7_dc = next(c for c in res_diff_cit.criteria if c.criterion_id == "G2-7")
    assert c7_dc.status == GateStatus.FAIL
    assert "cited_memory_ids" in c7_dc.evidence

    # 5. Different question-level conflict detection outcome detected -> FAIL
    traces_diff_conflict = {
        sys_id: [
            (
                QuestionEvalResult(
                    scenario_id=t.scenario_id,
                    question_id=t.question_id,
                    question_type=t.question_type,
                    expected=t.expected,
                    actual=t.actual,
                    conflict_flagged=not t.conflict_flagged,  # Inverted conflict flag!
                    is_current_correct=t.is_current_correct,
                    is_stale=t.is_stale,
                    provenance_correct=t.provenance_correct,
                    abstention_correct=t.abstention_correct,
                )
                if t.scenario_id == "dispute-001" and t.question_id == "q1"
                else t
            )
            for t in trs
        ]
        for sys_id, trs in traces1.items()
    }
    norm_diff_conf = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        retrieval_method="fastembed",
        traces_by_system=traces_diff_conflict,
    )
    res_diff_conf = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        retrieval_method="fastembed",
        traces_by_system=traces1,
        reproducibility_run=norm_diff_conf,
    )
    c7_dconf = next(c for c in res_diff_conf.criteria if c.criterion_id == "G2-7")
    assert c7_dconf.status == GateStatus.FAIL
    assert "conflict_flagged" in c7_dconf.evidence

    # 6. Different abstention outcome detected -> FAIL
    traces_diff_abst = {
        sys_id: [
            (
                QuestionEvalResult(
                    scenario_id=t.scenario_id,
                    question_id=t.question_id,
                    question_type=t.question_type,
                    expected=t.expected,
                    actual={**t.actual, "abstained": not t.actual.get("abstained")},
                    is_current_correct=t.is_current_correct,
                    is_stale=t.is_stale,
                    provenance_correct=t.provenance_correct,
                    abstention_correct=t.abstention_correct,
                )
                if t.scenario_id == "lose-001" and t.question_id == "q1"
                else t
            )
            for t in trs
        ]
        for sys_id, trs in traces1.items()
    }
    norm_diff_abst = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        retrieval_method="fastembed",
        traces_by_system=traces_diff_abst,
    )
    res_diff_abst = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        retrieval_method="fastembed",
        traces_by_system=traces1,
        reproducibility_run=norm_diff_abst,
    )
    c7_dabst = next(c for c in res_diff_abst.criteria if c.criterion_id == "G2-7")
    assert c7_dabst.status == GateStatus.FAIL
    assert "abstained" in c7_dabst.evidence


def test_p84_overall_gate_aggregation() -> None:
    """Verifies that overall G2 status is FAIL if any fail, INCONCLUSIVE if any inc, else PASS."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        rep_p, res_p = _write_valid_artifacts(td)
        rep_run = normalize_run_result(
            metrics_by_system=metrics, scenario_ids=sc_ids, traces_by_system=traces
        )

        # 1. All pass -> PASS
        res_pass = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
            reproducibility_run=rep_run,
        )
        assert res_pass.overall_status == GateStatus.PASS

        # 2. One inconclusive, zero fail -> INCONCLUSIVE
        res_inc = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
            reproducibility_run=None,  # G2-7 becomes INCONCLUSIVE
        )
        assert res_inc.overall_status == GateStatus.INCONCLUSIVE

        # 3. One fail, one inconclusive -> FAIL (FAIL takes precedence)
        res_fail = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs[:20],  # G2-3 becomes FAIL (< 40)
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
            reproducibility_run=None,  # G2-7 becomes INCONCLUSIVE
        )
        assert res_fail.overall_status == GateStatus.FAIL


def test_p84_report_generation_and_metrics_matching() -> None:
    """Verifies that Markdown report generation preserves exact metrics and structures."""
    m_mm = EvaluationMetrics(
        system_id="MM",
        system_name="MemoryMesh (structured)",
        current_accuracy=0.952,
        stale_answer_rate=0.012,
        engine_decision_accuracy=0.99,
        provenance_correctness=0.98,
        total_questions=20,
    )
    m_b0 = EvaluationMetrics(
        system_id="B0",
        system_name="Naive RAG",
        current_accuracy=0.421,
        stale_answer_rate=0.385,
        engine_decision_accuracy=None,
        provenance_correctness=None,
        total_questions=20,
    )

    trace_fail = QuestionEvalResult(
        scenario_id="replace-001",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={"value": "3.10", "answer_text": "3.10"},
        is_current_correct=False,
        is_stale=True,
    )

    scenarios = [{"id": "replace-001", "category": "simple_update", "questions": [{"id": "q1"}]}]

    gate_res = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.PASS,
        criteria=[
            GateCriterionResult(
                criterion_id="G2-1",
                title="Artifact Generation",
                target_rule="Rule 1",
                status=GateStatus.PASS,
                evidence="Artifacts verified.",
            )
        ],
        summary="Gate passed.",
    )

    md = generate_evaluation_report(
        metrics_by_system={"MM (structured)": m_mm, "B0 (baseline)": m_b0},
        traces_by_system={"MM (structured)": [], "B0 (baseline)": [trace_fail]},
        scenarios=scenarios,
        gate_result=gate_res,
        timestamp="20261115_000000",
    )

    # 1. Required sections exist
    for sec in REQUIRED_REPORT_SECTIONS:
        assert sec in md

    # 2. Metrics in report match underlying numbers
    assert "95.2%" in md  # MM current accuracy
    assert "42.1%" in md  # B0 current accuracy
    assert "1.2%" in md  # MM stale rate
    assert "38.5%" in md  # B0 stale rate

    # 3. Failed scenarios appear in failure analysis section
    assert "`replace-001`" in md
    assert "`q1`" in md
    assert "Stale / forbidden value present" in md


def test_p84_deterministic_providers_zero_yaml_leakage() -> None:
    """Verifies evaluation providers operate strictly on prompt text without scenario access."""
    e2e_provider = DeterministicEvaluationE2EProvider()
    b0_provider = DeterministicEvaluationBaselineProvider()

    # 1. E2E provider without attribute keywords does not guess or leak answers
    unrelated_q = 'User Question:\n"""\nWhat is the color of the datacenter floor?\n"""'
    res_e2e = json.loads(
        e2e_provider.complete("You are a query parser.", unrelated_q, model="test")
    )
    assert res_e2e["attribute"] is None
    assert res_e2e["confidence"] == 0.0

    # 2. Baseline provider with no retrieved messages returns clean abstention
    empty_msgs_prompt = (
        'User Question:\n"""\nWhat Python version does Server A use?\n"""\n\n'
        "Retrieved Messages:\n"
        "No messages retrieved.\n\n"
        "Instructions:\n..."
    )
    res_b0 = json.loads(
        b0_provider.complete(
            "You are an assistant answering from retrieved context.",
            empty_msgs_prompt,
            model="test",
        )
    )
    assert res_b0["abstained"] is True
    assert res_b0["value"] is None


def test_p84_end_to_end_evaluation_run_and_artifact_generation() -> None:
    """Executes multi-mode evaluation run on hero scenario and verifies full artifact generation."""
    cfg = load_app_config()
    scenarios = load_scenarios(Path("eval/scenarios"), target_scenario="hero-001")
    assert len(scenarios) == 1

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "report.md"
        results_path = Path(tmpdir) / "results.json"

        # 1. Run structured MM
        m_struct, traces_struct = run_system_on_scenarios(
            "MM", "MemoryMesh (structured)", scenarios, cfg, mode="structured"
        )
        # 2. Run B0 baseline
        b0_prov = DeterministicEvaluationBaselineProvider()
        llm_b0 = LLMClient(provider=b0_prov, config=cfg.engine.llm)
        m_b0, traces_b0 = run_system_on_scenarios(
            "B0", "Naive RAG", scenarios, cfg, mode="baseline", llm_client=llm_b0
        )

        metrics_map = {"MM (structured)": m_struct, "B0 (baseline)": m_b0}
        traces_map = {"MM (structured)": traces_struct, "B0 (baseline)": traces_b0}

        gate_res = evaluate_g2_gate(
            metrics_by_system=metrics_map,
            executed_scenarios=scenarios,
            total_questions=7,
            report_path=None,
            results_path=None,
            timestamp="20261115_test",
        )

        report_md = generate_evaluation_report(
            metrics_by_system=metrics_map,
            traces_by_system=traces_map,
            scenarios=scenarios,
            gate_result=gate_res,
            timestamp="20261115_test",
        )
        report_path.write_text(report_md, encoding="utf-8")

        results_data = {
            "timestamp": "20261115_test",
            "scenario_count": 1,
            "metrics": {k: v.model_dump() for k, v in metrics_map.items()},
            "scenarios": ["hero-001"],
            "g2_gate": gate_res.model_dump(),
        }
        results_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")

        # Verify artifacts exist and are valid
        assert report_path.exists() and report_path.stat().st_size > 500
        assert results_path.exists() and results_path.stat().st_size > 100

        # Verify JSON can be parsed
        parsed_json = json.loads(results_path.read_text(encoding="utf-8"))
        assert parsed_json["scenario_count"] == 1
        assert "MM (structured)" in parsed_json["metrics"]
        assert "B0 (baseline)" in parsed_json["metrics"]
        assert "g2_gate" in parsed_json


def test_p84_deterministic_reproducibility() -> None:
    """Verifies that repeated runs produce identical metrics and normalize_run_result matches."""
    cfg = load_app_config()
    scenarios = load_scenarios(Path("eval/scenarios"), target_scenario="hero-001")

    m1, traces1 = run_system_on_scenarios("MM", "Structured", scenarios, cfg, mode="structured")
    m2, traces2 = run_system_on_scenarios("MM", "Structured", scenarios, cfg, mode="structured")

    assert m1.current_accuracy == m2.current_accuracy
    assert m1.as_of_accuracy == m2.as_of_accuracy
    assert m1.stale_answer_rate == m2.stale_answer_rate
    assert m1.engine_decision_accuracy == m2.engine_decision_accuracy

    norm1 = normalize_run_result(
        metrics_by_system={"MM (structured)": m1},
        scenario_ids=["hero-001"],
        traces_by_system={"MM (structured)": traces1},
    )
    norm2 = normalize_run_result(
        metrics_by_system={"MM (structured)": m2},
        scenario_ids=["hero-001"],
        traces_by_system={"MM (structured)": traces2},
    )
    assert norm1.metrics_summary == norm2.metrics_summary
    assert norm1.scenario_ids == norm2.scenario_ids
    assert norm1.failure_keys == norm2.failure_keys
    assert norm1.total_failures == norm2.total_failures
    assert norm1.question_outcomes == norm2.question_outcomes


# ---------------------------------------------------------------------------
# P8.4 G2 Evidence Validation Regression Tests (G2-3, G2-6, G2-7)
# ---------------------------------------------------------------------------


def test_p84_g2_3_incomplete_question_traces_in_scenario_fails() -> None:
    """Verifies G2-3 fails if a scenario has incomplete question traces."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    # Drop q2 for update-001 in all systems
    for sys_id in traces:
        traces[sys_id] = [
            t
            for t in traces[sys_id]
            if not (t.scenario_id == "update-001" and t.question_id == "q2")
        ]

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
    )
    c3 = next(c for c in res.criteria if c.criterion_id == "G2-3")
    assert c3.status == GateStatus.FAIL
    assert "incomplete question execution" in c3.evidence
    assert "update-001" in c3.evidence


def test_p84_g2_3_missing_question_in_one_mode_fails() -> None:
    """Verifies G2-3 fails if even one canonical evaluation mode is missing a question trace."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    # Drop one question trace ONLY from B0 (naive RAG)
    traces["B0 (naive RAG)"] = [
        t
        for t in traces["B0 (naive RAG)"]
        if not (t.scenario_id == "ordering-001" and t.question_id == "q1")
    ]

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
    )
    c3 = next(c for c in res.criteria if c.criterion_id == "G2-3")
    assert c3.status == GateStatus.FAIL
    assert "missing question traces" in c3.evidence
    assert "B0 (naive RAG)" in c3.evidence


def test_p84_g2_3_duplicate_trace_for_question_fails() -> None:
    """Verifies G2-3 detects duplicate question traces in a system and fails."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    # Duplicate trace for update-001:q1 in MM (structured)
    dup_trace = traces["MM (structured)"][0].model_copy()
    traces["MM (structured)"].append(dup_trace)

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
    )
    c3 = next(c for c in res.criteria if c.criterion_id == "G2-3")
    assert c3.status == GateStatus.FAIL
    assert "duplicate question traces detected" in c3.evidence


def test_p84_g2_3_full_question_coverage_passes() -> None:
    """Verifies G2-3 passes when all questions in all scenarios are executed across all modes."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
    )
    c3 = next(c for c in res.criteria if c.criterion_id == "G2-3")
    assert c3.status == GateStatus.PASS
    assert "all verified executed with zero missing, duplicate, or mismatched" in c3.evidence


def test_p84_g2_3_missing_evidence_returns_inconclusive() -> None:
    """Verifies G2-3 returns INCONCLUSIVE when traces_by_system is missing or None."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=None,
    )
    c3 = next(c for c in res.criteria if c.criterion_id == "G2-3")
    assert c3.status == GateStatus.INCONCLUSIVE
    assert "unavailable for empirical coverage verification" in c3.evidence


def test_p84_g2_6_dtl_contradicting_section_e_status_fails() -> None:
    """Verifies G2-6 fails if Section E reports a failing or unexecuted status for a lose-* case."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        rep_p = td / "report.md"
        res_p = td / "results.json"

        # Section E table reports Observed Failure for lose-002
        sec_e = (
            "## E. Failure Analysis\n\n### 2. Accounting\n\n"
            "| Scenario ID | Invariant | Outcome | Status |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| `lose-001` | Invariant | Safe abstention | Honest Abstention (Pass) |\n"
            "| `lose-002` | Invariant | Conflict not flagged | Observed Failure |\n"
            "| `lose-003` | Invariant | Safe abstention | Honest Abstention (Pass) |\n"
            "| `lose-004` | Invariant | R-REJECT enforced | Honest Defense (Pass) |\n"
            "| `lose-005` | Invariant | Review flagged | Honest Review Flag (Pass) |\n"
        )
        body = [
            "## A. Executive Summary\nText",
            "## B. Evaluation Methodology\nText",
            "## C. Quantitative Results\nText",
            "## D. Scenario-Group Analysis\nText",
            sec_e,
            "## F. Limitations\nText",
            "## G. Conclusion\nText",
            "## H. G2 Evaluation Gate Verification\nText",
        ]
        rep_p.write_text("# Report\n\n" + "\n\n".join(body), encoding="utf-8")
        res_p.write_text(json.dumps({"g2_gate": {}}), encoding="utf-8")

        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.FAIL
        assert "lose-002" in c6.evidence
        assert "failing or unexecuted status in Section E" in c6.evidence


def test_p84_g2_6_lose_004_missing_rejection_evidence_fails() -> None:
    """Verifies G2-6 fails if lose-004 trace is missing R-REJECT decision evidence."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        rep_p, res_p = _write_valid_artifacts(td)

        # Strip R-REJECT from lose-004 actual rules and decision_records
        for t in traces["MM (structured)"]:
            if t.scenario_id == "lose-004":
                t.actual = {
                    **t.actual,
                    "rules": ["R-INSERT"],
                    "decision_records": [
                        {
                            "rule_id": "R-INSERT",
                            "action": "INSERT",
                            "claim_value": "3.12",
                            "claim_subject": "Server A",
                            "claim_attribute": "python_version",
                        }
                    ],
                }

        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.FAIL
        assert "lose-004" in c6.evidence
        assert "rejection" in c6.evidence.lower() or "missing" in c6.evidence.lower()


def test_p84_g2_6_lose_005_missing_dependency_review_fails() -> None:
    """Verifies G2-6 fails if lose-005 trace is missing review flag or dependent records."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        rep_p, res_p = _write_valid_artifacts(td)

        # Remove dependents and unflag review for lose-005
        for t in traces["MM (structured)"]:
            if t.scenario_id == "lose-005":
                t.actual = {
                    **t.actual,
                    "needs_review_flagged": False,
                    "dependents": [],
                    "dependent_records": [],
                }

        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.FAIL
        assert "lose-005" in c6.evidence
        assert "needs_review" in c6.evidence or "dependent" in c6.evidence.lower()


def test_p84_g2_6_b0_vs_mm_selection_separation() -> None:
    """Verifies G2-6 explicitly checks MemoryMesh structured traces, ignoring B0 failure states."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        rep_p, res_p = _write_valid_artifacts(td)

        # Corrupt B0 traces for all lose-* cases (e.g. B0 did not abstain or flag conflict)
        for t in traces["B0 (naive RAG)"]:
            if t.scenario_id.startswith("lose-"):
                t.actual = {
                    "value": "corrupted_hallucination",
                    "abstained": False,
                    "conflict_flagged": False,
                    "needs_review_flagged": False,
                    "rules": [],
                }

        # MemoryMesh structured traces remain valid
        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.PASS


def test_p84_g2_6_all_correct_outcomes_pass() -> None:
    """Verifies G2-6 passes when all 5 designed-to-lose scenarios satisfy their invariants."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        rep_p, res_p = _write_valid_artifacts(td)

        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.PASS
        assert "empirically executed with verified invariants" in c6.evidence


def test_p84_g2_7_duplicate_trace_in_run1_rejects() -> None:
    """Verifies G2-7 fails if Run 1 contains duplicate question traces."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces1 = _canonical_traces(sc_defs, list(metrics.keys()))

    # Duplicate trace in Run 1
    dup = traces1["MM (structured)"][0].model_copy()
    traces1_with_dup = {k: list(v) for k, v in traces1.items()}
    traces1_with_dup["MM (structured)"].append(dup)

    rep_run = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        traces_by_system=traces1,
    )

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces1_with_dup,
        reproducibility_run=rep_run,
    )
    c7 = next(c for c in res.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.FAIL
    assert "Run 1 contains duplicate question traces" in c7.evidence


def test_p84_g2_7_duplicate_trace_in_run2_rejects() -> None:
    """Verifies G2-7 fails if Run 2 contains duplicate question traces."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces1 = _canonical_traces(sc_defs, list(metrics.keys()))

    # Duplicate trace in Run 2
    dup = traces1["MM (structured)"][0].model_copy()
    traces2_with_dup = {k: list(v) for k, v in traces1.items()}
    traces2_with_dup["MM (structured)"].append(dup)

    rep_run_with_dup = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        traces_by_system=traces2_with_dup,
    )

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces1,
        reproducibility_run=rep_run_with_dup,
    )
    c7 = next(c for c in res.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.FAIL
    assert "Run 2 contains duplicate question traces" in c7.evidence


def test_p84_g2_7_missing_question_outcome_fails() -> None:
    """Verifies G2-7 fails if Run 2 is missing an expected question outcome."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces1 = _canonical_traces(sc_defs, list(metrics.keys()))

    # Drop one question trace in Run 2
    traces2_incomplete = {
        sys_id: [t for t in trs if not (t.scenario_id == "update-002" and t.question_id == "q1")]
        for sys_id, trs in traces1.items()
    }
    rep_run_incomplete = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        traces_by_system=traces2_incomplete,
    )

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces1,
        reproducibility_run=rep_run_incomplete,
    )
    c7 = next(c for c in res.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.FAIL
    assert "missing" in c7.evidence or "discrepancy" in c7.evidence


def test_p84_g2_7_differing_dependency_impact_fails() -> None:
    """Verifies G2-7 detects and fails on differences in dependency impact records."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces1 = _canonical_traces(sc_defs, list(metrics.keys()))

    traces2_diff_deps = {
        sys_id: [
            (
                QuestionEvalResult(
                    scenario_id=t.scenario_id,
                    question_id=t.question_id,
                    question_type=t.question_type,
                    expected=t.expected,
                    actual={
                        **t.actual,
                        "dependents": [("ChangedService", "changed_attr")],  # Changed dependent!
                    },
                    is_current_correct=t.is_current_correct,
                    is_stale=t.is_stale,
                    provenance_correct=t.provenance_correct,
                    abstention_correct=t.abstention_correct,
                )
                if t.scenario_id == "dependency-001" and t.question_id == "q1"
                else t
            )
            for t in trs
        ]
        for sys_id, trs in traces1.items()
    }

    rep_run_diff_deps = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        traces_by_system=traces2_diff_deps,
    )

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces1,
        reproducibility_run=rep_run_diff_deps,
    )
    c7 = next(c for c in res.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.FAIL
    assert "dependents" in c7.evidence


def test_p84_g2_7_differing_needs_review_fails() -> None:
    """Verifies G2-7 detects and fails on differences in needs_review_flagged outcomes."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces1 = _canonical_traces(sc_defs, list(metrics.keys()))

    traces2_diff_rev = {
        sys_id: [
            (
                QuestionEvalResult(
                    scenario_id=t.scenario_id,
                    question_id=t.question_id,
                    question_type=t.question_type,
                    expected=t.expected,
                    actual={
                        **t.actual,
                        "needs_review_flagged": not t.actual.get(
                            "needs_review_flagged", False
                        ),  # Inverted!
                    },
                    is_current_correct=t.is_current_correct,
                    is_stale=t.is_stale,
                    provenance_correct=t.provenance_correct,
                    abstention_correct=t.abstention_correct,
                )
                if t.scenario_id == "dependency-001" and t.question_id == "q1"
                else t
            )
            for t in trs
        ]
        for sys_id, trs in traces1.items()
    }

    rep_run_diff_rev = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        traces_by_system=traces2_diff_rev,
    )

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces1,
        reproducibility_run=rep_run_diff_rev,
    )
    c7 = next(c for c in res.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.FAIL
    assert "needs_review_flagged" in c7.evidence


def test_p84_g2_7_identical_complete_outcomes_passes() -> None:
    """Verifies G2-7 passes when both runs produce identical complete per-question outcomes."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces1 = _canonical_traces(sc_defs, list(metrics.keys()))

    rep_run_identical = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        traces_by_system=traces1,
    )

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces1,
        reproducibility_run=rep_run_identical,
    )
    c7 = next(c for c in res.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.PASS
    assert "100% identical normalized outcomes" in c7.evidence


# ---------------------------------------------------------------------------
# P8.4 Final Review: Strengthen Designed-to-Lose Evidence Validation Tests
# ---------------------------------------------------------------------------


def test_p84_g2_6_explicit_successful_abstention_passes() -> None:
    """Verifies explicit successful abstention (abstained=True, value=None) passes G2-6."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_001 = next(s for s in sc_defs if s["id"] == "lose-001")
    trace = QuestionEvalResult(
        scenario_id="lose-001",
        question_id="q1",
        question_type="current",
        expected={"abstain": True},
        actual={"abstained": True, "value": None},
        abstention_correct=True,
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-001", sc_def_001, trace)
    assert is_ok is True
    assert "Safe abstention verified" in outcome_text
    assert "Pass" in status


def test_p84_g2_6_abstention_false_with_none_value_fails() -> None:
    """Verifies ambiguous response (abstained=False, value=None) fails G2-6."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_001 = next(s for s in sc_defs if s["id"] == "lose-001")
    trace = QuestionEvalResult(
        scenario_id="lose-001",
        question_id="q1",
        question_type="current",
        expected={"abstain": True},
        actual={"abstained": False, "value": None},
        abstention_correct=False,
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-001", sc_def_001, trace)
    assert is_ok is False
    assert "Ambiguous non-abstaining null return" in outcome_text
    assert status == "Observed Failure"

    # Also verify G2 gate fails
    metrics = _canonical_metrics()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    for t in traces["MM (structured)"]:
        if t.scenario_id == "lose-001":
            t.actual = {"abstained": False, "value": None}
    with tempfile.TemporaryDirectory() as tmp:
        rep_p, res_p = _write_valid_artifacts(Path(tmp))
        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.FAIL
        assert "lose-001" in c6.evidence


def test_p84_g2_6_query_execution_error_with_no_value_fails() -> None:
    """Verifies query execution error with no value fails or is inconclusive, never passes."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_001 = next(s for s in sc_defs if s["id"] == "lose-001")
    trace = QuestionEvalResult(
        scenario_id="lose-001",
        question_id="q1",
        question_type="current",
        expected={"abstain": True},
        actual={"abstained": False, "value": None, "error": "Database query timeout"},
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-001", sc_def_001, trace)
    assert is_ok is False
    assert "Query execution error occurred" in outcome_text
    assert status == "Execution Error (Fail)"

    metrics = _canonical_metrics()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    for t in traces["MM (structured)"]:
        if t.scenario_id == "lose-001":
            t.actual = {"abstained": False, "value": None, "error": "Database query timeout"}
    with tempfile.TemporaryDirectory() as tmp:
        rep_p, res_p = _write_valid_artifacts(Path(tmp))
        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.FAIL


def test_p84_g2_6_non_abstaining_answer_to_unsupported_query_fails() -> None:
    """Verifies returning a value instead of abstaining for unsupported query fails G2-6."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_003 = next(s for s in sc_defs if s["id"] == "lose-003")
    trace = QuestionEvalResult(
        scenario_id="lose-003",
        question_id="q1",
        question_type="as_of",
        expected={"abstain": True},
        actual={"abstained": False, "value": "2020-01-01 Fabricated Fact"},
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-003", sc_def_003, trace)
    assert is_ok is False
    assert "Failed to abstain" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_claim_specific_rejection_passes() -> None:
    """Verifies rejection of the specific rumor claim (3.9) with R-REJECT passes G2-6."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-INSERT",
                    "action": "INSERT",
                    "claim_value": "3.12",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                },
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                    "explanation": "Lower authority untrusted source rejected",
                },
            ],
            "rules": ["R-INSERT", "R-REJECT"],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "context": {},
                    "context_key": "",
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                }
            ],
            "memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "context": {},
                    "context_key": "",
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                },
                {
                    "id": "2",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.9",
                    "status": "rejected",
                    "is_active": False,
                    "context": {},
                    "context_key": "",
                    "valid_from": "2026-01-12T11:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-12T11:00:00Z",
                    "closed_at": None,
                },
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "slot_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    },
                    {
                        "id": "2",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.9",
                        "status": "rejected",
                        "is_active": False,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-12T11:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-12T11:00:00Z",
                        "closed_at": None,
                    },
                ],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is True
    assert "R-REJECT enforced on claim '3.9'" in outcome_text
    assert status == "Honest Defense (Pass)"


def test_p84_g2_6_lose_004_empty_decision_records_with_aggregate_rules_fails() -> None:
    """Verifies empty decision_records fails even if aggregate rules contains R-REJECT."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "decision_records": [],  # Empty decision records!
            "rules": ["R-INSERT", "R-REJECT"],
            "ingestion_rules": ["R-REJECT"],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert "Decision evidence missing" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_lose_004_different_subject_fails() -> None:
    """Verifies that R-REJECT for a different subject fails lose-004."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server B",  # Different subject!
                    "claim_attribute": "python_version",
                },
            ],
            "rules": ["R-REJECT"],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert "Claim-specific rejection missing" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_lose_004_different_attribute_fails() -> None:
    """Verifies that R-REJECT for a different attribute fails lose-004."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "os_version",  # Different attribute!
                },
            ],
            "rules": ["R-REJECT"],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert "Claim-specific rejection missing" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_rumor_accepted_despite_unrelated_rejection_fails() -> None:
    """Verifies that if the rumor (3.9) was accepted into active memory, lose-004 fails."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.9",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "rules": ["R-REJECT"],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert "Rumor accepted despite rejection record" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_lose_004_decision_evidence_missing_never_passes() -> None:
    """Verifies that missing decision evidence in lose-004 never passes."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "decision_records": [],
            "rules": [],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert "Decision evidence missing" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_expected_dependent_flagged_passes() -> None:
    """Verifies expected dependent (Library X, compatibility_note) identified and flagged passes."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_005 = next(s for s in sc_defs if s["id"] == "lose-005")
    trace = QuestionEvalResult(
        scenario_id="lose-005",
        question_id="q1",
        question_type="current",
        expected={"dependents": [("Library X", "compatibility_note")]},
        actual={
            "value": None,
            "needs_review_flagged": True,
            "dependents": [("Library X", "compatibility_note")],
            "dependent_records": [
                {
                    "id": "m_dep_1",
                    "entity": "Library X",
                    "attribute": "compatibility_note",
                    "needs_review": True,
                }
            ],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-005", sc_def_005, trace)
    assert is_ok is True
    assert "Downstream impact flagged for review" in outcome_text
    assert status == "Honest Review Flag (Pass)"


def test_p84_g2_6_only_unrelated_dependent_flagged_fails() -> None:
    """Verifies that flagging only an unrelated dependent fails lose-005."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_005 = next(s for s in sc_defs if s["id"] == "lose-005")
    trace = QuestionEvalResult(
        scenario_id="lose-005",
        question_id="q1",
        question_type="current",
        expected={"dependents": [("Library X", "compatibility_note")]},
        actual={
            "value": None,
            "needs_review_flagged": True,
            "dependents": [("UnrelatedService", "unrelated_attribute")],
            "dependent_records": [
                {
                    "entity": "UnrelatedService",
                    "attribute": "unrelated_attribute",
                    "needs_review": True,
                }
            ],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-005", sc_def_005, trace)
    assert is_ok is False
    assert "Expected dependent 'Library X.compatibility_note' missing" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_lose_005_no_dependent_records_fails() -> None:
    """Verifies that global needs_review_flagged=True without dependent_records fails lose-005."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_005 = next(s for s in sc_defs if s["id"] == "lose-005")
    trace = QuestionEvalResult(
        scenario_id="lose-005",
        question_id="q1",
        question_type="impact",
        expected={"dependents": [("Library X", "compatibility_note")]},
        actual={
            "value": None,
            "needs_review_flagged": True,  # Global flag is True
            "dependents": [("Library X", "compatibility_note")],
            "dependent_records": [],  # But dependent_records is empty!
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-005", sc_def_005, trace)
    assert is_ok is False
    assert "Per-dependent evidence missing" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_lose_005_record_missing_needs_review_fails() -> None:
    """Verifies that dependent record missing needs_review key fails lose-005."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_005 = next(s for s in sc_defs if s["id"] == "lose-005")
    trace = QuestionEvalResult(
        scenario_id="lose-005",
        question_id="q1",
        question_type="impact",
        expected={"dependents": [("Library X", "compatibility_note")]},
        actual={
            "value": None,
            "needs_review_flagged": True,
            "dependents": [("Library X", "compatibility_note")],
            "dependent_records": [
                {
                    "entity": "Library X",
                    "attribute": "compatibility_note",
                    # "needs_review" field missing!
                }
            ],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-005", sc_def_005, trace)
    assert is_ok is False
    assert "missing explicit 'needs_review' field" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_lose_005_record_needs_review_false_fails() -> None:
    """Verifies that dependent record with needs_review=False fails lose-005."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_005 = next(s for s in sc_defs if s["id"] == "lose-005")
    trace = QuestionEvalResult(
        scenario_id="lose-005",
        question_id="q1",
        question_type="impact",
        expected={"dependents": [("Library X", "compatibility_note")]},
        actual={
            "value": None,
            "needs_review_flagged": True,
            "dependents": [("Library X", "compatibility_note")],
            "dependent_records": [
                {
                    "entity": "Library X",
                    "attribute": "compatibility_note",
                    "needs_review": False,  # Explicit False!
                }
            ],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-005", sc_def_005, trace)
    assert is_ok is False
    assert "does not have needs_review=True" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_lose_005_only_unrelated_records_fails() -> None:
    """Verifies that having only unrelated dependent records fails lose-005."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_005 = next(s for s in sc_defs if s["id"] == "lose-005")
    trace = QuestionEvalResult(
        scenario_id="lose-005",
        question_id="q1",
        question_type="impact",
        expected={"dependents": [("Library X", "compatibility_note")]},
        actual={
            "value": None,
            "needs_review_flagged": True,
            "dependents": [("UnrelatedService", "unrelated_attribute")],
            "dependent_records": [
                {
                    "entity": "UnrelatedService",
                    "attribute": "unrelated_attribute",
                    "needs_review": True,
                }
            ],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-005", sc_def_005, trace)
    assert is_ok is False
    assert "Expected dependent 'Library X.compatibility_note' missing" in outcome_text
    assert status == "Observed Failure"


def test_p84_g2_6_lose_005_exact_record_passes() -> None:
    """Verifies exact expected dependent record with needs_review=True passes lose-005."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_005 = next(s for s in sc_defs if s["id"] == "lose-005")
    trace = QuestionEvalResult(
        scenario_id="lose-005",
        question_id="q1",
        question_type="impact",
        expected={"dependents": [("Library X", "compatibility_note")]},
        actual={
            "value": None,
            "needs_review_flagged": True,
            "dependents": [("Library X", "compatibility_note")],
            "dependent_records": [
                {
                    "id": "m_dep_1",
                    "entity": "Library X",
                    "attribute": "compatibility_note",
                    "needs_review": True,
                }
            ],
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-005", sc_def_005, trace)
    assert is_ok is True
    assert "Downstream impact flagged for review" in outcome_text
    assert status == "Honest Review Flag (Pass)"


def test_p84_g2_6_selects_invariant_question_not_first_trace() -> None:
    """Verifies that G2-6 selects the invariant question trace even if it is not first."""
    sc_def = {
        "id": "lose-003",
        "category": "designed_to_lose",
        "questions": [
            {
                "id": "q0",
                "type": "current",
                "text": "Normal question",
                "expect": {"value": "3.12"},
            },
            {
                "id": "q1",
                "type": "as_of",
                "text": "Invariant abstention question",
                "expect": {"abstain": True, "value": None},
            },
        ],
    }

    t_q0 = QuestionEvalResult(
        scenario_id="lose-003",
        question_id="q0",
        question_type="current",
        expected={"value": "3.12"},
        actual={"value": "3.12", "abstained": False},  # Non-abstaining!
    )
    t_q1 = QuestionEvalResult(
        scenario_id="lose-003",
        question_id="q1",
        question_type="as_of",
        expected={"abstain": True, "value": None},
        actual={"value": None, "abstained": True},  # Correct abstention!
    )

    # Question traces list where q0 is first and q1 is second
    traces = [t_q0, t_q1]

    # Helper must select q1, not q0
    selected, err = select_dtl_invariant_trace("lose-003", sc_def, traces)
    assert err is None
    assert selected is not None
    assert selected.question_id == "q1"

    # Verify G2 gate correctly passes using q1
    metrics = _canonical_metrics()
    canonical_defs, _ = _canonical_scenarios_and_categories()
    all_defs = [s if s["id"] != "lose-003" else sc_def for s in canonical_defs]
    gate_traces = _canonical_traces(all_defs, list(metrics.keys()))

    with tempfile.TemporaryDirectory() as tmp:
        rep_p, res_p = _write_valid_artifacts(Path(tmp))
        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=all_defs,
            traces_by_system=gate_traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.PASS


def test_p84_g2_6_missing_invariant_question_trace_fails() -> None:
    """Verifies that missing the invariant question trace fails G2-6."""
    sc_def = {
        "id": "lose-003",
        "category": "designed_to_lose",
        "questions": [
            {
                "id": "q0",
                "type": "current",
                "text": "Normal question",
                "expect": {"value": "3.12"},
            },
            {
                "id": "q1",
                "type": "as_of",
                "text": "Invariant abstention question",
                "expect": {"abstain": True, "value": None},
            },
        ],
    }
    t_q0 = QuestionEvalResult(
        scenario_id="lose-003",
        question_id="q0",
        question_type="current",
        expected={"value": "3.12"},
        actual={"value": "3.12", "abstained": False},
    )

    # Trace only contains q0, missing q1
    selected, err = select_dtl_invariant_trace("lose-003", sc_def, [t_q0])
    assert selected is None
    assert err is not None
    assert "Invariant question 'q1' missing" in err

    # Gate must fail G2-6
    metrics = _canonical_metrics()
    canonical_defs, _ = _canonical_scenarios_and_categories()
    all_defs = [s if s["id"] != "lose-003" else sc_def for s in canonical_defs]
    gate_traces = _canonical_traces(all_defs, list(metrics.keys()))
    # Remove q1 from lose-003 traces in MM
    for sys_name in gate_traces:
        gate_traces[sys_name] = [
            t
            for t in gate_traces[sys_name]
            if not (t.scenario_id == "lose-003" and t.question_id == "q1")
        ]

    with tempfile.TemporaryDirectory() as tmp:
        rep_p, res_p = _write_valid_artifacts(Path(tmp))
        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=all_defs,
            traces_by_system=gate_traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.FAIL
        assert "selection failed" in c6.evidence or "missing" in c6.evidence


def test_p84_g2_6_duplicate_invariant_question_trace_fails() -> None:
    """Verifies that duplicate traces for the invariant question fail and cannot pass silently."""
    sc_def = {
        "id": "lose-003",
        "category": "designed_to_lose",
        "questions": [
            {
                "id": "q1",
                "type": "as_of",
                "text": "Invariant abstention question",
                "expect": {"abstain": True, "value": None},
            },
        ],
    }
    t_q1_a = QuestionEvalResult(
        scenario_id="lose-003",
        question_id="q1",
        question_type="as_of",
        expected={"abstain": True, "value": None},
        actual={"value": None, "abstained": True},
    )
    t_q1_b = QuestionEvalResult(
        scenario_id="lose-003",
        question_id="q1",
        question_type="as_of",
        expected={"abstain": True, "value": None},
        actual={"value": None, "abstained": True},
    )

    selected, err = select_dtl_invariant_trace("lose-003", sc_def, [t_q1_a, t_q1_b])
    assert selected is None
    assert err is not None
    assert "Duplicate traces found for invariant question 'q1'" in err


def test_p84_g2_6_report_claims_pass_when_verified_outcome_fails_rejects() -> None:
    """Verifies that Section E claiming Pass when verified outcome fails is rejected by G2-6."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    # Corrupt lose-001 in actual trace (abstained=False, value=None)
    for t in traces["MM (structured)"]:
        if t.scenario_id == "lose-001":
            t.actual = {"abstained": False, "value": None}

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        # But report.md claims Pass for lose-001
        rep_p, res_p = _write_valid_artifacts(td)

        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.FAIL
        assert "lose-001" in c6.evidence
        assert (
            "claimed Pass in Section E but failed actual invariant" in c6.evidence
            or "failed invariant verification" in c6.evidence
        )


def test_p84_g2_6_missing_outcome_cannot_be_represented_as_abstention() -> None:
    """Verifies missing outcome is reported as UNEXECUTED, not as successful abstention."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_001 = next(s for s in sc_defs if s["id"] == "lose-001")

    # When trace is None
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-001", sc_def_001, None)
    assert is_ok is False
    assert "No execution traces recorded" in outcome_text
    assert "**`UNEXECUTED`**" in status
    assert "Pass" not in status
    assert "Abstention" not in status


# -----------------------------------------------------------------------------
# Active-Memory Verification Regression Tests (P8.4)
# -----------------------------------------------------------------------------


def test_p84_g2_6_lose_004_312_active_and_39_absent_passes() -> None:
    """Verifies that 3.12 active and 3.9 absent in active memory passes lose-004."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "context": {},
                    "context_key": "",
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                }
            ],
            "memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "context": {},
                    "context_key": "",
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "slot_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is True
    assert "Honest Defense (Pass)" in status
    assert "R-REJECT enforced" in outcome_text


def test_p84_g2_6_lose_004_312_returned_but_both_active_fails() -> None:
    """Verifies that 3.12 returned but both 3.12 and 3.9 active fails lose-004."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",  # Returned value is 3.12
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                },
                {
                    "id": "2",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.9",
                    "status": "active",  # Both active!
                    "is_active": True,
                    "valid_from": "2026-01-12T11:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-12T11:00:00Z",
                    "closed_at": None,
                },
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    },
                    {
                        "id": "2",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.9",
                        "status": "active",
                        "is_active": True,
                        "valid_from": "2026-01-12T11:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-12T11:00:00Z",
                        "closed_at": None,
                    },
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Rejected value '3.9' is present in active memory" in outcome_text


def test_p84_g2_6_lose_004_39_active_and_312_active_fails() -> None:
    """Verifies that 3.9 active alongside 3.12 active in memory_records fails lose-004."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                },
                {
                    "id": "2",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.9",
                    "status": "active",
                    "is_active": True,
                    "valid_from": "2026-01-12T11:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-12T11:00:00Z",
                    "closed_at": None,
                },
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    },
                    {
                        "id": "2",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.9",
                        "status": "active",
                        "is_active": True,
                        "valid_from": "2026-01-12T11:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-12T11:00:00Z",
                        "closed_at": None,
                    },
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Rejected value '3.9' is present in active memory" in outcome_text


def test_p84_g2_6_lose_004_active_memory_evidence_missing_fails() -> None:
    """Verifies that missing active-memory evidence fails lose-004."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            # repository_retrieval is omitted
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Active-memory evidence missing" in outcome_text


def test_p84_g2_6_lose_004_39_present_only_as_rejected_historical_passes() -> None:
    """Verifies 3.9 present only as rejected/historical evidence, not active, passes."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "slot_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    },
                    {
                        "id": "2",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.9",
                        "status": "rejected",
                        "is_active": False,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-12T11:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-12T11:00:00Z",
                        "closed_at": None,
                    },
                    {
                        "id": "3",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.8",
                        "status": "historical",
                        "is_active": False,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2025-01-01T00:00:00Z",
                        "valid_to": "2026-01-10T09:00:00Z",
                        "recorded_at": "2025-01-01T00:00:00Z",
                        "closed_at": None,
                    },
                ],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is True
    assert status == "Honest Defense (Pass)"
    assert "R-REJECT enforced" in outcome_text


# -----------------------------------------------------------------------------
# DTL Invariant Question ID Derivation Regression Tests (P8.4)
# -----------------------------------------------------------------------------


def test_p84_g2_6_find_dtl_invariant_question_id_selects_matching_question() -> None:
    """Verifies scenario definition containing invariant question selects it accurately."""
    sc_def = {
        "id": "lose-001",
        "category": "designed_to_lose",
        "questions": [
            {"id": "q_dummy", "text": "Dummy", "expect": {"value": "x"}},
            {"id": "q_inv", "text": "Abstain", "expect": {"abstain": True}},
        ],
    }
    qid = find_dtl_invariant_question_id("lose-001", sc_def)
    assert qid == "q_inv"


def test_p84_g2_6_find_dtl_invariant_question_id_no_match_returns_none_and_fails_gate() -> None:
    """Verifies questions present but none matching invariant returns None and fails G2-6."""
    sc_def = {
        "id": "lose-001",
        "category": "designed_to_lose",
        "questions": [
            {"id": "q_one", "text": "Query 1", "expect": {"value": "1"}},
            {"id": "q_two", "text": "Query 2", "expect": {"value": "2"}},
        ],
    }
    # Must NOT fall back to "q1" from canonical defaults
    qid = find_dtl_invariant_question_id("lose-001", sc_def)
    assert qid is None

    # select_dtl_invariant_trace must fail
    trace = QuestionEvalResult(
        scenario_id="lose-001",
        question_id="q_one",
        question_type="current",
        expected={"value": "1"},
        actual={"value": None, "abstained": True},
    )
    selected, err = select_dtl_invariant_trace("lose-001", sc_def, [trace])
    assert selected is None
    assert err is not None
    assert "Could not determine invariant-testing question ID" in err

    # evaluate_g2_gate must fail G2-6
    metrics = _canonical_metrics()
    canonical_defs, _ = _canonical_scenarios_and_categories()
    all_defs = [s if s["id"] != "lose-001" else sc_def for s in canonical_defs]
    traces = _canonical_traces(all_defs, list(metrics.keys()))

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        rep_p, res_p = _write_valid_artifacts(td)
        res = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=all_defs,
            traces_by_system=traces,
            report_path=rep_p,
            results_path=res_p,
        )
        c6 = next(c for c in res.criteria if c.criterion_id == "G2-6")
        assert c6.status == GateStatus.FAIL
        assert "lose-001" in c6.evidence


def test_p84_g2_6_find_dtl_invariant_question_id_none_scenario_uses_canonical_mapping() -> None:
    """Verifies that scenario_def=None safely falls back to canonical question mapping."""
    for sid in ("lose-001", "lose-002", "lose-003", "lose-004", "lose-005"):
        qid = find_dtl_invariant_question_id(sid, None)
        assert qid == "q1"


def test_p84_g2_6_find_dtl_invariant_question_id_missing_questions_uses_canonical_mapping() -> None:
    """Verifies that scenario_def without questions safely falls back to canonical mapping."""
    sc_def = {"id": "lose-002", "category": "designed_to_lose"}
    qid = find_dtl_invariant_question_id("lose-002", sc_def)
    assert qid == "q1"


# -----------------------------------------------------------------------------
# Context-Aware Active-Memory & Dynamic Expected Current Tests (P8.4)
# -----------------------------------------------------------------------------


def test_p84_g2_6_lose_004_context_scoped_production_staging_passes() -> None:
    """Verifies that Production scenario passes when Production has 3.12 and Staging has 3.9."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "current",
                "plan": {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "context": {"env": "production"},
                },
                "expect": {"value": "3.12", "forbidden": ["3.9"]},
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "context": {"env": "production"},
                    "context_key": '{"env":"production"}',
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                },
                {
                    "id": "2",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.9",
                    "status": "active",
                    "is_active": True,
                    "context": {"env": "staging"},
                    "context_key": '{"env":"staging"}',
                    "valid_from": "2026-01-12T11:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-12T11:00:00Z",
                    "closed_at": None,
                },
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {"env": "production"},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {"env": "production"},
                        "context_key": '{"env":"production"}',
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "slot_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {"env": "production"},
                        "context_key": '{"env":"production"}',
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    },
                    {
                        "id": "2",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.9",
                        "status": "active",
                        "is_active": True,
                        "context": {"env": "staging"},
                        "context_key": '{"env":"staging"}',
                        "valid_from": "2026-01-12T11:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-12T11:00:00Z",
                        "closed_at": None,
                    },
                ],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is True
    assert status == "Honest Defense (Pass)"
    assert "R-REJECT enforced" in outcome_text


def test_p84_g2_6_lose_004_context_scoped_both_in_production_fails() -> None:
    """Verifies that when Production has both 3.12 and 3.9 active, lose-004 fails."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "current",
                "plan": {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "context": {"env": "production"},
                },
                "expect": {"value": "3.12", "forbidden": ["3.9"]},
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "context": {"env": "production"},
                    "context_key": '{"env":"production"}',
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                },
                {
                    "id": "2",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.9",
                    "status": "active",
                    "is_active": True,
                    "context": {"env": "production"},
                    "context_key": '{"env":"production"}',
                    "valid_from": "2026-01-12T11:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-12T11:00:00Z",
                    "closed_at": None,
                },
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {"env": "production"},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {"env": "production"},
                        "context_key": '{"env":"production"}',
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    },
                    {
                        "id": "2",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.9",
                        "status": "active",
                        "is_active": True,
                        "context": {"env": "production"},
                        "context_key": '{"env":"production"}',
                        "valid_from": "2026-01-12T11:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-12T11:00:00Z",
                        "closed_at": None,
                    },
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Rejected value '3.9' is present in active memory" in outcome_text


def test_p84_g2_6_lose_004_context_scoped_production_has_39_active_fails() -> None:
    """Verifies that when Production has 3.9 active, lose-004 fails."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "current",
                "plan": {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "context": {"env": "production"},
                },
                "expect": {"value": "3.12", "forbidden": ["3.9"]},
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.9",
                    "status": "active",
                    "is_active": True,
                    "context": {"env": "production"},
                    "context_key": '{"env":"production"}',
                    "valid_from": "2026-01-12T11:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-12T11:00:00Z",
                    "closed_at": None,
                },
                {
                    "id": "2",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "context": {"env": "staging"},
                    "context_key": '{"env":"staging"}',
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                },
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {"env": "production"},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.9",
                        "status": "active",
                        "is_active": True,
                        "context": {"env": "production"},
                        "context_key": '{"env":"production"}',
                        "valid_from": "2026-01-12T11:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-12T11:00:00Z",
                        "closed_at": None,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert (
        "Rejected value '3.9' is present in active memory" in outcome_text
        or "Expected current value" in outcome_text
    )


def test_p84_g2_6_lose_004_context_scoped_missing_context_evidence_fails() -> None:
    """Verifies missing required context evidence in context-specific scenario fails."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "current",
                "plan": {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "context": {"env": "production"},
                },
                "expect": {"value": "3.12", "forbidden": ["3.9"]},
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                    # "context" field omitted!
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": None,  # Context is None!
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Required context evidence missing" in outcome_text


def test_p84_g2_7_equivalent_records_different_insertion_orders_identical_signature() -> None:
    """Verifies equivalent records in different insertion orders produce identical signatures."""
    metrics = _canonical_metrics()
    rec1 = {
        "id": "1",
        "entity": "Server A",
        "attribute": "python_version",
        "value": "3.12",
        "status": "active",
        "is_active": True,
        "context": {"env": "production"},
        "context_key": '{"env":"production"}',
        "valid_from": "2026-01-10T09:00:00Z",
        "valid_to": None,
        "recorded_at": "2026-01-10T09:00:00Z",
        "closed_at": None,
    }
    rec2 = {
        "id": "2",
        "entity": "Server A",
        "attribute": "python_version",
        "value": "3.9",
        "status": "rejected",
        "is_active": False,
        "context": {"env": "staging"},
        "context_key": '{"env":"staging"}',
        "valid_from": "2026-01-12T11:00:00Z",
        "valid_to": None,
        "recorded_at": "2026-01-12T11:00:00Z",
        "closed_at": None,
    }

    t_order1 = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "active_memory_records": [rec1],
            "memory_records": [rec1, rec2],  # order 1
        },
    )
    t_order2 = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "active_memory_records": [rec1],
            "memory_records": [rec2, rec1],  # order 2 (reversed)
        },
    )

    norm1 = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=["lose-004"],
        traces_by_system={"MM": [t_order1]},
    )
    norm2 = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=["lose-004"],
        traces_by_system={"MM": [t_order2]},
    )

    assert norm1.question_outcomes == norm2.question_outcomes


def test_p84_g2_6_lose_004_scenario_derived_current_value_313_passes() -> None:
    """Verifies that scenario with different expected current value (3.13) passes."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "current",
                "expect": {"value": "3.13", "forbidden": ["3.9"]},
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.13"},
        actual={
            "value": "3.13",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.13",
                    "status": "active",
                    "is_active": True,
                    "context": {},
                    "context_key": "",
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                }
            ],
            "memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.13",
                    "status": "active",
                    "is_active": True,
                    "context": {},
                    "context_key": "",
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.13",
                        "status": "active",
                        "is_active": True,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "slot_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.13",
                        "status": "active",
                        "is_active": True,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is True
    assert status == "Honest Defense (Pass)"
    assert "R-REJECT enforced on claim '3.9'" in outcome_text
    assert "value=3.13" in outcome_text


def test_p84_g2_6_lose_004_cannot_derive_expected_current_value_fails() -> None:
    """Verifies that scenario without derivable expected current value fails."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "current",
                "expect": {"forbidden": ["3.9"]},  # Missing 'value' key!
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"forbidden": ["3.9"]},  # Missing 'value' key!
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Could not derive expected current value from scenario data" in outcome_text


def test_p84_g2_6_find_dtl_invariant_question_id_malformed_questions_returns_none() -> None:
    """Verifies malformed questions field (None, str, dict) returns None and does not fallback."""
    for malformed_val in (None, "q1", {"id": "q1"}, 12345):
        sc_def = {
            "id": "lose-001",
            "category": "designed_to_lose",
            "questions": malformed_val,
        }
        qid = find_dtl_invariant_question_id("lose-001", sc_def)
        assert qid is None


# -----------------------------------------------------------------------------
# Temporal Validity Active-Memory Verification Regression Tests (P8.4)
# -----------------------------------------------------------------------------


def test_p84_g2_6_lose_004_active_memory_outside_valid_time_fails() -> None:
    """Verifies a lifecycle-active memory outside requested valid-time window fails."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "current",
                "plan": {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "intent": "current",
                },
                "expect": {"value": "3.12", "forbidden": ["3.9"]},
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "valid_from": "2026-07-01T00:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [],  # Outside valid time, so repository retrieved nothing
                "slot_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "valid_from": "2026-07-01T00:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Expected current value '3.12' is not active in memory" in outcome_text


def test_p84_g2_6_lose_004_memory_recorded_after_known_at_fails() -> None:
    """Verifies a memory recorded after the requested known_at fails verification."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "current",
                "plan": {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "intent": "current",
                },
                "expect": {"value": "3.12", "forbidden": ["3.9"]},
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-07-01T00:00:00Z",
                    "closed_at": None,
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                # Recorded after known_at, so repository retrieved nothing
                "retrieved_memories": [],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Expected current value '3.12' is not active in memory" in outcome_text


def test_p84_g2_6_lose_004_historical_query_identifies_past_valid_value_passes() -> None:
    """Verifies a historical query correctly identifies the value valid at its requested time."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "as_of",
                "plan": {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "intent": "as_of",
                    "valid_at": "2025-06-01T00:00:00Z",
                    "known_at": "2026-06-25T09:00:00Z",
                },
                "expect": {"value": "3.11", "forbidden": ["3.9"]},
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="as_of",
        expected={"value": "3.11"},
        actual={
            "value": "3.11",
            "valid_at": "2025-06-01T00:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.11",
                    "status": "historical",
                    "is_active": False,
                    "valid_from": "2025-01-01T00:00:00Z",
                    "valid_to": "2026-01-10T09:00:00Z",
                    "recorded_at": "2025-01-01T00:00:00Z",
                    "closed_at": None,
                },
                {
                    "id": "2",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                },
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2025-06-01T00:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.11",
                        "status": "active",
                        "is_active": True,
                        "valid_from": "2025-01-01T00:00:00Z",
                        "valid_to": "2026-01-10T09:00:00Z",
                        "recorded_at": "2025-01-01T00:00:00Z",
                        "closed_at": None,
                    }
                ],
                "slot_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.11",
                        "status": "active",
                        "is_active": True,
                        "valid_from": "2025-01-01T00:00:00Z",
                        "valid_to": "2026-01-10T09:00:00Z",
                        "recorded_at": "2025-01-01T00:00:00Z",
                        "closed_at": None,
                    },
                    {
                        "id": "2",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    },
                ],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is True
    assert status == "Honest Defense (Pass)"
    assert "R-REJECT enforced on claim '3.9'" in outcome_text
    assert "value=3.11" in outcome_text


def test_p84_g2_6_lose_004_current_time_behavior_remains_correct() -> None:
    """Verifies current-time query at scenario now correctly identifies active value 3.12."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "current",
                "plan": {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "intent": "current",
                },
                "expect": {"value": "3.12", "forbidden": ["3.9"]},
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                }
            ],
            "memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                    "valid_from": "2026-01-10T09:00:00Z",
                    "valid_to": None,
                    "recorded_at": "2026-01-10T09:00:00Z",
                    "closed_at": None,
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "slot_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                        "context": {},
                        "context_key": "",
                        "valid_from": "2026-01-10T09:00:00Z",
                        "valid_to": None,
                        "recorded_at": "2026-01-10T09:00:00Z",
                        "closed_at": None,
                    }
                ],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is True
    assert status == "Honest Defense (Pass)"
    assert "R-REJECT enforced on claim '3.9'" in outcome_text
    assert "value=3.12" in outcome_text


def test_p84_g2_6_lose_004_missing_temporal_evidence_fails() -> None:
    """Verifies that missing temporal metadata (valid_at/known_at) fails verification."""
    sc_def = {
        "id": "lose-004",
        "category": "designed_to_lose",
        "now": "2026-06-25T09:00:00Z",
        "questions": [
            {
                "id": "q1",
                "type": "current",
                "plan": {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "intent": "current",
                },
                "expect": {"value": "3.12", "forbidden": ["3.9"]},
            }
        ],
    }
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "active_memory_records": [
                {
                    "id": "1",
                    "entity": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "status": "active",
                    "is_active": True,
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                # valid_at and known_at are omitted!
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Retrieval metadata missing" in outcome_text


def test_p84_g2_6_lose_004_slot_contains_historical_rejected_retrieval_excludes_passes() -> None:
    """Verifies that gate evaluates actual retrieval when slot records contain historical/rejected.

    When candidate slot records include historical or rejected claims but actual repository
    retrieval excludes them and returns the active current memory, the gate passes.
    """
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")

    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    },
                    {
                        "id": "2",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.9",
                        "status": "rejected",
                        "is_active": False,
                    },
                    {
                        "id": "3",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.8",
                        "status": "historical",
                        "is_active": False,
                    },
                ],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is True
    assert status == "Honest Defense (Pass)"
    assert "R-REJECT enforced on claim '3.9'" in outcome_text


def test_p84_g2_6_lose_004_temporally_invalid_memory_in_retrieved_fails() -> None:
    """Verifies that gate fails if non-active memory is returned in retrieved_memories."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")

    # Trace where retrieved_memories incorrectly includes a historical record
    trace_with_historical = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "historical",
                        "is_active": False,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome(
        "lose-004", sc_def_004, trace_with_historical
    )
    assert is_ok is False
    assert status == "Observed Failure"
    assert "non-active status 'historical'" in outcome_text


def test_p84_g2_6_lose_004_retrieval_metadata_mismatch_fails() -> None:
    """Verifies that gate fails when retrieval metadata does not match the evaluated query."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")

    # Trace where retrieval valid_at does not match evaluated query valid_at
    trace_mismatch = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "now": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2024-01-01T00:00:00Z",  # Mismatched valid_at!
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome(
        "lose-004", sc_def_004, trace_mismatch
    )
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Retrieval metadata mismatch: valid_at" in outcome_text


# -----------------------------------------------------------------------------
# Actual Repository Retrieval Evidence Regression Tests (P8.4)
# -----------------------------------------------------------------------------


def test_p84_g2_6_actual_repo_later_rejection() -> None:
    """Case A: Later rejection.
    A claim active at known_at_1 is rejected at known_at_2.
    A query at known_at_1 sees it active; a query at known_at_2 sees it rejected.
    """
    conn = init_db(":memory:")
    cfg = load_app_config()
    svc = MemoryService(db_target=conn, cfg=cfg)

    t1 = datetime(2026, 1, 10, 9, 0, tzinfo=UTC)
    t2 = datetime(2026, 2, 10, 9, 0, tzinfo=UTC)

    # Ingest active memory at t1
    msg1 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A uses Python 3.12",
        event_time=t1,
    )
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t1,
        extract_confidence=0.95,
    )
    svc.ingest_claims(msg1, [claim1], recorded_at=t1)
    ent_id = int(svc.repo.get_entity_by_name("server a")["id"])

    # Query at known_at = t1 -> active
    mems_t1 = retrieve_valid_memories(
        svc.repo,
        entity_id=ent_id,
        attribute="python_version",
        valid_at=t1,
        known_at=t1,
    )
    assert len(mems_t1) == 1
    assert mems_t1[0].value_display == "3.12"
    assert mems_t1[0].status == MemoryStatus.ACTIVE

    # Retract/reject memory at t2
    mem_id = mems_t1[0].id
    svc.repo.update_memory(mem_id, {"status": MemoryStatus.REJECTED, "closed_at": t2})

    # Query at known_at = t2 -> memory is no longer active
    mems_t2 = retrieve_valid_memories(
        svc.repo,
        entity_id=ent_id,
        attribute="python_version",
        valid_at=t1,
        known_at=t2,
    )
    assert len(mems_t2) == 0

    # Query at known_at = t1 still reconstructs it as active (belief time)
    mems_t1_asof = retrieve_valid_memories(
        svc.repo,
        entity_id=ent_id,
        attribute="python_version",
        valid_at=t1,
        known_at=t1,
    )
    assert len(mems_t1_asof) == 1
    assert mems_t1_asof[0].value_display == "3.12"
    assert mems_t1_asof[0].status == MemoryStatus.ACTIVE

    conn.close()


def test_p84_g2_6_actual_repo_later_replacement() -> None:
    """Case B: Later replacement.
    A claim is replaced at a later valid-time. Queries before the replacement
    valid-time see the old value; queries after see the new value.
    """
    conn = init_db(":memory:")
    cfg = load_app_config()
    svc = MemoryService(db_target=conn, cfg=cfg)

    t_ingest = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    v1_time = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    v2_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    # Ingest 3.11 valid from 2025-01-01
    msg1 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs Python 3.11",
        event_time=t_ingest,
    )
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.11",
        valid_from=v1_time,
        extract_confidence=0.95,
    )
    svc.ingest_claims(msg1, [claim1], recorded_at=t_ingest)

    # Ingest replacement 3.12 valid from 2026-01-01
    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs Python 3.12",
        event_time=t_ingest,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=v2_time,
        extract_confidence=0.95,
    )
    svc.ingest_claims(msg2, [claim2], recorded_at=t_ingest)
    ent_id = int(svc.repo.get_entity_by_name("server a")["id"])

    # Query before replacement valid-time (2025-06-01) -> returns 3.11
    mems_past = retrieve_valid_memories(
        svc.repo,
        entity_id=ent_id,
        attribute="python_version",
        valid_at=datetime(2025, 6, 1, 0, 0, tzinfo=UTC),
        known_at=t_ingest,
    )
    assert len(mems_past) == 1
    assert mems_past[0].value_display == "3.11"

    # Query after replacement valid-time (2026-06-01) -> returns 3.12
    mems_curr = retrieve_valid_memories(
        svc.repo,
        entity_id=ent_id,
        attribute="python_version",
        valid_at=datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
        known_at=t_ingest,
    )
    assert len(mems_curr) == 1
    assert mems_curr[0].value_display == "3.12"

    conn.close()


def test_p84_g2_6_actual_repo_backfill_knowledge_time() -> None:
    """Case C: Backfill.
    A memory is inserted later with an earlier valid-time. Queries evaluated before
    the insertion knowledge-time do not see it; queries after see it.
    """
    conn = init_db(":memory:")
    cfg = load_app_config()
    svc = MemoryService(db_target=conn, cfg=cfg)

    t_knowledge_1 = datetime(2026, 1, 15, 0, 0, tzinfo=UTC)
    t_knowledge_2 = datetime(2026, 2, 15, 0, 0, tzinfo=UTC)
    v_past = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)

    # Ingest backfilled claim recorded at t_knowledge_2
    msg = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A was configured with Python 3.10 back in 2025",
        event_time=t_knowledge_2,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        valid_from=v_past,
        extract_confidence=0.95,
    )
    svc.ingest_claims(msg, [claim], recorded_at=t_knowledge_2)
    ent_id = int(svc.repo.get_entity_by_name("server a")["id"])

    # Query at known_at = t_knowledge_1 (before backfill recorded) -> empty
    mems_before = retrieve_valid_memories(
        svc.repo,
        entity_id=ent_id,
        attribute="python_version",
        valid_at=v_past,
        known_at=t_knowledge_1,
    )
    assert len(mems_before) == 0

    # Query at known_at = t_knowledge_2 (after backfill recorded) -> returns 3.10
    mems_after = retrieve_valid_memories(
        svc.repo,
        entity_id=ent_id,
        attribute="python_version",
        valid_at=v_past,
        known_at=t_knowledge_2,
    )
    assert len(mems_after) == 1
    assert mems_after[0].value_display == "3.10"

    conn.close()


def test_p84_g2_6_actual_repo_later_dispute() -> None:
    """Case D: Dispute.
    A dispute is recorded at a later knowledge-time. Queries before the dispute see
    the undisputed memory; queries after see the dispute, and active-memory verification proves
    the disputed claim is not active.
    """
    conn = init_db(":memory:")
    cfg = load_app_config()
    svc = MemoryService(db_target=conn, cfg=cfg)

    t1 = datetime(2026, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 20, 0, 0, tzinfo=UTC)

    # Ingest undisputed claim from Priya at t1
    msg1 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A uses Python 3.12",
        event_time=t1,
    )
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t1,
        extract_confidence=0.95,
    )
    svc.ingest_claims(msg1, [claim1], recorded_at=t1)
    ent_id = int(svc.repo.get_entity_by_name("server a")["id"])

    # Query at t1 -> undisputed active memory
    mems_t1 = retrieve_valid_memories(
        svc.repo,
        entity_id=ent_id,
        attribute="python_version",
        valid_at=t1,
        known_at=t1,
    )
    assert len(mems_t1) == 1
    assert mems_t1[0].value_display == "3.12"
    assert mems_t1[0].status == MemoryStatus.ACTIVE

    # Seed equal reliability source and ingest conflicting claim at t2 to create dispute
    svc.repo.insert_source(
        Source(
            id="sysadmin_marcus",
            name="Marcus",
            type="admin",
            reliability=0.95,  # Same reliability as priya
        )
    )
    msg2 = IngestMessage(
        source_id="sysadmin_marcus",
        text="Server A uses Python 3.13",
        event_time=t2,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.13",
        valid_from=t1,
        extract_confidence=0.95,
    )
    svc.ingest_claims(msg2, [claim2], recorded_at=t2)

    # Query at t2 -> conflicting claims are DISPUTED, none are active
    mems_t2 = retrieve_valid_memories(
        svc.repo,
        entity_id=ent_id,
        attribute="python_version",
        valid_at=t1,
        known_at=t2,
    )
    assert len(mems_t2) == 2
    assert all(m.status == MemoryStatus.DISPUTED for m in mems_t2)
    active_mems_t2 = [m for m in mems_t2 if m.status == MemoryStatus.ACTIVE]
    assert len(active_mems_t2) == 0

    # Verify that gate rejects trace if disputed memory is returned as active
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace_with_dispute = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": t1.isoformat(),
            "known_at": t2.isoformat(),
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": t1.isoformat(),
                "known_at": t2.isoformat(),
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "disputed",
                        "is_active": False,
                        "context": {},
                        "valid_from": t1.isoformat(),
                        "recorded_at": t1.isoformat(),
                    }
                ],
                "slot_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "disputed",
                        "is_active": False,
                        "context": {},
                        "valid_from": t1.isoformat(),
                        "recorded_at": t1.isoformat(),
                    }
                ],
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome(
        "lose-004", sc_def_004, trace_with_dispute
    )
    assert is_ok is False
    assert status == "Observed Failure"
    assert "disputed" in outcome_text.lower()

    conn.close()


def test_p84_g2_6_actual_repo_previously_closed() -> None:
    """Case E: Previously closed memory.
    A memory whose valid-time interval has ended is historical, not active, even if no
    replacement exists.
    """
    conn = init_db(":memory:")
    cfg = load_app_config()
    svc = MemoryService(db_target=conn, cfg=cfg)

    t_ingest = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    v_start = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    v_end = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)

    # Ingest memory with explicit validity window ending at v_end
    msg = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A legacy runtime",
        event_time=t_ingest,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.8",
        valid_from=v_start,
        valid_to=v_end,
        extract_confidence=0.95,
    )
    svc.ingest_claims(msg, [claim], recorded_at=t_ingest)
    ent_id = int(svc.repo.get_entity_by_name("server a")["id"])
    mems = svc.repo.find_memories(entity_id=ent_id, attribute="python_version")
    mem_id = mems[0].id

    # Close memory window with status HISTORICAL
    svc.repo.close_memory(
        memory_id=mem_id,
        valid_to=v_end,
        closed_at=t_ingest,
        status=MemoryStatus.HISTORICAL,
    )

    # Query at valid_at = 2026-06-01 (after validity window expired)
    mems_expired = retrieve_valid_memories(
        svc.repo,
        entity_id=ent_id,
        attribute="python_version",
        valid_at=datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
        known_at=t_ingest,
    )
    assert len(mems_expired) == 0

    # Slot memories shows the record has status historical and valid_to in the past
    slot_mems = svc.repo.find_memories(
        entity_id=ent_id,
        attribute="python_version",
        valid_at=None,
        known_at=t_ingest,
        include_rejected=True,
    )
    assert len(slot_mems) == 1
    assert slot_mems[0].status == MemoryStatus.HISTORICAL
    assert slot_mems[0].valid_to is not None
    assert slot_mems[0].valid_to <= datetime(2026, 6, 1, 0, 0, tzinfo=UTC)

    conn.close()


def test_p84_g2_6_actual_repo_missing_retrieval_evidence_fails() -> None:
    """Case F: Missing retrieval evidence.
    If a trace lacks repository retrieval evidence, the gate must reject it with Observed Failure.
    """
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")

    # Trace with decision records and answer, but lacking repository_retrieval
    trace_no_ret = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            # repository_retrieval is omitted!
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome(
        "lose-004", sc_def_004, trace_no_ret
    )
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Active-memory evidence missing" in outcome_text
    assert "repository retrieval evidence" in outcome_text

    # Trace where repository_retrieval was not performed
    trace_unperformed = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": False,
                "error": "Retrieval execution aborted",
            },
        },
    )
    is_ok2, outcome_text2, status2 = verify_designed_to_lose_outcome(
        "lose-004", sc_def_004, trace_unperformed
    )
    assert is_ok2 is False
    assert status2 == "Observed Failure"
    assert "repository retrieval error" in outcome_text2


# -----------------------------------------------------------------------------
# P8.4 Strict Query Metadata & Agreed G2 Criterion Mapping Tests
# -----------------------------------------------------------------------------


def test_p84_lose_004_missing_answer_valid_at_fails() -> None:
    """1. Missing answer valid_at fails."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            # valid_at timestamp omitted!
            "known_at": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Query metadata missing: answer lacks valid_at timestamp" in outcome_text


def test_p84_lose_004_missing_answer_known_at_fails() -> None:
    """2. Missing answer known_at fails."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            # known_at timestamp omitted!
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Query metadata missing: answer lacks known_at timestamp" in outcome_text


def test_p84_lose_004_invalid_answer_timestamp_fails() -> None:
    """3. Invalid answer timestamp fails."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "invalid-timestamp",
            "known_at": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Query metadata invalid: unable to parse answer valid_at" in outcome_text


def test_p84_lose_004_retrieval_timestamp_mismatch_fails() -> None:
    """4. Retrieval timestamp mismatch fails."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    # Case a: valid_at mismatch
    trace_v = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2025-01-01T00:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok_v, text_v, status_v = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace_v)
    assert is_ok_v is False
    assert status_v == "Observed Failure"
    assert "Retrieval metadata mismatch: valid_at" in text_v

    # Case b: known_at mismatch
    trace_k = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2025-01-01T00:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok_k, text_k, status_k = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace_k)
    assert is_ok_k is False
    assert status_k == "Observed Failure"
    assert "Retrieval metadata mismatch: known_at" in text_k


def test_p84_lose_004_string_retrieval_context_fails() -> None:
    """5. String retrieval context fails."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": "env=production",
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Retrieval context invalid: expected dictionary, got str" in outcome_text


def test_p84_lose_004_list_retrieval_context_fails() -> None:
    """6. List retrieval context fails."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": ["prod", "us-east"],
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Retrieval context invalid: expected dictionary, got list" in outcome_text


def test_p84_lose_004_null_retrieval_context_fails() -> None:
    """7. Null retrieval context fails."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": None,
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is False
    assert status == "Observed Failure"
    assert "Required context evidence missing" in outcome_text


def test_p84_lose_004_valid_unscoped_context_passes() -> None:
    """8. A valid unscoped {} context still passes."""
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = next(s for s in sc_defs if s["id"] == "lose-004")
    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is True
    assert status == "Honest Defense (Pass)"
    assert "R-REJECT enforced on claim '3.9'" in outcome_text


def test_p84_lose_004_valid_explicitly_scoped_context_passes() -> None:
    """9. A valid explicitly scoped context still passes."""
    import copy

    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_def_004 = copy.deepcopy(next(s for s in sc_defs if s["id"] == "lose-004"))
    sc_def_004.setdefault("questions", [{}])[0].setdefault("plan", {})["context"] = {"env": "prod"}

    trace = QuestionEvalResult(
        scenario_id="lose-004",
        question_id="q1",
        question_type="current",
        expected={"value": "3.12"},
        actual={
            "value": "3.12",
            "valid_at": "2026-06-25T09:00:00Z",
            "known_at": "2026-06-25T09:00:00Z",
            "decision_records": [
                {
                    "rule_id": "R-REJECT",
                    "action": "REJECT",
                    "claim_value": "3.9",
                    "claim_subject": "Server A",
                    "claim_attribute": "python_version",
                }
            ],
            "repository_retrieval": {
                "performed": True,
                "entity": "Server A",
                "attribute": "python_version",
                "context": {"env": "prod"},
                "valid_at": "2026-06-25T09:00:00Z",
                "known_at": "2026-06-25T09:00:00Z",
                "retrieved_memories": [
                    {
                        "id": "1",
                        "entity": "Server A",
                        "attribute": "python_version",
                        "value": "3.12",
                        "status": "active",
                        "is_active": True,
                    }
                ],
                "slot_memories": [],
                "error": None,
            },
        },
    )
    is_ok, outcome_text, status = verify_designed_to_lose_outcome("lose-004", sc_def_004, trace)
    assert is_ok is True
    assert status == "Honest Defense (Pass)"
    assert "R-REJECT enforced on claim '3.9'" in outcome_text


def test_p84_all_seven_g2_criteria_agreed_ids_and_titles() -> None:
    """10. Each of the seven G2 criteria has the correct agreed ID and title."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
    )
    expected_criteria = [
        ("G2-1", "Artifact Generation"),
        ("G2-2", "Multi-Mode Comparative Evaluation"),
        ("G2-3", "Scenario Catalog Coverage"),
        ("G2-4", "Write-Time Resolution Superiority"),
        ("G2-5", "Decision Auditability and Provenance"),
        ("G2-6", "Designed-To-Lose Honesty & Failure Accounting"),
        ("G2-7", "Deterministic Reproducibility"),
    ]
    actual_criteria = [(c.criterion_id, c.title) for c in res.criteria]
    assert actual_criteria == expected_criteria


def test_p84_g2_2_fails_when_any_required_mode_missing_or_incomplete() -> None:
    """11. G2-2 fails when any one of the four required modes is missing or incomplete."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    all_completed = {
        "structured": ModeExecutionStatus.COMPLETED,
        "engine-only": ModeExecutionStatus.COMPLETED,
        "e2e": ModeExecutionStatus.COMPLETED,
        "baseline": ModeExecutionStatus.COMPLETED,
    }

    for mode in ("structured", "engine-only", "e2e", "baseline"):
        missing_statuses = {k: v for k, v in all_completed.items() if k != mode}
        res_missing = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            mode_execution_statuses=missing_statuses,
        )
        c2_miss = next(c for c in res_missing.criteria if c.criterion_id == "G2-2")
        assert c2_miss.status != GateStatus.PASS
        assert mode in c2_miss.evidence

        failed_statuses = dict(all_completed)
        failed_statuses[mode] = ModeExecutionStatus.FAILED
        res_failed = evaluate_g2_gate(
            metrics_by_system=metrics,
            executed_scenarios=sc_defs,
            traces_by_system=traces,
            mode_execution_statuses=failed_statuses,
        )
        c2_fail = next(c for c in res_failed.criteria if c.criterion_id == "G2-2")
        assert c2_fail.status == GateStatus.FAIL
        assert f"{mode} FAILED" in c2_fail.evidence


def test_p84_g2_4_fails_when_metrics_missing_or_below_thresholds() -> None:
    """12. G2-4 fails when its required metrics are missing or do not meet agreed thresholds."""
    sc_defs, _ = _canonical_scenarios_and_categories()

    # 1. Missing MM metric -> does not pass
    res_no_mm = evaluate_g2_gate(
        metrics_by_system={"B0 (naive RAG)": _canonical_metrics()["B0 (naive RAG)"]},
        executed_scenarios=sc_defs,
    )
    c4_no_mm = next(c for c in res_no_mm.criteria if c.criterion_id == "G2-4")
    assert c4_no_mm.status != GateStatus.PASS

    # 2. Missing B0 metric -> does not pass
    res_no_b0 = evaluate_g2_gate(
        metrics_by_system={"MM (structured)": _canonical_metrics()["MM (structured)"]},
        executed_scenarios=sc_defs,
    )
    c4_no_b0 = next(c for c in res_no_b0.criteria if c.criterion_id == "G2-4")
    assert c4_no_b0.status != GateStatus.PASS

    # 3. Missing individual metric value (current_accuracy is None) -> does not pass
    m_none = _canonical_metrics()
    m_none["MM (structured)"].current_accuracy = None
    res_none = evaluate_g2_gate(
        metrics_by_system=m_none,
        executed_scenarios=sc_defs,
    )
    c4_none = next(c for c in res_none.criteria if c.criterion_id == "G2-4")
    assert c4_none.status != GateStatus.PASS

    # 4. Inferior accuracy (MM current_accuracy <= B0 current_accuracy) -> FAIL
    m_bad_acc = _canonical_metrics()
    m_bad_acc["MM (structured)"].current_accuracy = 0.50
    m_bad_acc["B0 (naive RAG)"].current_accuracy = 0.60
    res_bad_acc = evaluate_g2_gate(
        metrics_by_system=m_bad_acc,
        executed_scenarios=sc_defs,
    )
    c4_bad_acc = next(c for c in res_bad_acc.criteria if c.criterion_id == "G2-4")
    assert c4_bad_acc.status == GateStatus.FAIL
    assert "MM failed to outperform B0" in c4_bad_acc.evidence

    # 5. Inferior staleness (MM stale_answer_rate >= B0 stale_answer_rate) -> FAIL
    m_bad_stale = _canonical_metrics()
    m_bad_stale["MM (structured)"].stale_answer_rate = 0.40
    m_bad_stale["B0 (naive RAG)"].stale_answer_rate = 0.20
    res_bad_stale = evaluate_g2_gate(
        metrics_by_system=m_bad_stale,
        executed_scenarios=sc_defs,
    )
    c4_bad_stale = next(c for c in res_bad_stale.criteria if c.criterion_id == "G2-4")
    assert c4_bad_stale.status == GateStatus.FAIL
    assert "MM failed to outperform B0" in c4_bad_stale.evidence


def test_p84_markdown_and_json_use_same_criterion_mapping_and_statuses(tmp_path: Path) -> None:
    """13. Generated Markdown and JSON use the same criterion mapping and statuses."""
    import json

    from eval.report_generator import generate_evaluation_report

    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    report_file = tmp_path / "report.md"
    results_file = tmp_path / "results.json"
    report_file.write_text("dummy", encoding="utf-8")
    results_file.write_text("{}", encoding="utf-8")

    all_completed = {
        "structured": ModeExecutionStatus.COMPLETED,
        "engine-only": ModeExecutionStatus.COMPLETED,
        "e2e": ModeExecutionStatus.COMPLETED,
        "baseline": ModeExecutionStatus.COMPLETED,
    }

    gate_res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        report_path=report_file,
        results_path=results_file,
        traces_by_system=traces,
        mode_execution_statuses=all_completed,
    )

    md_content = generate_evaluation_report(
        metrics_by_system=metrics,
        traces_by_system=traces,
        scenarios=sc_defs,
        gate_result=gate_res,
        timestamp="20261115_000000",
        mode_execution_statuses=all_completed,
    )

    json_payload = {
        "timestamp": "20261115_000000",
        "mode": "all",
        "g2_gate": gate_res.model_dump(),
    }
    json_str = json.dumps(json_payload)
    parsed_json = json.loads(json_str)

    json_criteria = parsed_json["g2_gate"]["criteria"]
    assert len(json_criteria) == len(gate_res.criteria) == 7

    for idx, c in enumerate(gate_res.criteria):
        jc = json_criteria[idx]
        assert jc["criterion_id"] == c.criterion_id
        assert jc["title"] == c.title
        assert jc["status"] == c.status.value
        assert jc["evidence"] == c.evidence

        assert f"`{c.criterion_id}`" in md_content
        assert c.title in md_content
        assert f"`{c.status.value}`" in md_content


# =========================================================================
# Additional Hardened Regression Tests (P8.4 Gate Hardening)
# =========================================================================


def test_p84_g2_2_empty_status_dict_returns_inconclusive() -> None:
    """G2-2: Explicit empty status dict must treat all canonical modes as NOT_EXECUTED."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        mode_execution_statuses={},
    )
    c2 = next(c for c in res.criteria if c.criterion_id == "G2-2")
    assert c2.status == GateStatus.INCONCLUSIVE
    assert "Required evaluation mode(s) not executed" in c2.evidence


def test_p84_g2_2_missing_canonical_mode_status_returns_inconclusive() -> None:
    """G2-2: Missing canonical mode status (only 3 provided) must be NOT_EXECUTED and never PASS."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    statuses = {
        "structured": ModeExecutionStatus.COMPLETED,
        "engine-only": ModeExecutionStatus.COMPLETED,
        "e2e": ModeExecutionStatus.COMPLETED,
        # baseline is missing
    }
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
        mode_execution_statuses=statuses,
    )
    c2 = next(c for c in res.criteria if c.criterion_id == "G2-2")
    assert c2.status == GateStatus.INCONCLUSIVE
    assert "baseline" in c2.evidence


def test_p84_g2_2_mode_completed_but_missing_metrics_fails() -> None:
    """G2-2: Mode marked COMPLETED but missing from metrics fails due to inconsistency."""
    metrics = _canonical_metrics()
    del metrics["MM (structured)"]  # Remove structured metric
    sc_defs, _ = _canonical_scenarios_and_categories()
    all_completed = {
        "structured": ModeExecutionStatus.COMPLETED,
        "engine-only": ModeExecutionStatus.COMPLETED,
        "e2e": ModeExecutionStatus.COMPLETED,
        "baseline": ModeExecutionStatus.COMPLETED,
    }
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        mode_execution_statuses=all_completed,
    )
    c2 = next(c for c in res.criteria if c.criterion_id == "G2-2")
    assert c2.status == GateStatus.FAIL
    assert "missing metric evidence" in c2.evidence


def test_p84_g2_2_mode_completed_but_missing_traces_fails() -> None:
    """G2-2: Mode marked COMPLETED but missing traces fails due to inconsistency."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    traces["MM (structured)"] = []  # Empty traces for structured
    all_completed = {
        "structured": ModeExecutionStatus.COMPLETED,
        "engine-only": ModeExecutionStatus.COMPLETED,
        "e2e": ModeExecutionStatus.COMPLETED,
        "baseline": ModeExecutionStatus.COMPLETED,
    }
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
        mode_execution_statuses=all_completed,
    )
    c2 = next(c for c in res.criteria if c.criterion_id == "G2-2")
    assert c2.status == GateStatus.FAIL
    assert "missing trace evidence" in c2.evidence


def test_p84_g2_2_metrics_present_but_mode_status_failed() -> None:
    """G2-2: Metrics present but mode status is FAILED must fail and never PASS."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    statuses = {
        "structured": ModeExecutionStatus.COMPLETED,
        "engine-only": ModeExecutionStatus.COMPLETED,
        "e2e": ModeExecutionStatus.COMPLETED,
        "baseline": ModeExecutionStatus.FAILED,
    }
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
        mode_execution_statuses=statuses,
    )
    c2 = next(c for c in res.criteria if c.criterion_id == "G2-2")
    assert c2.status == GateStatus.FAIL
    assert "baseline FAILED" in c2.evidence


def test_p84_g2_3_entire_mode_missing_from_traces_fails() -> None:
    """G2-3: Trace evidence must be present for all 4 canonical modes; missing mode fails."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    del traces["B0 (naive RAG)"]  # Omit baseline mode from traces
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
    )
    c3 = next(c for c in res.criteria if c.criterion_id == "G2-3")
    assert c3.status == GateStatus.FAIL
    assert "canonical mode 'baseline' missing from execution traces" in c3.evidence


def test_p84_g2_3_one_question_missing_from_one_mode_fails() -> None:
    """G2-3: A single missing question trace in any mode fails coverage."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    # Remove one question from structured
    traces["MM (structured)"] = traces["MM (structured)"][1:]
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
    )
    c3 = next(c for c in res.criteria if c.criterion_id == "G2-3")
    assert c3.status == GateStatus.FAIL
    assert "missing question traces" in c3.evidence


def test_p84_g2_3_duplicate_question_trace_in_one_mode_fails() -> None:
    """G2-3: Duplicate question trace in any mode fails coverage."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    dup = traces["MM (e2e)"][0].model_copy()
    traces["MM (e2e)"].append(dup)
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
    )
    c3 = next(c for c in res.criteria if c.criterion_id == "G2-3")
    assert c3.status == GateStatus.FAIL
    assert "duplicate question traces detected" in c3.evidence


def test_p84_g2_3_unexpected_question_trace_fails() -> None:
    """G2-3: Unexpected question trace not in declared scenarios fails coverage."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    extra_trace = QuestionEvalResult(
        scenario_id="update-999",
        question_id="q999",
        question_type="current",
        expected={},
        actual={"value": "unexpected"},
        is_current_correct=True,
    )
    traces["MM (structured)"].append(extra_trace)
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces,
    )
    c3 = next(c for c in res.criteria if c.criterion_id == "G2-3")
    assert c3.status == GateStatus.FAIL
    assert "unexpected question traces" in c3.evidence


def test_p84_validate_evaluation_metric_all_cases() -> None:
    """Validate reusable metric validator across invalid, boundary, and edge cases."""

    # 1. Valid numbers
    ok, val, err = validate_evaluation_metric(0.5, "metric")
    assert ok is True and val == 0.5 and err is None

    # 2. Boundaries at 0.0 and 1.0
    ok, val, err = validate_evaluation_metric(0.0, "zero")
    assert ok is True and val == 0.0 and err is None
    ok, val, err = validate_evaluation_metric(1.0, "one")
    assert ok is True and val == 1.0 and err is None
    ok, val, err = validate_evaluation_metric(0, "int_zero")
    assert ok is True and val == 0.0 and err is None
    ok, val, err = validate_evaluation_metric(1, "int_one")
    assert ok is True and val == 1.0 and err is None

    # 3. None handling
    ok, val, err = validate_evaluation_metric(None, "none_strict", allow_none=False)
    assert ok is False and val is None and "missing" in str(err)
    ok, val, err = validate_evaluation_metric(None, "none_allowed", allow_none=True)
    assert ok is True and val is None and err is None

    # 4. Booleans (bool is a subclass of int in Python)
    ok, val, err = validate_evaluation_metric(True, "bool_true")
    assert ok is False and "must be numeric, got bool" in str(err)
    ok, val, err = validate_evaluation_metric(False, "bool_false")
    assert ok is False and "must be numeric, got bool" in str(err)

    # 5. Strings
    ok, val, err = validate_evaluation_metric("0.85", "str_num")
    assert ok is False and "must be numeric, got str" in str(err)

    # 6. NaN and infinities
    ok, val, err = validate_evaluation_metric(float("nan"), "nan_val")
    assert ok is False and "must be finite" in str(err)
    ok, val, err = validate_evaluation_metric(float("inf"), "pos_inf")
    assert ok is False and "must be finite" in str(err)
    ok, val, err = validate_evaluation_metric(float("-inf"), "neg_inf")
    assert ok is False and "must be finite" in str(err)

    # 7. Out of range: negative and greater than 1
    ok, val, err = validate_evaluation_metric(-0.01, "neg_val")
    assert ok is False and "must be in range [0.0, 1.0]" in str(err)
    ok, val, err = validate_evaluation_metric(1.05, "over_val")
    assert ok is False and "must be in range [0.0, 1.0]" in str(err)


def test_p84_g2_4_metric_validation_failures() -> None:
    """G2-4 fails on NaN, inf, bool, str, neg, >1 metrics, and inconclusive on None."""
    sc_defs, _ = _canonical_scenarios_and_categories()

    # NaN in MM current_accuracy -> FAIL
    m_nan = _canonical_metrics()
    m_nan["MM (structured)"].current_accuracy = float("nan")
    res = evaluate_g2_gate(metrics_by_system=m_nan, executed_scenarios=sc_defs)
    c4 = next(c for c in res.criteria if c.criterion_id == "G2-4")
    assert c4.status == GateStatus.FAIL
    assert "must be finite" in c4.evidence

    # Positive infinity in B0 current_accuracy -> FAIL
    m_inf = _canonical_metrics()
    m_inf["B0 (naive RAG)"].current_accuracy = float("inf")
    res = evaluate_g2_gate(metrics_by_system=m_inf, executed_scenarios=sc_defs)
    c4 = next(c for c in res.criteria if c.criterion_id == "G2-4")
    assert c4.status == GateStatus.FAIL
    assert "must be finite" in c4.evidence

    # Negative infinity in MM stale_answer_rate -> FAIL
    m_ninf = _canonical_metrics()
    m_ninf["MM (structured)"].stale_answer_rate = float("-inf")
    res = evaluate_g2_gate(metrics_by_system=m_ninf, executed_scenarios=sc_defs)
    c4 = next(c for c in res.criteria if c.criterion_id == "G2-4")
    assert c4.status == GateStatus.FAIL
    assert "must be finite" in c4.evidence

    # Boolean in MM current_accuracy -> FAIL
    m_bool = _canonical_metrics()
    m_bool["MM (structured)"].current_accuracy = True
    res = evaluate_g2_gate(metrics_by_system=m_bool, executed_scenarios=sc_defs)
    c4 = next(c for c in res.criteria if c.criterion_id == "G2-4")
    assert c4.status == GateStatus.FAIL
    assert "must be numeric, got bool" in c4.evidence

    # String in B0 stale_answer_rate -> FAIL
    m_str = _canonical_metrics()
    m_str["B0 (naive RAG)"].stale_answer_rate = "0.35"
    res = evaluate_g2_gate(metrics_by_system=m_str, executed_scenarios=sc_defs)
    c4 = next(c for c in res.criteria if c.criterion_id == "G2-4")
    assert c4.status == GateStatus.FAIL
    assert "must be numeric, got str" in c4.evidence

    # Negative metric -> FAIL
    m_neg = _canonical_metrics()
    m_neg["MM (structured)"].current_accuracy = -0.1
    res = evaluate_g2_gate(metrics_by_system=m_neg, executed_scenarios=sc_defs)
    c4 = next(c for c in res.criteria if c.criterion_id == "G2-4")
    assert c4.status == GateStatus.FAIL
    assert "must be in range [0.0, 1.0]" in c4.evidence

    # Metric > 1.0 -> FAIL
    m_over = _canonical_metrics()
    m_over["MM (structured)"].current_accuracy = 1.2
    res = evaluate_g2_gate(metrics_by_system=m_over, executed_scenarios=sc_defs)
    c4 = next(c for c in res.criteria if c.criterion_id == "G2-4")
    assert c4.status == GateStatus.FAIL
    assert "must be in range [0.0, 1.0]" in c4.evidence


def test_p84_g2_5_metric_validation_and_b0_none_strictness() -> None:
    """G2-5 requires MM metrics in [0, 1] and strictly requires B0 audit metrics to be None."""
    sc_defs, _ = _canonical_scenarios_and_categories()

    # 1. B0 reporting non-None values (e.g. 0.0, NaN, string) -> FAIL
    for bad_b0 in (0.0, float("nan"), "n/a", 0.5):
        m_b0_bad = _canonical_metrics()
        m_b0_bad["B0 (naive RAG)"].engine_decision_accuracy = bad_b0
        res = evaluate_g2_gate(metrics_by_system=m_b0_bad, executed_scenarios=sc_defs)
        c5 = next(c for c in res.criteria if c.criterion_id == "G2-5")
        assert c5.status == GateStatus.FAIL
        assert "must report 'n/a' (None)" in c5.evidence

    # 2. MM reporting boolean -> FAIL
    m_bool = _canonical_metrics()
    m_bool["MM (structured)"].engine_decision_accuracy = True
    res = evaluate_g2_gate(metrics_by_system=m_bool, executed_scenarios=sc_defs)
    c5 = next(c for c in res.criteria if c.criterion_id == "G2-5")
    assert c5.status == GateStatus.FAIL
    assert "must be numeric, got bool" in c5.evidence

    # 3. MM reporting NaN -> FAIL
    m_nan = _canonical_metrics()
    m_nan["MM (structured)"].provenance_correctness = float("nan")
    res = evaluate_g2_gate(metrics_by_system=m_nan, executed_scenarios=sc_defs)
    c5 = next(c for c in res.criteria if c.criterion_id == "G2-5")
    assert c5.status == GateStatus.FAIL
    assert "must be finite" in c5.evidence

    # 4. MM reporting string -> FAIL
    m_str = _canonical_metrics()
    m_str["MM (structured)"].provenance_correctness = "0.95"
    res = evaluate_g2_gate(metrics_by_system=m_str, executed_scenarios=sc_defs)
    c5 = next(c for c in res.criteria if c.criterion_id == "G2-5")
    assert c5.status == GateStatus.FAIL
    assert "must be numeric, got str" in c5.evidence

    # 5. Valid 1.0 and 0.0 boundaries
    m_pass = _canonical_metrics()
    m_pass["MM (structured)"].engine_decision_accuracy = 1.0
    m_pass["MM (structured)"].provenance_correctness = 0.80  # exact threshold
    res_pass = evaluate_g2_gate(metrics_by_system=m_pass, executed_scenarios=sc_defs)
    c5_pass = next(c for c in res_pass.criteria if c.criterion_id == "G2-5")
    assert c5_pass.status == GateStatus.PASS


def test_p84_g2_7_both_runs_omit_canonical_mode_fails() -> None:
    """G2-7 independently enforces all 4 canonical modes; fails if both runs omit the same mode."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    # Both omit e2e
    metrics_no_e2e = {k: v for k, v in metrics.items() if "e2e" not in k.lower()}
    traces_no_e2e = {k: v for k, v in traces.items() if "e2e" not in k.lower()}

    run2 = normalize_run_result(
        metrics_by_system=metrics_no_e2e,
        scenario_ids=sc_ids,
        traces_by_system=traces_no_e2e,
    )

    res = evaluate_g2_gate(
        metrics_by_system=metrics_no_e2e,
        executed_scenarios=sc_defs,
        traces_by_system=traces_no_e2e,
        reproducibility_run=run2,
    )
    c7 = next(c for c in res.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.FAIL
    assert "missing canonical evaluation mode(s)" in c7.evidence
    assert "e2e" in c7.evidence


def test_p84_g2_7_one_run_omits_question_fails() -> None:
    """G2-7 fails if one run omits a declared question."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces1 = _canonical_traces(sc_defs, list(metrics.keys()))
    traces2 = _canonical_traces(sc_defs, list(metrics.keys()))

    # Run 2 omits first question in baseline
    traces2["B0 (naive RAG)"] = traces2["B0 (naive RAG)"][1:]

    run2 = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        traces_by_system=traces2,
    )

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces1,
        reproducibility_run=run2,
    )
    c7 = next(c for c in res.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.FAIL
    assert "missing question" in c7.evidence


def test_p84_g2_7_both_runs_duplicate_identities_fails() -> None:
    """G2-7 fails if runs contain duplicate question identities."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces1 = _canonical_traces(sc_defs, list(metrics.keys()))

    # Duplicate question in Run 1
    dup = traces1["MM (engine-only)"][0].model_copy()
    traces1["MM (engine-only)"].append(dup)

    run2 = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        traces_by_system=traces1,
    )

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces1,
        reproducibility_run=run2,
    )
    c7 = next(c for c in res.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.FAIL
    assert "duplicate question identities" in c7.evidence


def test_p84_g2_7_unexpected_question_in_one_run_fails() -> None:
    """G2-7 fails if one run contains an unexpected question outcome."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces1 = _canonical_traces(sc_defs, list(metrics.keys()))
    traces2 = _canonical_traces(sc_defs, list(metrics.keys()))

    extra_trace = QuestionEvalResult(
        scenario_id="update-999",
        question_id="q999",
        question_type="current",
        expected={},
        actual={"value": "unexpected"},
        is_current_correct=True,
    )
    traces2["MM (structured)"].append(extra_trace)

    run2 = normalize_run_result(
        metrics_by_system=metrics,
        scenario_ids=sc_ids,
        traces_by_system=traces2,
    )

    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=traces1,
        reproducibility_run=run2,
    )
    c7 = next(c for c in res.criteria if c.criterion_id == "G2-7")
    assert c7.status == GateStatus.FAIL
    assert "unexpected question" in c7.evidence


def test_p84_g2_1_read_back_validation_defects(tmp_path: Path) -> None:
    """G2-1 read-back verification fails on missing JSON gate, status/title/overall mismatch."""
    expected_gate = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.PASS,
        criteria=[
            GateCriterionResult(
                criterion_id=cid,
                title=title,
                target_rule="",
                status=GateStatus.PASS,
                evidence=f"Evidence for {cid}",
            )
            for cid, title in CANONICAL_G2_CRITERIA.items()
        ],
        summary="Gate G2 PASS",
    )

    # 1. Missing final JSON gate
    rep_p, res_p = _write_valid_artifacts(tmp_path / "case1")
    res_data = json.loads(res_p.read_text(encoding="utf-8"))
    del res_data["g2_gate"]
    res_p.write_text(json.dumps(res_data), encoding="utf-8")
    ok, ev, errs = verify_persisted_artifacts(rep_p, res_p, expected_gate)
    assert ok is False
    assert any("missing final 'g2_gate' object" in e for e in errs)

    # 2. JSON criterion status differs from in-memory result
    rep_p2, res_p2 = _write_valid_artifacts(tmp_path / "case2")
    res_data2 = json.loads(res_p2.read_text(encoding="utf-8"))
    res_data2["g2_gate"]["criteria"][3]["status"] = "FAIL"  # G2-4 in JSON is FAIL
    res_p2.write_text(json.dumps(res_data2), encoding="utf-8")
    ok, ev, errs = verify_persisted_artifacts(rep_p2, res_p2, expected_gate)
    assert ok is False
    assert any("Results JSON criterion 'G2-4' status mismatch" in e for e in errs)

    # 3. Markdown status differs from in-memory result
    rep_p3, res_p3 = _write_valid_artifacts(tmp_path / "case3")
    rep_text3 = rep_p3.read_text(encoding="utf-8").replace(
        "| `G2-4` | Write-Time Resolution Superiority | Rule for G2-4 | **`PASS`**",
        "| `G2-4` | Write-Time Resolution Superiority | Rule for G2-4 | **`FAIL`**",
    )
    rep_p3.write_text(rep_text3, encoding="utf-8")
    ok, ev, errs = verify_persisted_artifacts(rep_p3, res_p3, expected_gate)
    assert ok is False
    assert any("Report Section H criterion 'G2-4' status mismatch" in e for e in errs)

    # 4. Duplicate criterion ID in JSON
    rep_p4, res_p4 = _write_valid_artifacts(tmp_path / "case4")
    res_data4 = json.loads(res_p4.read_text(encoding="utf-8"))
    res_data4["g2_gate"]["criteria"].append(
        dict(res_data4["g2_gate"]["criteria"][0])  # duplicate G2-1
    )
    res_p4.write_text(json.dumps(res_data4), encoding="utf-8")
    ok, ev, errs = verify_persisted_artifacts(rep_p4, res_p4, expected_gate)
    assert ok is False
    assert any("duplicate criterion ID" in e for e in errs)

    # 5. Missing criterion in JSON
    rep_p5, res_p5 = _write_valid_artifacts(tmp_path / "case5")
    res_data5 = json.loads(res_p5.read_text(encoding="utf-8"))
    res_data5["g2_gate"]["criteria"] = res_data5["g2_gate"]["criteria"][:6]  # omit G2-7
    res_p5.write_text(json.dumps(res_data5), encoding="utf-8")
    ok, ev, errs = verify_persisted_artifacts(rep_p5, res_p5, expected_gate)
    assert ok is False
    assert any("missing canonical criterion ID" in e for e in errs)

    # 6. Overall status mismatch
    rep_p6, res_p6 = _write_valid_artifacts(tmp_path / "case6", overall_status="FAIL")
    ok, ev, errs = verify_persisted_artifacts(rep_p6, res_p6, expected_gate)
    assert ok is False
    assert any("overall status mismatch" in e for e in errs)


def test_p84_g2_2_completed_mode_traces_none_fails() -> None:
    """G2-2: Mode marked COMPLETED with traces_by_system=None must fail for missing trace."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    all_completed = {
        "structured": ModeExecutionStatus.COMPLETED,
        "engine-only": ModeExecutionStatus.COMPLETED,
        "e2e": ModeExecutionStatus.COMPLETED,
        "baseline": ModeExecutionStatus.COMPLETED,
    }
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system=None,
        mode_execution_statuses=all_completed,
    )
    c2 = next(c for c in res.criteria if c.criterion_id == "G2-2")
    assert c2.status == GateStatus.FAIL
    assert "missing trace evidence" in c2.evidence


def test_p84_g2_2_completed_mode_traces_empty_dict_fails() -> None:
    """G2-2: Mode marked COMPLETED with traces_by_system={} must fail for missing trace."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    all_completed = {
        "structured": ModeExecutionStatus.COMPLETED,
        "engine-only": ModeExecutionStatus.COMPLETED,
        "e2e": ModeExecutionStatus.COMPLETED,
        "baseline": ModeExecutionStatus.COMPLETED,
    }
    res = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        traces_by_system={},
        mode_execution_statuses=all_completed,
    )
    c2 = next(c for c in res.criteria if c.criterion_id == "G2-2")
    assert c2.status == GateStatus.FAIL
    assert "missing trace evidence" in c2.evidence


def test_p84_verify_persisted_artifacts_json_evidence_mismatch(tmp_path: Path) -> None:
    """verify_persisted_artifacts fails when Results JSON criterion evidence differs."""
    expected_gate = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.PASS,
        criteria=[
            GateCriterionResult(
                criterion_id=cid,
                title=title,
                target_rule=f"Rule for {cid}",
                status=GateStatus.PASS,
                evidence=f"Evidence for {cid}",
            )
            for cid, title in CANONICAL_G2_CRITERIA.items()
        ],
        summary="Gate G2 PASS",
    )
    rep_p, res_p = _write_valid_artifacts(tmp_path)
    res_data = json.loads(res_p.read_text(encoding="utf-8"))
    res_data["g2_gate"]["criteria"][2]["evidence"] = "Tampered JSON evidence string"
    res_p.write_text(json.dumps(res_data), encoding="utf-8")

    ok, ev, errs = verify_persisted_artifacts(
        rep_p, res_p, expected_gate, require_exact_evidence=True
    )
    assert ok is False
    assert any("Results JSON criterion 'G2-3' evidence mismatch" in e for e in errs)


def test_p84_verify_persisted_artifacts_markdown_evidence_mismatch(tmp_path: Path) -> None:
    """verify_persisted_artifacts fails when Report Section H criterion evidence differs."""
    expected_gate = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.PASS,
        criteria=[
            GateCriterionResult(
                criterion_id=cid,
                title=title,
                target_rule=f"Rule for {cid}",
                status=GateStatus.PASS,
                evidence=f"Evidence for {cid}",
            )
            for cid, title in CANONICAL_G2_CRITERIA.items()
        ],
        summary="Gate G2 PASS",
    )
    rep_p, res_p = _write_valid_artifacts(tmp_path)
    rep_text = rep_p.read_text(encoding="utf-8").replace(
        "Evidence for G2-4 |", "Tampered Markdown evidence |"
    )
    rep_p.write_text(rep_text, encoding="utf-8")

    ok, ev, errs = verify_persisted_artifacts(
        rep_p, res_p, expected_gate, require_exact_evidence=True
    )
    assert ok is False
    assert any("Report Section H criterion 'G2-4' evidence mismatch" in e for e in errs)


def test_p84_verify_persisted_artifacts_markdown_summary_mismatch(tmp_path: Path) -> None:
    """verify_persisted_artifacts fails when Report Section H overall summary differs."""
    expected_gate = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.PASS,
        criteria=[
            GateCriterionResult(
                criterion_id=cid,
                title=title,
                target_rule=f"Rule for {cid}",
                status=GateStatus.PASS,
                evidence=f"Evidence for {cid}",
            )
            for cid, title in CANONICAL_G2_CRITERIA.items()
        ],
        summary="Gate G2 PASS",
    )
    rep_p, res_p = _write_valid_artifacts(tmp_path)
    rep_text = rep_p.read_text(encoding="utf-8").replace(
        "Gate G2 PASS", "Tampered overall markdown summary"
    )
    rep_p.write_text(rep_text, encoding="utf-8")

    ok, ev, errs = verify_persisted_artifacts(rep_p, res_p, expected_gate)
    assert ok is False
    assert any("Report Section H overall summary mismatch" in e for e in errs)


def test_p84_verify_persisted_artifacts_json_summary_mismatch(tmp_path: Path) -> None:
    """verify_persisted_artifacts fails when Results JSON overall summary differs."""
    expected_gate = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.PASS,
        criteria=[
            GateCriterionResult(
                criterion_id=cid,
                title=title,
                target_rule=f"Rule for {cid}",
                status=GateStatus.PASS,
                evidence=f"Evidence for {cid}",
            )
            for cid, title in CANONICAL_G2_CRITERIA.items()
        ],
        summary="Gate G2 PASS",
    )
    rep_p, res_p = _write_valid_artifacts(tmp_path)
    res_data = json.loads(res_p.read_text(encoding="utf-8"))
    res_data["g2_gate"]["summary"] = "Tampered overall JSON summary"
    res_p.write_text(json.dumps(res_data), encoding="utf-8")

    ok, ev, errs = verify_persisted_artifacts(rep_p, res_p, expected_gate)
    assert ok is False
    assert any("Results JSON overall summary mismatch" in e for e in errs)


def test_p84_verify_persisted_artifacts_json_and_markdown_status_mismatches(
    tmp_path: Path,
) -> None:
    """verify_persisted_artifacts fails on JSON status or Markdown status mismatch."""
    expected_gate = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.PASS,
        criteria=[
            GateCriterionResult(
                criterion_id=cid,
                title=title,
                target_rule=f"Rule for {cid}",
                status=GateStatus.PASS,
                evidence=f"Evidence for {cid}",
            )
            for cid, title in CANONICAL_G2_CRITERIA.items()
        ],
        summary="Gate G2 PASS",
    )

    # 1. JSON overall status mismatch
    rep_p1, res_p1 = _write_valid_artifacts(tmp_path / "t1")
    res_data1 = json.loads(res_p1.read_text(encoding="utf-8"))
    res_data1["g2_gate"]["overall_status"] = "FAIL"
    res_p1.write_text(json.dumps(res_data1), encoding="utf-8")
    ok1, ev1, errs1 = verify_persisted_artifacts(rep_p1, res_p1, expected_gate)
    assert ok1 is False
    assert any("Results JSON overall status mismatch" in e for e in errs1)

    # 2. Markdown overall status mismatch
    rep_p2, res_p2 = _write_valid_artifacts(tmp_path / "t2")
    rep_text2 = rep_p2.read_text(encoding="utf-8").replace(
        "### Overall Gate Determination: **`PASS`**",
        "### Overall Gate Determination: **`FAIL`**",
    )
    rep_p2.write_text(rep_text2, encoding="utf-8")
    ok2, ev2, errs2 = verify_persisted_artifacts(rep_p2, res_p2, expected_gate)
    assert ok2 is False
    assert any("Report Section H overall status mismatch" in e for e in errs2)


def test_p84_finalization_read_back_success_preserves_gate(tmp_path: Path) -> None:
    """finalize_evaluation_artifacts preserves PASS gate on successful disk read-back."""
    metrics = _canonical_metrics()
    sc_defs, cats = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    all_completed = {
        m: ModeExecutionStatus.COMPLETED for m in ("structured", "engine-only", "e2e", "baseline")
    }

    rep_file = tmp_path / "report.md"
    res_file = tmp_path / "results.json"

    # Pre-populate provisional gate result
    rep_run = normalize_run_result(
        metrics_by_system=metrics, scenario_ids=sc_ids, traces_by_system=traces
    )
    provisional = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        report_path=rep_file,
        results_path=res_file,
        traces_by_system=traces,
        reproducibility_run=rep_run,
        mode_execution_statuses=all_completed,
    )

    payload = {
        "timestamp": "20261115_000000",
        "mode": "all",
        "mode_execution_statuses": {k: v.value for k, v in all_completed.items()},
        "scenario_count": len(sc_defs),
        "question_count": sum(len(s.get("questions", [])) for s in sc_defs),
        "scenarios": sc_ids,
        "categories": {cat: 1 for cat in cats},
        "retrieval_method": "fastembed",
        "provider_info": {"e2e": "test", "baseline": "test"},
        "metrics": {k: v.model_dump() for k, v in metrics.items()},
        "traces": {k: [t.model_dump() for t in v] for k, v in traces.items()},
        "failures": [],
        "designed_to_lose_accounting": [],
        "g2_gate": {},
    }

    final_gate, ok = finalize_evaluation_artifacts(
        report_file=rep_file,
        out_file=res_file,
        provisional_gate_result=provisional,
        metrics_by_system=metrics,
        traces_by_system=traces,
        scenarios=sc_defs,
        timestamp_str="20261115_000000",
        mode_statuses=all_completed,
        results_payload=payload,
    )

    assert ok is True
    assert final_gate.overall_status == GateStatus.PASS
    assert final_gate.criteria[0].status == GateStatus.PASS
    assert (
        "Artifacts verified on disk and read back successfully" in final_gate.criteria[0].evidence
    )

    # Disk artifacts match final_gate exactly
    disk_ok, disk_ev, disk_errs = verify_persisted_artifacts(
        report_path=rep_file,
        results_path=res_file,
        expected_gate_result=final_gate,
        ignore_g2_1_status=False,
        require_exact_evidence=True,
    )
    assert disk_ok is True
    assert disk_errs == []


def test_p84_finalization_read_back_failure_forces_g2_1_and_overall_fail(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """finalize_evaluation_artifacts forces G2-1 and overall gate to FAIL on read-back failure."""
    import eval.run_eval as re_mod

    metrics = _canonical_metrics()
    sc_defs, cats = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    all_completed = {
        m: ModeExecutionStatus.COMPLETED for m in ("structured", "engine-only", "e2e", "baseline")
    }

    rep_file = tmp_path / "report.md"
    res_file = tmp_path / "results.json"

    rep_run = normalize_run_result(
        metrics_by_system=metrics, scenario_ids=sc_ids, traces_by_system=traces
    )
    provisional = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        report_path=rep_file,
        results_path=res_file,
        traces_by_system=traces,
        reproducibility_run=rep_run,
        mode_execution_statuses=all_completed,
    )

    payload = {
        "timestamp": "20261115_000000",
        "mode": "all",
        "mode_execution_statuses": {k: v.value for k, v in all_completed.items()},
        "scenario_count": len(sc_defs),
        "question_count": sum(len(s.get("questions", [])) for s in sc_defs),
        "scenarios": sc_ids,
        "categories": {cat: 1 for cat in cats},
        "retrieval_method": "fastembed",
        "provider_info": {"e2e": "test", "baseline": "test"},
        "metrics": {k: v.model_dump() for k, v in metrics.items()},
        "traces": {k: [t.model_dump() for t in v] for k, v in traces.items()},
        "failures": [],
        "designed_to_lose_accounting": [],
        "g2_gate": {},
    }

    orig_verify = re_mod.verify_persisted_artifacts
    step6_call_count = 0

    def mock_verify(*args: Any, **kwargs: Any) -> tuple[bool, list[str], list[str]]:
        nonlocal step6_call_count
        # Step 3 passes ignore_g2_1_status=True; let it succeed
        if kwargs.get("ignore_g2_1_status"):
            return orig_verify(*args, **kwargs)
        # Step 6 passes ignore_g2_1_status=False
        step6_call_count += 1
        if step6_call_count == 1:
            # First call in Step 6: simulate verification failure (e.g. disk corruption)
            return False, [], ["Simulated disk corruption on final read-back"]
        # Subsequent call verifying the corrected failure artifacts: let it run normally
        return orig_verify(*args, **kwargs)

    monkeypatch.setattr(re_mod, "verify_persisted_artifacts", mock_verify)

    final_gate, ok = finalize_evaluation_artifacts(
        report_file=rep_file,
        out_file=res_file,
        provisional_gate_result=provisional,
        metrics_by_system=metrics,
        traces_by_system=traces,
        scenarios=sc_defs,
        timestamp_str="20261115_000000",
        mode_statuses=all_completed,
        results_payload=payload,
    )

    # 1. Verification returned False
    assert ok is False

    # 2. G2-1 forced to FAIL
    g2_1 = next(c for c in final_gate.criteria if c.criterion_id == "G2-1")
    assert g2_1.status == GateStatus.FAIL
    assert "Simulated disk corruption on final read-back" in g2_1.evidence

    # 3. Overall gate forced to FAIL, NEVER PASS
    assert final_gate.overall_status == GateStatus.FAIL
    assert "Gate G2 FAILED: Final artifact verification failed." in final_gate.summary

    # 4. Corrected disk artifacts reflect the FAIL result
    res_disk = json.loads(res_file.read_text(encoding="utf-8"))
    assert res_disk["g2_gate"]["overall_status"] == "FAIL"
    assert res_disk["g2_gate"]["criteria"][0]["status"] == "FAIL"


def test_p84_markdown_cell_reversible_round_trip() -> None:
    """Verifies that Markdown table cell serialization and deserialization is 100% reversible."""
    cases = [
        "",
        "plain text without special characters",
        "text with | pipe",
        "text with || double | pipes",
        "text with \\ single backslash",
        "text with \\\\ double backslash",
        "text with \\| escaped pipe already",
        "Line 1\nLine 2",
        "Line 1\r\nLine 2\rLine 3",
        "Literal <br> in text",
        "Literal <br/> in text",
        "Literal <br /> in text",
        "Literal <BR> and <Br/> in text",
        "Literal \\<br> with pre-existing backslash",
        "Multiple <br> and <br/> and <br /> together",
        "Combined: \\ and | and <br> and <br/> and <br /> and \n and \r\n and <BR> tag",
        "C:\\Users\\test\\path | status: OK\nNotes: <br> tag test | done",
        "<tag attr=\"value\"> & 'quotes' < 100",
    ]

    for original in cases:
        serialized = serialize_markdown_table_cell(original)
        # Ensure serialized form contains no raw newlines or unescaped pipes
        assert "\n" not in serialized
        assert "\r" not in serialized
        # Ensure literal <br> is escaped with backslash
        if "<br>" in original:
            assert "\\<br>" in serialized

        deserialized = deserialize_markdown_table_cell(serialized)
        expected = original.replace("\r\n", "\n").replace("\r", "\n")
        assert deserialized == expected, (
            f"Round-trip failed for {original!r}:\n"
            f"  Serialized: {serialized!r}\n"
            f"  Got:        {deserialized!r}\n"
            f"  Expected:   {expected!r}"
        )


def test_p84_section_h_pipe_and_newline_serialization_and_recovery(tmp_path: Path) -> None:
    """Section H serializes pipes, newlines, and <br> safely and recovers original exactly."""
    metrics = _canonical_metrics()
    sc_defs, cats = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    all_completed = {
        m: ModeExecutionStatus.COMPLETED for m in ("structured", "engine-only", "e2e", "baseline")
    }

    rep_file = tmp_path / "report.md"
    res_file = tmp_path / "results.json"

    # Define criteria with explicit literal pipes, backslashes, <br>, and newlines across fields
    test_criteria = [
        GateCriterionResult(
            criterion_id="G2-1",
            title="Artifact Generation",
            target_rule="Artifacts present | valid format\nVerified on disk with <br> tag",
            status=GateStatus.PASS,
            evidence="2/2 artifacts present | report.md & results.json\nStrict with <br>",
        ),
        GateCriterionResult(
            criterion_id="G2-2",
            title="Multi-Mode Comparative Evaluation",
            target_rule="All 4 modes executed | COMPLETED | rule has <br/>",
            status=GateStatus.PASS,
            evidence="Modes: structured | engine-only | e2e | baseline\nAll 4 completed with <br/>",
        ),
        GateCriterionResult(
            criterion_id="G2-3",
            title="Scenario Catalog Coverage",
            target_rule=">=40 scenarios | >=100 questions\n10 categories present | <br />",
            status=GateStatus.PASS,
            evidence="Coverage: 40/40 sc | 100/100 q | 10/10 cats\nZero missing cats | <br />",
        ),
        GateCriterionResult(
            criterion_id="G2-4",
            title="Write-Time Resolution Superiority",
            target_rule="MM accuracy >= baseline + delta\nStaleness <= threshold \\ test",
            status=GateStatus.PASS,
            evidence="Delta: +0.25 | Staleness: 0.05 | Outperforms baseline\nConfirmed \\ ok",
        ),
        GateCriterionResult(
            criterion_id="G2-5",
            title="Decision Auditability and Provenance",
            target_rule="Audit coverage >= 0.90 | Provenance >= 0.80",
            status=GateStatus.PASS,
            evidence="Audit log: 1.00 | Provenance: 1.00 | Full trace complete\nPass",
        ),
        GateCriterionResult(
            criterion_id="G2-6",
            title="Designed-To-Lose Honesty & Failure Accounting",
            target_rule=">=3 DTL scenarios | honest failure\nAccounting logged in Section E",
            status=GateStatus.PASS,
            evidence="DTL cases: 5/5 valid | Honest review & abstention\nVerified",
        ),
        GateCriterionResult(
            criterion_id="G2-7",
            title="Deterministic Reproducibility",
            target_rule="Identical results across repeated runs\nZero divergence",
            status=GateStatus.PASS,
            evidence="Run 1 vs Run 2: 100% agreement | Zero drift\nReproducibility PASS",
        ),
    ]

    expected_gate = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.PASS,
        criteria=test_criteria,
        summary="Gate G2 PASSED: All 7 acceptance criteria satisfied with empirical evidence.",
    )

    report_content = generate_evaluation_report(
        metrics_by_system=metrics,
        traces_by_system=traces,
        scenarios=sc_defs,
        gate_result=expected_gate,
        timestamp="20261115_000000",
        mode_execution_statuses=all_completed,
    )
    rep_file.write_text(report_content, encoding="utf-8")

    results_payload = {
        "timestamp": "20261115_000000",
        "mode": "all",
        "mode_execution_statuses": {k: v.value for k, v in all_completed.items()},
        "scenario_count": len(sc_defs),
        "question_count": sum(len(s.get("questions", [])) for s in sc_defs),
        "scenarios": sc_ids,
        "categories": {cat: 1 for cat in cats},
        "retrieval_method": "fastembed",
        "provider_info": {"e2e": "test", "baseline": "test"},
        "metrics": {k: v.model_dump() for k, v in metrics.items()},
        "traces": {k: [t.model_dump() for t in v] for k, v in traces.items()},
        "failures": [],
        "designed_to_lose_accounting": [],
        "g2_gate": expected_gate.model_dump(),
    }
    res_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    # 1. verify_persisted_artifacts passes with exact evidence required
    assert serialize_markdown_table_cell("a|b\nc") == "a\\|b<br>c"
    is_ok, ev, errs = verify_persisted_artifacts(
        report_path=rep_file,
        results_path=res_file,
        expected_gate_result=expected_gate,
        ignore_g2_1_status=False,
        require_exact_evidence=True,
    )
    assert is_ok is True, f"Verification failed with errors: {errs}"
    assert errs == []

    # 2. Check Section H structure: exactly 7 parseable rows
    sec_h = report_content.split("## H. G2 Evaluation Gate Verification", 1)[1]
    table_lines = [line.strip() for line in sec_h.splitlines() if line.strip().startswith("| `G2-")]
    assert len(table_lines) == 7

    for line, exp_c in zip(table_lines, test_criteria, strict=True):
        cells = split_markdown_table_row(line)
        assert len(cells) == 5
        cid = cells[0].strip("`")
        assert cid == exp_c.criterion_id
        assert deserialize_markdown_table_cell(cells[1]) == exp_c.title
        assert deserialize_markdown_table_cell(cells[2]) == exp_c.target_rule
        assert f"**`{exp_c.status.value}`**" in cells[3]
        decoded_evidence = deserialize_markdown_table_cell(cells[4])
        assert decoded_evidence == exp_c.evidence


def test_p84_section_h_tampered_serialized_evidence_fails(tmp_path: Path) -> None:
    """Tampered serialized evidence in Markdown report causes verification failure."""
    metrics = _canonical_metrics()
    sc_defs, cats = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    all_completed = {
        m: ModeExecutionStatus.COMPLETED for m in ("structured", "engine-only", "e2e", "baseline")
    }

    rep_file = tmp_path / "report.md"
    res_file = tmp_path / "results.json"

    expected_gate = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.PASS,
        criteria=[
            GateCriterionResult(
                criterion_id=cid,
                title=title,
                target_rule=f"Rule for {cid}",
                status=GateStatus.PASS,
                evidence=f"Evidence with pipe | and\nnewline for {cid}",
            )
            for cid, title in CANONICAL_G2_CRITERIA.items()
        ],
        summary="Gate G2 PASS",
    )

    report_content = generate_evaluation_report(
        metrics_by_system=metrics,
        traces_by_system=traces,
        scenarios=sc_defs,
        gate_result=expected_gate,
        timestamp="20261115_000000",
        mode_execution_statuses=all_completed,
    )
    tampered_content = report_content.replace(
        "Evidence with pipe \\| and<br>newline for G2-4",
        "Corrupted evidence \\| modified",
    )
    rep_file.write_text(tampered_content, encoding="utf-8")

    results_payload = {
        "timestamp": "20261115_000000",
        "mode": "all",
        "mode_execution_statuses": {k: v.value for k, v in all_completed.items()},
        "scenario_count": len(sc_defs),
        "question_count": sum(len(s.get("questions", [])) for s in sc_defs),
        "scenarios": sc_ids,
        "categories": {cat: 1 for cat in cats},
        "retrieval_method": "fastembed",
        "provider_info": {"e2e": "test", "baseline": "test"},
        "metrics": {k: v.model_dump() for k, v in metrics.items()},
        "traces": {k: [t.model_dump() for t in v] for k, v in traces.items()},
        "failures": [],
        "designed_to_lose_accounting": [],
        "g2_gate": expected_gate.model_dump(),
    }
    res_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    is_ok, ev, errs = verify_persisted_artifacts(
        report_path=rep_file,
        results_path=res_file,
        expected_gate_result=expected_gate,
        ignore_g2_1_status=False,
        require_exact_evidence=True,
    )
    assert is_ok is False
    assert any("Report Section H criterion 'G2-4' evidence mismatch" in e for e in errs)


def test_p84_section_h_table_contains_exactly_7_parseable_rows_regardless_of_content(
    tmp_path: Path,
) -> None:
    """Extreme evidence (pipes, newlines, quotes, backslashes) produces exactly 7 5-column rows."""
    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    all_completed = {
        m: ModeExecutionStatus.COMPLETED for m in ("structured", "engine-only", "e2e", "baseline")
    }

    extreme_evidence_list = [
        "Multiple pipes: |||| in || a || row",
        "Many newlines:\n\n\nline2\r\nline3\n\nline4",
        "Backslashes: C:\\Program Files\\MemoryMesh\\eval | OK\\done",
        "Quotes and brackets: <tag attr=\"val\"> & 'single' | `code`",
        "Trailing backslash: path\\to\\dir\\",
        "",
        "Mixed: \\| escaped in source\nnext line | trailing |",
    ]

    criteria = [
        GateCriterionResult(
            criterion_id=cid,
            title=title,
            target_rule=f"Rule with | pipe\nand newline {i}",
            status=GateStatus.PASS,
            evidence=extreme_evidence_list[i],
        )
        for i, (cid, title) in enumerate(CANONICAL_G2_CRITERIA.items())
    ]

    gate_result = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.PASS,
        criteria=criteria,
        summary="Extreme evidence test",
    )

    report_content = generate_evaluation_report(
        metrics_by_system=metrics,
        traces_by_system=traces,
        scenarios=sc_defs,
        gate_result=gate_result,
        timestamp="20261115_000000",
        mode_execution_statuses=all_completed,
    )

    sec_h = report_content.split("## H. G2 Evaluation Gate Verification", 1)[1]
    table_lines = [line.strip() for line in sec_h.splitlines() if line.strip().startswith("| `G2-")]
    # Exactly 7 table rows
    assert len(table_lines) == 7

    for i, line in enumerate(table_lines):
        cells = split_markdown_table_row(line)
        assert len(cells) == 5, f"Row {i} did not have 5 columns: {cells}"
        decoded_ev = deserialize_markdown_table_cell(cells[4])
        expected_ev = extreme_evidence_list[i].replace("\r\n", "\n").replace("\r", "\n")
        assert decoded_ev == expected_ev


def test_p84_finalization_double_read_back_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """finalize_evaluation_artifacts handles persistent read-back failure deterministically."""
    import eval.run_eval as re_mod

    metrics = _canonical_metrics()
    sc_defs, cats = _canonical_scenarios_and_categories()
    sc_ids = [s["id"] for s in sc_defs]
    traces = _canonical_traces(sc_defs, list(metrics.keys()))
    all_completed = {
        m: ModeExecutionStatus.COMPLETED for m in ("structured", "engine-only", "e2e", "baseline")
    }

    rep_file = tmp_path / "report.md"
    res_file = tmp_path / "results.json"

    rep_run = normalize_run_result(
        metrics_by_system=metrics, scenario_ids=sc_ids, traces_by_system=traces
    )
    provisional = evaluate_g2_gate(
        metrics_by_system=metrics,
        executed_scenarios=sc_defs,
        report_path=rep_file,
        results_path=res_file,
        traces_by_system=traces,
        reproducibility_run=rep_run,
        mode_execution_statuses=all_completed,
    )

    payload = {
        "timestamp": "20261115_000000",
        "mode": "all",
        "mode_execution_statuses": {k: v.value for k, v in all_completed.items()},
        "scenario_count": len(sc_defs),
        "question_count": sum(len(s.get("questions", [])) for s in sc_defs),
        "scenarios": sc_ids,
        "categories": {cat: 1 for cat in cats},
        "retrieval_method": "fastembed",
        "provider_info": {"e2e": "test", "baseline": "test"},
        "metrics": {k: v.model_dump() for k, v in metrics.items()},
        "traces": {k: [t.model_dump() for t in v] for k, v in traces.items()},
        "failures": [],
        "designed_to_lose_accounting": [],
        "g2_gate": {},
    }

    orig_verify = re_mod.verify_persisted_artifacts
    step6_call_count = 0

    def mock_verify_persistent_fail(*args: Any, **kwargs: Any) -> tuple[bool, list[str], list[str]]:
        nonlocal step6_call_count
        # Step 3 passes ignore_g2_1_status=True; let it succeed
        if kwargs.get("ignore_g2_1_status"):
            return orig_verify(*args, **kwargs)
        # Step 6 passes ignore_g2_1_status=False
        step6_call_count += 1
        if step6_call_count == 1:
            # Initial final read-back failure
            return False, [], ["Initial disk verification failure"]
        if step6_call_count == 2:
            # Subsequent read-back of corrected FAIL artifacts also fails
            return False, [], ["Persistent disk verification failure"]
        return orig_verify(*args, **kwargs)

    monkeypatch.setattr(re_mod, "verify_persisted_artifacts", mock_verify_persistent_fail)

    final_gate, ok = finalize_evaluation_artifacts(
        report_file=rep_file,
        out_file=res_file,
        provisional_gate_result=provisional,
        metrics_by_system=metrics,
        traces_by_system=traces,
        scenarios=sc_defs,
        timestamp_str="20261115_000000",
        mode_statuses=all_completed,
        results_payload=payload,
    )

    # 1. finalize_evaluation_artifacts returns verification_succeeded=False
    assert ok is False

    # 2. Overall gate status is FAIL
    assert final_gate.overall_status == GateStatus.FAIL
    assert "Gate G2 FAILED: Final artifact verification failed." in final_gate.summary

    # 3. G2-1 is FAIL
    g2_1 = next(c for c in final_gate.criteria if c.criterion_id == "G2-1")
    assert g2_1.status == GateStatus.FAIL
    assert "Initial disk verification failure" in g2_1.evidence

    # 4. No infinite loop or recursion occurred (Step 6 called exactly twice)
    assert step6_call_count == 2

    # 5. Caller surfaces failure and exit code is not 0 (returns 1)
    caller_exit_code = 0 if (ok and final_gate.overall_status != GateStatus.FAIL) else 1
    assert caller_exit_code == 1


def test_p84_main_exit_code_failure_on_read_back_failure(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Verifies that main() returns exit code 1 and surfaces artifact verification failure."""
    import eval.run_eval as re_mod

    rep_file = tmp_path / "report.md"
    res_file = tmp_path / "results.json"

    failed_gate = GateEvaluationResult(
        gate_name="G2",
        timestamp="20261115_000000",
        overall_status=GateStatus.FAIL,
        criteria=[
            GateCriterionResult(
                criterion_id="G2-1",
                title="Artifact Generation",
                target_rule="Generate report.md and results.json",
                status=GateStatus.FAIL,
                evidence="Artifact read-back verification failed",
            )
        ],
        summary="Gate G2 FAILED: Final artifact verification failed.",
    )

    monkeypatch.setattr(
        re_mod,
        "finalize_evaluation_artifacts",
        lambda **kwargs: (failed_gate, False),
    )

    metrics = _canonical_metrics()
    sc_defs, _ = _canonical_scenarios_and_categories()
    traces = _canonical_traces(sc_defs, list(metrics.keys()))

    monkeypatch.setattr(re_mod, "load_scenarios", lambda *a, **kw: sc_defs[:1])
    monkeypatch.setattr(
        re_mod,
        "run_system_on_scenarios",
        lambda sys_id, *a, **kw: (metrics["MM (structured)"], traces["MM (structured)"][:1]),
    )

    test_args = [
        "run_eval.py",
        "--mode",
        "structured",
        "--output",
        str(res_file),
        "--report-output",
        str(rep_file),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    exit_code = re_mod.main()
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "G2 EVALUATION GATE STATUS: [FAIL]" in captured
    assert "[ARTIFACT VERIFICATION FAILURE]" in captured


def test_p84_split_markdown_table_row_backslashes_and_delimiters() -> None:
    """Verifies that split_markdown_table_row correctly distinguishes backslashes and delimiters."""
    # 1. Plain delimiters
    assert split_markdown_table_row("| a | b |") == ["a", "b"]

    # 2. Single backslash before character (ordinary escaped character)
    assert split_markdown_table_row("| foo\\bar | baz |") == ["foo\\bar", "baz"]

    # 3. Single backslash before pipe: escaped pipe treated as content
    assert split_markdown_table_row("| a\\|b | c |") == ["a\\|b", "c"]

    # 4. Two backslashes immediately before pipe: escaped backslash, pipe is delimiter
    assert split_markdown_table_row("| a\\\\| b |") == ["a\\\\", "b"]
    assert split_markdown_table_row("| a\\\\ | b |") == ["a\\\\", "b"]

    # 5. Three backslashes before pipe: escaped backslash + escaped pipe (content)
    assert split_markdown_table_row("| a\\\\\\|b | c |") == ["a\\\\\\|b", "c"]

    # 6. Four backslashes before pipe: two escaped backslashes, pipe is delimiter
    assert split_markdown_table_row("| a\\\\\\\\| b |") == ["a\\\\\\\\", "b"]

    # 7. Five backslashes before pipe: two escaped backslashes + escaped pipe (content)
    assert split_markdown_table_row("| a\\\\\\\\\\|b | c |") == ["a\\\\\\\\\\|b", "c"]

    # 8. Deserialization recovers exact original text from parsed cells
    row = "| a\\\\| b |"
    cells = split_markdown_table_row(row)
    assert deserialize_markdown_table_cell(cells[0]) == "a\\"

    row3 = "| a\\\\\\|b | c |"
    cells3 = split_markdown_table_row(row3)
    assert deserialize_markdown_table_cell(cells3[0]) == "a\\|b"

    # 9. Full Section H 5-column row with backslash immediately before delimiter
    orig_path = "C:\\Program Files\\MemoryMesh\\"
    ser_path = serialize_markdown_table_cell(orig_path)
    h_row = f"| `G2-1` | Title | Rule | **`PASS`** | {ser_path} |"
    h_cells = split_markdown_table_row(h_row)
    assert len(h_cells) == 5
    assert h_cells[0].strip("`") == "G2-1"
    assert deserialize_markdown_table_cell(h_cells[4]) == orig_path


def test_p84_section_h_strict_five_columns_enforced(tmp_path: Path) -> None:
    """verify_persisted_artifacts fails when Section H table row does not have exactly 5 columns."""
    rep_p, res_p = _write_valid_artifacts(tmp_path)
    # Inject an extra unescaped pipe column into G2-1 row (producing 6 columns)
    rep_text = rep_p.read_text(encoding="utf-8")
    rep_text_tampered = rep_text.replace(
        "| `G2-1` | Artifact Generation |",
        "| `G2-1` | Artifact Generation | Extra Unescaped Column |",
    )
    rep_p.write_text(rep_text_tampered, encoding="utf-8")

    ok, ev, errs = verify_persisted_artifacts(rep_p, res_p)
    assert ok is False
    assert any("Report Section H table row malformed for criterion 'G2-1'" in e for e in errs)
    assert any("expected 5 columns, got 6" in e for e in errs)
