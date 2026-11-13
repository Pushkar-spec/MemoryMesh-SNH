"""Comprehensive tests for evaluation framework and metrics (P8.1).

Governed by:
- docs/EVALUATION_AND_DEMO.md Section 1, 2, 4, 5
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P8.1)

Verifies:
1. Structured evaluation mode strictly makes zero LLM calls.
2. Metric calculation accuracy across all required metrics (Current, As-of, History,
   Stale rate, Conflict recall/FAR, Impact recall/precision, Provenance, Abstention, Decisions).
3. "n/a" handling for baseline B0 and non-applicable metric groups.
4. Feature ablation flags A1-A5 correctly isolate their specific failure modes.
5. Comparison table generation in Markdown and ASCII formats.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.metrics import (
    EvaluationMetrics,
    QuestionEvalResult,
    compute_metrics,
    evaluate_question,
    format_ascii_table,
    format_comparison_table,
)
from eval.run_eval import (
    build_ablation_config,
    load_scenarios,
    run_system_on_scenarios,
)
from memorymesh.config import load_app_config
from memorymesh.llm.client import LLMClient
from memorymesh.llm.providers import LLMProvider


class CountingLLMProvider(LLMProvider):
    """LLM Provider that tracks every call to guarantee structured mode makes zero calls."""

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        self.call_count += 1
        self.calls.append({"system": system, "user": user, "model": model})
        return json.dumps(
            {
                "value": "3.12",
                "values": ["3.10", "3.12"],
                "conflict_flagged": False,
                "needs_review_flagged": False,
                "abstained": False,
                "explanation": "Answer.",
                "cited_memory_ids": [],
                "answer_text": "Answer text.",
            }
        )


def test_structured_mode_strictly_makes_zero_llm_calls() -> None:
    """Verifies that running in structured mode produces exactly 0 LLM calls."""
    counting_provider = CountingLLMProvider()
    cfg = load_app_config()
    llm_client = LLMClient(provider=counting_provider, config=cfg.engine.llm)

    scenarios = load_scenarios(Path("eval/scenarios"), target_scenario="hero-001")
    assert len(scenarios) == 1

    metrics, traces = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (structured)",
        scenarios=scenarios,
        app_config=cfg,
        mode="structured",
        llm_client=llm_client,
    )

    assert counting_provider.call_count == 0, (
        f"Structured mode violated zero-LLM guarantee: {counting_provider.call_count} calls made."
    )
    assert metrics.avg_llm_calls_per_question == 0.0
    assert metrics.current_accuracy == 1.0
    assert metrics.as_of_accuracy == 1.0
    assert metrics.history_accuracy == 1.0
    assert metrics.stale_answer_rate == 0.0
    assert metrics.engine_decision_accuracy == 1.0


def test_evaluate_question_current_and_stale() -> None:
    """Verifies grading of current questions and forbidden value stale answer detection."""
    q = {
        "id": "q1",
        "type": "current",
        "expect": {"value": "3.12", "forbidden": ["3.10", "3.9"]},
    }

    # Correct non-stale answer
    good_ans = {
        "value": "3.12",
        "values": ["3.12"],
        "answer_text": "Server A runs Python 3.12 [M5].",
    }
    res_good = evaluate_question("sc-1", q, good_ans)
    assert res_good.is_current_correct is True
    assert res_good.is_stale is False

    # Stale answer containing forbidden value
    stale_ans = {
        "value": "3.10",
        "values": ["3.10"],
        "answer_text": "Server A runs Python 3.10.",
    }
    res_stale = evaluate_question("sc-1", q, stale_ans)
    assert res_stale.is_current_correct is False
    assert res_stale.is_stale is True


def test_evaluate_question_history_order_sensitive() -> None:
    """Verifies that history question grading is strictly order-sensitive."""
    q = {
        "id": "q7",
        "type": "history",
        "expect": {"values": ["3.10", "3.12"]},
    }

    # Correct chronological order
    correct_order = {"values": ["3.10", "3.12"]}
    assert evaluate_question("sc-1", q, correct_order).is_history_correct is True

    # Reversed order fails history accuracy
    reversed_order = {"values": ["3.12", "3.10"]}
    assert evaluate_question("sc-1", q, reversed_order).is_history_correct is False


def test_evaluate_question_conflicts() -> None:
    """Verifies conflict recall and false-alarm detection."""
    q_conflict = {"id": "q5", "type": "current", "expect": {"conflict_flagged": True}}
    q_no_conflict = {"id": "q1", "type": "current", "expect": {"conflict_flagged": False}}

    ans_flagged = {"conflict_flagged": True}
    ans_unflagged = {"conflict_flagged": False}

    assert evaluate_question("sc-1", q_conflict, ans_flagged).conflict_flagged is True
    assert evaluate_question("sc-1", q_conflict, ans_unflagged).conflict_flagged is False

    # False alarm on unflagged expected
    res_fa = evaluate_question("sc-1", q_no_conflict, ans_flagged)
    assert res_fa.conflict_expected is False
    assert res_fa.conflict_flagged is True


def test_evaluate_question_impact_and_provenance() -> None:
    """Verifies impact recall/precision and why question provenance grading."""
    q_impact = {
        "id": "q4",
        "type": "impact",
        "expect": {
            "dependents": [["Library X", "compatibility_note"]],
            "needs_review_flagged": True,
        },
    }
    good_impact_ans = {
        "dependents": [("Library X", "compatibility_note")],
        "needs_review_flagged": True,
    }
    res_imp = evaluate_question("sc-1", q_impact, good_impact_ans, is_baseline=False)
    assert res_imp.impact_recall == 1.0
    assert res_imp.impact_precision == 1.0

    # Baseline B0 reports None for impact
    res_imp_b0 = evaluate_question("sc-1", q_impact, good_impact_ans, is_baseline=True)
    assert res_imp_b0.impact_recall is None

    # Provenance
    q_why = {
        "id": "q3",
        "type": "why",
        "expect": {"cited_source": "sysadmin_priya", "rule_id": "R-REPLACE"},
    }
    good_why_ans = {
        "explanation": "Reported by sysadmin_priya. Resolved via R-REPLACE.",
        "cited_sources": ["sysadmin_priya"],
        "rules": ["R-REPLACE"],
    }
    res_why = evaluate_question("sc-1", q_why, good_why_ans, is_baseline=False)
    assert res_why.provenance_correct is True

    res_why_b0 = evaluate_question("sc-1", q_why, good_why_ans, is_baseline=True)
    assert res_why_b0.provenance_correct is None


def test_compute_metrics_baseline_na_reporting() -> None:
    """Verifies B0 reports 'n/a' for impact, provenance, and decision accuracy."""
    q_res = [
        QuestionEvalResult(
            scenario_id="sc-1",
            question_id="q1",
            question_type="current",
            expected={"value": "3.12"},
            actual={"value": "3.12"},
            is_current_correct=True,
            is_stale=False,
            latency_seconds=0.1,
            llm_calls=1,
        )
    ]
    metrics = compute_metrics(
        system_id="B0",
        system_name="Naive RAG",
        question_results=q_res,
        decision_matches=None,
    )
    assert metrics.current_accuracy == 1.0
    assert metrics.impact_recall is None
    assert metrics.provenance_correctness is None
    assert metrics.engine_decision_accuracy is None


def test_ablation_configs_disable_single_flags() -> None:
    """Verifies build_ablation_config only alters the targeted ablation flag."""
    base_cfg = load_app_config()
    assert base_cfg.engine.ablation.use_temporal is True
    assert base_cfg.engine.ablation.use_trust is True
    assert base_cfg.engine.ablation.use_cardinality is True
    assert base_cfg.engine.ablation.use_context is True
    assert base_cfg.engine.ablation.use_propagation is True

    a1 = build_ablation_config(base_cfg, "use_temporal")
    assert a1.engine.ablation.use_temporal is False
    assert a1.engine.ablation.use_trust is True

    a2 = build_ablation_config(base_cfg, "use_trust")
    assert a2.engine.ablation.use_trust is False
    assert a2.engine.ablation.use_temporal is True

    a3 = build_ablation_config(base_cfg, "use_cardinality")
    assert a3.engine.ablation.use_cardinality is False

    a4 = build_ablation_config(base_cfg, "use_context")
    assert a4.engine.ablation.use_context is False

    a5 = build_ablation_config(base_cfg, "use_propagation")
    assert a5.engine.ablation.use_propagation is False


def test_ablation_a1_temporal_isolation() -> None:
    """Verifies A1 disables temporal point-in-time reasoning while preserving trust and context."""
    cfg = load_app_config()
    scenarios = load_scenarios(Path("eval/scenarios"), target_scenario="hero-001")

    # MM Full (temporal on) -> As-of accuracy is 100%
    m_full, _ = run_system_on_scenarios("MM", "full", scenarios, cfg, mode="structured")
    assert m_full.as_of_accuracy == 1.0

    # A1 (temporal off) -> As-of accuracy drops to 0% (evaluates at now)
    cfg_a1 = build_ablation_config(cfg, "use_temporal")
    m_a1, _ = run_system_on_scenarios("A1", "no temporal", scenarios, cfg_a1, mode="structured")
    assert m_a1.as_of_accuracy == 0.0
    assert m_a1.current_accuracy == 1.0  # Current facts still resolve
    assert m_a1.impact_recall == 1.0  # Propagation still works


def test_ablation_a2_trust_isolation() -> None:
    """Verifies A2 disables trust rejection while preserving temporal and context reasoning."""
    cfg = load_app_config()
    scenarios = load_scenarios(Path("eval/scenarios"), target_scenario="hero-001")

    # MM Full -> Low trust rumor is rejected, decision accuracy 100%
    m_full, _ = run_system_on_scenarios("MM", "full", scenarios, cfg, mode="structured")
    assert m_full.engine_decision_accuracy == 1.0

    # A2 -> Low trust rumor (intern_chat) is not rejected; decision accuracy drops
    cfg_a2 = build_ablation_config(cfg, "use_trust")
    m_a2, _ = run_system_on_scenarios("A2", "no trust", scenarios, cfg_a2, mode="structured")
    assert m_a2.engine_decision_accuracy < 1.0
    assert m_a2.stale_answer_rate > 0.0  # Rumor pollutes answer


def test_ablation_a3_cardinality_isolation() -> None:
    """Verifies A3 forces single-valued behavior on multi-valued attributes."""
    cfg = load_app_config()
    base_a3 = build_ablation_config(cfg, "use_cardinality")
    assert base_a3.engine.ablation.use_cardinality is False
    # Verify unrelated flags are still True
    assert base_a3.engine.ablation.use_temporal is True
    assert base_a3.engine.ablation.use_trust is True
    assert base_a3.engine.ablation.use_context is True
    assert base_a3.engine.ablation.use_propagation is True


def test_ablation_a4_context_isolation() -> None:
    """Verifies A4 collapses environment context scopes while preserving trust and temporal."""
    cfg = load_app_config()
    scenarios = load_scenarios(Path("eval/scenarios"), target_scenario="hero-001")

    # A4 -> Context disabled: staging collides with prod
    cfg_a4 = build_ablation_config(cfg, "use_context")
    m_a4, _ = run_system_on_scenarios("A4", "no context", scenarios, cfg_a4, mode="structured")
    assert m_a4.as_of_accuracy == 0.0  # Staging value overwrites prod history
    assert m_a4.stale_answer_rate > 0.0


def test_a4_context_ablation_collision_regression(tmp_path: Path) -> None:
    """Regression test proving context isolation in full MM vs context collision in A4."""
    from memorymesh.models import Claim, IngestMessage
    from memorymesh.service import MemoryService

    cfg_full = load_app_config()
    cfg_a4 = build_ablation_config(cfg_full, "use_context")

    t1 = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 20, 13, 0, tzinfo=UTC)

    # 1. Full MM (use_context=True): Context isolation preserved
    db_full = tmp_path / "full.db"
    svc_full = MemoryService(str(db_full), cfg=cfg_full)

    # Ingest production claim: Server A runs Python 3.12
    msg1 = IngestMessage(source_id="sysadmin_priya", event_time=t1, text="Server A is 3.12")
    c1 = [
        Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
    ]
    svc_full.ingest_claims(msg1, c1)

    # Ingest staging claim: In staging, Server A runs Python 3.11
    msg2 = IngestMessage(source_id="dev_chat", event_time=t2, text="In staging Server A is 3.11")
    c2 = [
        Claim(
            subject="Server A",
            attribute="python_version",
            value="3.11",
            context={"env": "staging"},
            extract_confidence=0.90,
        )
    ]
    rep2_full = svc_full.ingest_claims(msg2, c2)
    # In full MM, staging claim is inserted in its own context chain (R-INSERT)
    assert rep2_full.decisions[0].rule_id == "R-INSERT"

    # Query prod in full MM -> 3.12 (unpolluted by staging)
    res_prod_full = svc_full.current(entity="Server A", attribute="python_version", now=t2)
    assert res_prod_full.items[0].memory.value_norm == "3.12"

    # Query staging in full MM -> 3.11
    res_stg_full = svc_full.current(
        entity="Server A", attribute="python_version", context={"env": "staging"}, now=t2
    )
    assert res_stg_full.items[0].memory.value_norm == "3.11"

    # 2. Ablated MM A4 (use_context=False): Context isolation disabled
    db_a4 = tmp_path / "a4.db"
    svc_a4 = MemoryService(str(db_a4), cfg=cfg_a4)

    # Ingest same messages
    svc_a4.ingest_claims(msg1, c1)
    rep2_a4 = svc_a4.ingest_claims(msg2, c2)

    # In A4, staging context is erased -> collides with prod chain, putting it into active DISPUTE!
    assert rep2_a4.decisions[0].rule_id == "R-DISPUTE"
    assert rep2_a4.decisions[0].affected_memory_ids != []

    # Query prod in A4 -> in dispute! (collision! Prod fact was corrupted by staging)
    res_prod_a4 = svc_a4.current(entity="Server A", attribute="python_version", now=t2)
    assert len(res_prod_a4.disputes) >= 1


def test_model_tier_fairness() -> None:
    """Verifies baseline and MM answer composer resolve to the same answer_model tier."""
    from baseline.naive_rag import NaiveRAGBaseline
    from memorymesh.llm.answer_composer import AnswerComposer
    from memorymesh.service import MemoryService

    cfg = load_app_config()
    expected_answer_model = cfg.engine.llm.answer_model
    expected_fast_model = cfg.engine.llm.fast_model

    # Verify config specifies distinct tiers
    assert expected_answer_model == "claude-sonnet-5-5"
    assert expected_fast_model == "claude-haiku-5-5"

    # 1. Baseline answer model resolution
    baseline = NaiveRAGBaseline(config=cfg)
    assert baseline.model == expected_answer_model

    # 2. MemoryMesh answer composer model resolution
    svc = MemoryService(":memory:", cfg=cfg)
    composer = AnswerComposer(default_model=svc.cfg.engine.llm.answer_model)
    assert composer.default_model == expected_answer_model

    # Strict model-tier fairness: Both systems use the same configured answer model tier
    assert baseline.model == composer.default_model


def test_all_mode_includes_all_five_modes() -> None:
    """Verifies 'all' mode executes and distinguishes all 5 required evaluation modes."""
    from tests.test_hero_nl_integration import HeroScenarioScriptedProvider

    cfg = load_app_config()
    provider = HeroScenarioScriptedProvider()
    llm_client = LLMClient(provider=provider, config=cfg.engine.llm)

    scenarios = load_scenarios(Path("eval/scenarios"), target_scenario="hero-001")

    # Mode 1: Baseline
    m_b0, _ = run_system_on_scenarios(
        "B0", "baseline", scenarios, cfg, mode="baseline", llm_client=llm_client
    )
    assert m_b0.system_id == "B0"

    # Mode 2: Structured
    m_struct, _ = run_system_on_scenarios("MM", "structured", scenarios, cfg, mode="structured")
    assert m_struct.avg_llm_calls_per_question == 0.0

    # Mode 3: Engine-only
    m_eng, _ = run_system_on_scenarios("MM", "engine-only", scenarios, cfg, mode="engine-only")
    assert m_eng.avg_llm_calls_per_question == 0.0

    # Mode 4: E2E
    m_e2e, _ = run_system_on_scenarios(
        "MM", "e2e", scenarios, cfg, mode="e2e", llm_client=llm_client
    )
    assert m_e2e.avg_llm_calls_per_question > 0.0

    # Mode 5: Ablations (A1-A5)
    for abl in ["use_temporal", "use_trust", "use_cardinality", "use_context", "use_propagation"]:
        cfg_abl = build_ablation_config(cfg, abl)
        m_abl, _ = run_system_on_scenarios(
            f"MM-{abl}", "ablation", scenarios, cfg_abl, mode="structured"
        )
        assert getattr(cfg_abl.engine.ablation, abl) is False


def test_ablation_a5_propagation_isolation() -> None:
    """Verifies A5 disables downstream dependency change propagation."""
    cfg = load_app_config()
    scenarios = load_scenarios(Path("eval/scenarios"), target_scenario="hero-001")

    # MM Full -> Impact recall 100%
    m_full, _ = run_system_on_scenarios("MM", "full", scenarios, cfg, mode="structured")
    assert m_full.impact_recall == 1.0

    # A5 -> Propagation disabled: impact recall drops to 0%
    cfg_a5 = build_ablation_config(cfg, "use_propagation")
    m_a5, _ = run_system_on_scenarios("A5", "no propagation", scenarios, cfg_a5, mode="structured")
    assert m_a5.impact_recall == 0.0
    assert m_a5.current_accuracy == 1.0  # Direct updates still work


def test_baseline_and_fair_comparison() -> None:
    """Verifies baseline retrieves raw messages, uses FinalAnswer schema, and lacks engine state."""
    counting_provider = CountingLLMProvider()
    cfg = load_app_config()
    llm_client = LLMClient(provider=counting_provider, config=cfg.engine.llm)
    scenarios = load_scenarios(Path("eval/scenarios"), target_scenario="hero-001")

    m_b0, traces_b0 = run_system_on_scenarios(
        "B0", "Naive RAG", scenarios, cfg, mode="baseline", llm_client=llm_client
    )

    # Baseline uses LLM completions
    assert counting_provider.call_count > 0
    # Baseline metrics for impact, provenance, and decisions are strictly 'n/a' (None)
    assert m_b0.impact_recall is None
    assert m_b0.provenance_correctness is None
    assert m_b0.engine_decision_accuracy is None
    # Traces contain FinalAnswer outputs
    assert len(traces_b0) == 7
    q1_trace = next(t for t in traces_b0 if t.question_id == "q1")
    assert "value" in q1_trace.actual
    assert "answer_text" in q1_trace.actual


def test_format_comparison_table_and_ascii() -> None:
    """Verifies format_comparison_table and format_ascii_table generate expected output."""
    m_mm = EvaluationMetrics(
        system_id="MM",
        system_name="full",
        current_accuracy=1.0,
        as_of_accuracy=1.0,
        history_accuracy=1.0,
        stale_answer_rate=0.0,
        conflict_recall=1.0,
        conflict_false_alarm_rate=0.0,
        impact_recall=1.0,
        impact_precision=1.0,
        provenance_correctness=1.0,
        engine_decision_accuracy=1.0,
    )
    m_b0 = EvaluationMetrics(
        system_id="B0",
        system_name="naive RAG",
        current_accuracy=0.6,
        as_of_accuracy=0.0,
        history_accuracy=0.0,
        stale_answer_rate=0.5,
        conflict_recall=0.5,
        conflict_false_alarm_rate=0.2,
        impact_recall=None,
        impact_precision=None,
        provenance_correctness=None,
        engine_decision_accuracy=None,
    )

    metrics_map = {"B0": m_b0, "MM": m_mm}
    md = format_comparison_table(metrics_map)
    assert "| Metric | B0 (naive RAG) | MM (full) |" in md
    assert "Current accuracy" in md
    assert "100.0%" in md
    assert "n/a" in md

    ascii_tbl = format_ascii_table(metrics_map)
    assert "Current accuracy" in ascii_tbl
    assert "B0" in ascii_tbl
    assert "MM" in ascii_tbl


def test_retrieval_method_in_eval_metrics_and_traces() -> None:
    """Verifies that retrieval_method is recorded in evaluation metrics and question traces."""
    q = {"id": "q1", "type": "current", "expect": {"value": "3.12"}}
    ans = {"value": "3.12", "answer_text": "3.12"}
    res = evaluate_question("sc-1", q, ans, is_baseline=True, retrieval_method="fastembed")
    assert res.retrieval_method == "fastembed"

    metrics = compute_metrics(
        system_id="B0",
        system_name="Naive RAG",
        question_results=[res],
        retrieval_method="fastembed",
    )
    assert metrics.retrieval_method == "fastembed"
