"""MemoryMesh Evaluation Runner (P8.1, P8.4).

Governed by:
- docs/EVALUATION_AND_DEMO.md
- docs/ROADMAP_AND_TASKS.md (T8.4, G2 Gate)
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P8.1, P8.4)

Supported execution modes:
- structured: deterministic engine evaluation with structured claims (zero LLM calls)
- e2e: full natural-language ingestion, query parsing, and answer composition
- baseline: naive RAG baseline (B0) using FastEmbed top-k=5 and baseline_answer prompt
- engine-only: MemoryMesh with deterministic answer composition
- ablation: runs MM (full) and feature ablations A1-A5
- all: runs baseline B0, structured MM, engine-only MM, and e2e MM
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from baseline.naive_rag import NaiveRAGBaseline  # noqa: E402
from eval.gate import (  # noqa: E402
    CANONICAL_MODES,
    GateCriterionResult,
    GateEvaluationResult,
    GateStatus,
    ModeExecutionStatus,
    evaluate_g2_gate,
    normalize_run_result,
    select_dtl_invariant_trace,
    verify_designed_to_lose_outcome,
    verify_persisted_artifacts,
)
from eval.metrics import (  # noqa: E402
    EvaluationMetrics,
    QuestionEvalResult,
    compute_metrics,
    evaluate_question,
    format_ascii_table,
)
from eval.providers import (  # noqa: E402
    DeterministicEvaluationBaselineProvider,
    DeterministicEvaluationE2EProvider,
)
from eval.report_generator import generate_evaluation_report  # noqa: E402
from memorymesh.config import AppConfig, load_app_config  # noqa: E402
from memorymesh.engine.entity import normalize_entity_name  # noqa: E402
from memorymesh.llm.client import LLMClient  # noqa: E402
from memorymesh.models import Claim, IngestMessage, MemoryStatus  # noqa: E402
from memorymesh.retrieval.retrieve import retrieve_valid_memories  # noqa: E402
from memorymesh.service import MemoryService  # noqa: E402

logger = logging.getLogger("memorymesh.eval")


def parse_iso_datetime(val: Any) -> datetime | None:
    """Safely parse ISO datetime string with UTC fallback."""
    if not val:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=UTC)
        return val.astimezone(UTC)
    s = str(val).strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def load_scenarios(scenarios_dir: Path, target_scenario: str | None = None) -> list[dict[str, Any]]:
    """Load evaluation scenarios from YAML files in scenarios_dir."""
    scenarios: list[dict[str, Any]] = []
    if not scenarios_dir.exists():
        logger.warning("Scenarios directory does not exist: %s", scenarios_dir)
        return []

    files = sorted(scenarios_dir.glob("*.yaml")) + sorted(scenarios_dir.glob("*.yml"))
    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                sc_id = str(data.get("id", ""))
                if target_scenario and target_scenario not in (
                    file_path.name,
                    file_path.stem,
                    sc_id,
                ):
                    continue
                data["_file_name"] = file_path.name
                scenarios.append(data)
        except Exception as e:
            logger.warning("Failed to parse scenario file %s: %s", file_path, e)
    return scenarios


def run_system_on_scenarios(
    system_id: str,
    system_name: str,
    scenarios: list[dict[str, Any]],
    app_config: AppConfig,
    *,
    mode: str = "structured",
    llm_client: LLMClient | None = None,
    retrieval_method: str | None = None,
    embedding_model: Any | None = None,
) -> tuple[EvaluationMetrics, list[QuestionEvalResult]]:
    """Run an evaluation run for a specified system across all loaded scenarios."""
    all_q_results: list[QuestionEvalResult] = []
    all_dec_matches: list[bool] = []
    total_messages = 0

    is_baseline = system_id.upper() in ("B0", "B0-LEXICAL")
    eff_retrieval_method = retrieval_method or ("fastembed" if is_baseline else "memorymesh")

    # In baseline mode, ensure an operational LLMClient with simulation provider if not supplied
    if is_baseline and llm_client is None:
        b0_provider = DeterministicEvaluationBaselineProvider()
        llm_client = LLMClient(provider=b0_provider, config=app_config.engine.llm)

    # In e2e mode, ensure an operational LLMClient with simulation provider if not supplied
    if mode == "e2e" and llm_client is None:
        e2e_provider = DeterministicEvaluationE2EProvider()
        llm_client = LLMClient(provider=e2e_provider, config=app_config.engine.llm)

    # In baseline mode, reuse pre-initialized embedding model across scenarios to avoid reloading
    shared_emb = embedding_model
    if is_baseline and eff_retrieval_method == "fastembed" and shared_emb is None:
        try:
            from fastembed import TextEmbedding

            shared_emb = TextEmbedding()
        except Exception:
            shared_emb = None

    for sc in scenarios:
        sc_id = str(sc.get("id", "scenario"))
        now_dt = parse_iso_datetime(sc.get("now")) or datetime.now(UTC)
        messages = sc.get("messages", [])
        total_messages += len(messages)
        questions = sc.get("questions", [])
        expected_decisions = sc.get("expected_decisions", {})
        structured_claims = sc.get("structured_claims", {})

        if is_baseline:
            # Baseline execution (B0 or B0-lexical)
            baseline = NaiveRAGBaseline(
                llm_client=llm_client,
                embedding_model=shared_emb,
                config=app_config,
                retrieval_method=eff_retrieval_method,
            )
            baseline.index_messages(messages)

            for q in questions:
                q_text = str(q.get("text", ""))
                t0 = time.perf_counter()
                initial_calls = llm_client.call_count if llm_client else 0
                ans = baseline.ask(question=q_text, reference_time=now_dt)
                lat = time.perf_counter() - t0
                call_diff = (llm_client.call_count - initial_calls) if llm_client else 0

                b_plan = q.get("plan") if isinstance(q.get("plan"), dict) else {}
                b_valid_at = b_plan.get("valid_at") or q.get("valid_at") or q.get("as_of")
                b_known_at = b_plan.get("known_at") or q.get("known_at")

                ans_dict = {
                    "value": ans.value,
                    "values": ans.values,
                    "answer_text": ans.answer_text,
                    "explanation": ans.explanation,
                    "conflict_flagged": ans.conflict_flagged,
                    "needs_review_flagged": ans.needs_review_flagged,
                    "abstained": ans.abstained,
                    "valid_at": str(b_valid_at) if b_valid_at is not None else now_dt.isoformat(),
                    "known_at": str(b_known_at) if b_known_at is not None else now_dt.isoformat(),
                    "now": now_dt.isoformat(),
                    "cited_memory_ids": ans.cited_memory_ids,
                    "dependents": [],
                    "dependent_records": [],
                    "cited_sources": [],
                    "rules": [],
                    "ingestion_rules": [],
                    "decision_records": [],
                    "active_memory_records": [],
                    "memory_records": [],
                    "repository_retrieval": {
                        "performed": False,
                        "error": "Baseline B0 does not use MemoryMesh repository retrieval",
                    },
                }
                q_res = evaluate_question(
                    scenario_id=sc_id,
                    question=q,
                    answer_dict=ans_dict,
                    is_baseline=True,
                    latency_seconds=lat,
                    llm_calls=call_diff,
                    retrieval_method=baseline.retrieval_method,
                )
                all_q_results.append(q_res)

        else:
            # MemoryMesh (MM or Ablation)
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
                db_path = tf.name

            try:
                service_llm = llm_client if mode == "e2e" else None
                service = MemoryService(db_path, cfg=app_config, llm=service_llm)

                # Ingestion
                if mode in ("structured", "engine-only") or "structured_claims" in sc:
                    for msg in messages:
                        m_id = str(msg.get("id", ""))
                        t_dt = parse_iso_datetime(msg.get("t")) or now_dt
                        im = IngestMessage(
                            source_id=str(msg.get("source", "unknown")),
                            event_time=t_dt,
                            text=str(msg.get("text", "")),
                        )
                        raw_claims = structured_claims.get(m_id, [])
                        claims = [Claim.model_validate(c) for c in raw_claims]
                        report = service.ingest_claims(im, claims)

                        if m_id in expected_decisions:
                            exp_rule = str(expected_decisions[m_id]).strip()
                            actual_rule = (
                                report.decisions[0].rule_id
                                if report.decisions and report.decisions[0].rule_id
                                else "NONE"
                            )
                            all_dec_matches.append(actual_rule == exp_rule)
                else:
                    # e2e natural-language ingestion
                    for msg in messages:
                        m_id = str(msg.get("id", ""))
                        t_dt = parse_iso_datetime(msg.get("t")) or now_dt
                        report = service.ingest(
                            source_id=str(msg.get("source", "unknown")),
                            event_time=t_dt,
                            text=str(msg.get("text", "")),
                        )
                        if m_id in expected_decisions:
                            exp_rule = str(expected_decisions[m_id]).strip()
                            actual_rule = (
                                report.decisions[0].rule_id
                                if report.decisions and report.decisions[0].rule_id
                                else "NONE"
                            )
                            all_dec_matches.append(actual_rule == exp_rule)

                # Collect ingestion-time decision rules for auditability and verification
                dec_logs = service.get_decision_logs()
                ingest_rules = [d.rule_id for d in dec_logs if d.rule_id]
                dec_records = [
                    {
                        "rule_id": str(dl.rule_id),
                        "action": (
                            dl.action.value if hasattr(dl.action, "value") else str(dl.action)
                        ),
                        "claim_value": (
                            str(dl.claim_json.get("value"))
                            if dl.claim_json and dl.claim_json.get("value") is not None
                            else None
                        ),
                        "claim_subject": (
                            str(dl.claim_json.get("subject"))
                            if dl.claim_json and dl.claim_json.get("subject") is not None
                            else None
                        ),
                        "claim_attribute": (
                            str(dl.claim_json.get("attribute"))
                            if dl.claim_json and dl.claim_json.get("attribute") is not None
                            else None
                        ),
                        "explanation": str(dl.explanation or ""),
                    }
                    for dl in dec_logs
                ]

                # Question answering
                for q in questions:
                    q_text = str(q.get("text", ""))
                    plan = q.get("plan") if mode != "e2e" else None
                    det_only = mode in ("structured", "engine-only")

                    t0 = time.perf_counter()
                    initial_calls = llm_client.call_count if (llm_client and mode == "e2e") else 0
                    ans_result = service.ask(
                        question=q_text,
                        plan=plan,
                        reference_time=now_dt,
                        deterministic_only=det_only,
                    )
                    lat = time.perf_counter() - t0
                    call_diff = (
                        (llm_client.call_count - initial_calls)
                        if (llm_client and mode == "e2e")
                        else 0
                    )

                    deps = (
                        [
                            (imp.memory.memory.entity_name, imp.memory.memory.attribute)
                            for imp in ans_result.result.impact
                        ]
                        if ans_result.result.impact
                        else []
                    )
                    dep_records = (
                        [
                            {
                                "id": str(imp.memory.memory.id),
                                "entity": imp.memory.memory.entity_name,
                                "attribute": imp.memory.memory.attribute,
                                "needs_review": bool(imp.memory.memory.needs_review),
                                "review_reason": str(imp.memory.memory.review_reason or ""),
                            }
                            for imp in ans_result.result.impact
                        ]
                        if ans_result.result.impact
                        else []
                    )

                    srcs: list[str] = []
                    for it in ans_result.result.items:
                        srcs.extend(getattr(it, "source_names", []))

                    query_rules = [d.rule_id for d in ans_result.result.decisions if d.rule_id]
                    all_rules = sorted(set(query_rules + ingest_rules))

                    all_mems = service.repo.get_all_memories(include_evidence=False)
                    mem_records = [
                        {
                            "id": str(m.id),
                            "entity": m.entity_name,
                            "attribute": m.attribute,
                            "value": m.value_display or m.value_norm,
                            "status": str(
                                m.status.value if hasattr(m.status, "value") else m.status
                            ),
                            "is_active": (
                                (m.status.value if hasattr(m.status, "value") else m.status)
                                == MemoryStatus.ACTIVE.value
                            ),
                            "context": dict(m.context) if m.context else {},
                            "context_key": m.context_key or "",
                            "valid_from": m.valid_from.isoformat() if m.valid_from else None,
                            "valid_to": m.valid_to.isoformat() if m.valid_to else None,
                            "recorded_at": m.recorded_at.isoformat() if m.recorded_at else None,
                            "closed_at": m.closed_at.isoformat() if m.closed_at else None,
                        }
                        for m in all_mems
                    ]
                    act_mem_records = [r for r in mem_records if r["is_active"]]

                    q_plan_dict = q.get("plan") if isinstance(q.get("plan"), dict) else {}
                    q_ent_name = (
                        ans_result.result.plan.entity
                        if ans_result.result.plan and ans_result.result.plan.entity
                        else q_plan_dict.get("entity")
                    )
                    q_attr_name = (
                        ans_result.result.plan.attribute
                        if ans_result.result.plan and ans_result.result.plan.attribute
                        else q_plan_dict.get("attribute")
                    )
                    q_ctx = (
                        ans_result.result.plan.context
                        if ans_result.result.plan and ans_result.result.plan.context is not None
                        else (q_plan_dict.get("context") or {})
                    )

                    req_valid_at_dt: datetime = (
                        ans_result.result.plan.valid_at
                        if ans_result.result.plan and ans_result.result.plan.valid_at
                        else (
                            parse_iso_datetime(q_plan_dict.get("valid_at"))
                            or parse_iso_datetime(q.get("valid_at"))
                            or parse_iso_datetime(q.get("as_of"))
                            or now_dt
                        )
                    )
                    req_known_at_dt: datetime = (
                        ans_result.result.plan.known_at
                        if ans_result.result.plan and ans_result.result.plan.known_at
                        else (
                            parse_iso_datetime(q_plan_dict.get("known_at"))
                            or parse_iso_datetime(q.get("known_at"))
                            or now_dt
                        )
                    )
                    req_valid_at_str = req_valid_at_dt.isoformat()
                    req_known_at_str = req_known_at_dt.isoformat()

                    retrieval_evidence: dict[str, Any]
                    retrieved_mem_records: list[dict[str, Any]] = []

                    if not q_ent_name or not q_attr_name:
                        retrieval_evidence = {
                            "performed": False,
                            "entity": q_ent_name,
                            "attribute": q_attr_name,
                            "context": q_ctx,
                            "valid_at": req_valid_at_str,
                            "known_at": req_known_at_str,
                            "retrieved_memories": [],
                            "slot_memories": [],
                            "error": "Query plan lacks entity or attribute for retrieval evidence",
                        }
                    else:
                        ent_record = service.repo.get_entity_by_name(q_ent_name)
                        if ent_record is None:
                            ent_record = service.repo.get_entity_by_alias(
                                normalize_entity_name(q_ent_name)
                            )

                        if ent_record is None:
                            retrieval_evidence = {
                                "performed": False,
                                "entity": q_ent_name,
                                "attribute": q_attr_name,
                                "context": q_ctx,
                                "valid_at": req_valid_at_str,
                                "known_at": req_known_at_str,
                                "retrieved_memories": [],
                                "slot_memories": [],
                                "error": f"Entity '{q_ent_name}' not found in repository",
                            }
                        else:
                            ent_id = int(ent_record["id"])
                            # Actual repository retrieval path
                            retrieved_mem_objs = retrieve_valid_memories(
                                repo=service.repo,
                                entity_id=ent_id,
                                attribute=q_attr_name,
                                valid_at=req_valid_at_dt,
                                known_at=req_known_at_dt,
                                context=q_ctx,
                                include_rejected=False,
                            )
                            # All slot candidate memories known as of known_at
                            slot_cand_objs = service.repo.find_memories(
                                entity_id=ent_id,
                                attribute=q_attr_name,
                                valid_at=None,
                                known_at=req_known_at_dt,
                                include_rejected=True,
                            )

                            retrieved_mem_records = [
                                {
                                    "id": str(m.id),
                                    "entity": m.entity_name or q_ent_name,
                                    "attribute": m.attribute,
                                    "value": m.value_display or m.value_norm,
                                    "status": str(
                                        m.status.value if hasattr(m.status, "value") else m.status
                                    ),
                                    "is_active": (
                                        str(getattr(m.status, "value", m.status)).lower()
                                        == "active"
                                    ),
                                    "context": dict(m.context) if m.context else {},
                                    "context_key": m.context_key or "",
                                    "valid_from": (
                                        m.valid_from.isoformat() if m.valid_from else None
                                    ),
                                    "valid_to": m.valid_to.isoformat() if m.valid_to else None,
                                    "recorded_at": (
                                        m.recorded_at.isoformat() if m.recorded_at else None
                                    ),
                                    "closed_at": (m.closed_at.isoformat() if m.closed_at else None),
                                }
                                for m in retrieved_mem_objs
                            ]

                            retrieved_id_set = {str(m.id) for m in retrieved_mem_objs}
                            slot_mem_records = [
                                {
                                    "id": str(m.id),
                                    "entity": m.entity_name or q_ent_name,
                                    "attribute": m.attribute,
                                    "value": m.value_display or m.value_norm,
                                    "status": str(
                                        m.status.value if hasattr(m.status, "value") else m.status
                                    ),
                                    "is_active": (
                                        str(m.id) in retrieved_id_set
                                        and str(getattr(m.status, "value", m.status)).lower()
                                        == "active"
                                    ),
                                    "context": dict(m.context) if m.context else {},
                                    "context_key": m.context_key or "",
                                    "valid_from": (
                                        m.valid_from.isoformat() if m.valid_from else None
                                    ),
                                    "valid_to": m.valid_to.isoformat() if m.valid_to else None,
                                    "recorded_at": (
                                        m.recorded_at.isoformat() if m.recorded_at else None
                                    ),
                                    "closed_at": (m.closed_at.isoformat() if m.closed_at else None),
                                }
                                for m in slot_cand_objs
                            ]

                            retrieval_evidence = {
                                "performed": True,
                                "entity": q_ent_name,
                                "attribute": q_attr_name,
                                "context": q_ctx,
                                "valid_at": req_valid_at_str,
                                "known_at": req_known_at_str,
                                "retrieved_memories": retrieved_mem_records,
                                "slot_memories": slot_mem_records,
                                "error": None,
                            }

                    ans_dict = {
                        "value": ans_result.answer.value,
                        "values": ans_result.answer.values,
                        "answer_text": ans_result.answer.answer_text,
                        "explanation": ans_result.answer.explanation,
                        "conflict_flagged": ans_result.answer.conflict_flagged,
                        "needs_review_flagged": ans_result.answer.needs_review_flagged,
                        "abstained": ans_result.answer.abstained,
                        "valid_at": req_valid_at_str,
                        "known_at": req_known_at_str,
                        "now": now_dt.isoformat(),
                        "cited_memory_ids": ans_result.answer.cited_memory_ids,
                        "dependents": deps,
                        "dependent_records": dep_records,
                        "cited_sources": srcs,
                        "rules": all_rules,
                        "ingestion_rules": ingest_rules,
                        "decision_records": dec_records,
                        "active_memory_records": (
                            retrieved_mem_records
                            if retrieval_evidence.get("performed")
                            else act_mem_records
                        ),
                        "memory_records": mem_records,
                        "repository_retrieval": retrieval_evidence,
                    }

                    q_res = evaluate_question(
                        scenario_id=sc_id,
                        question=q,
                        answer_dict=ans_dict,
                        is_baseline=False,
                        latency_seconds=lat,
                        llm_calls=call_diff,
                    )
                    all_q_results.append(q_res)

            finally:
                if os.path.exists(db_path):
                    try:
                        os.remove(db_path)
                    except Exception:
                        pass

    metrics = compute_metrics(
        system_id=system_id,
        system_name=system_name,
        question_results=all_q_results,
        decision_matches=all_dec_matches,
        total_messages=total_messages,
        retrieval_method=eff_retrieval_method,
    )
    return metrics, all_q_results


def build_ablation_config(base_cfg: AppConfig, ablation_key: str) -> AppConfig:
    """Clone application config and flip a single ablation flag to False."""
    cfg = copy.deepcopy(base_cfg)
    if hasattr(cfg.engine.ablation, ablation_key):
        setattr(cfg.engine.ablation, ablation_key, False)
    return cfg


def execute_evaluation_pass(
    mode: str,
    scenarios: list[dict[str, Any]],
    base_config: AppConfig,
    retrieval_method: str = "fastembed",
    with_ablations: bool = False,
    shared_emb: Any | None = None,
) -> tuple[
    dict[str, EvaluationMetrics],
    dict[str, list[QuestionEvalResult]],
    list[dict[str, Any]],
    dict[str, ModeExecutionStatus],
]:
    """Execute a complete evaluation pass across configured modes."""
    metrics_by_system: dict[str, EvaluationMetrics] = {}
    traces_by_system: dict[str, list[QuestionEvalResult]] = {}
    mode_statuses: dict[str, ModeExecutionStatus] = {
        m: ModeExecutionStatus.NOT_EXECUTED for m in CANONICAL_MODES
    }

    if mode == "structured":
        try:
            m, traces = run_system_on_scenarios(
                "MM",
                "MemoryMesh (structured)",
                scenarios,
                base_config,
                mode="structured",
                embedding_model=shared_emb,
            )
            metrics_by_system["MM (structured)"] = m
            traces_by_system["MM (structured)"] = traces
            mode_statuses["structured"] = ModeExecutionStatus.COMPLETED
        except Exception as e:
            logger.error("Structured mode execution failed: %s", e)
            mode_statuses["structured"] = ModeExecutionStatus.FAILED

    elif mode == "engine-only":
        try:
            m, traces = run_system_on_scenarios(
                "MM",
                "Engine-only",
                scenarios,
                base_config,
                mode="engine-only",
                embedding_model=shared_emb,
            )
            metrics_by_system["MM (engine-only)"] = m
            traces_by_system["MM (engine-only)"] = traces
            mode_statuses["engine-only"] = ModeExecutionStatus.COMPLETED
        except Exception as e:
            logger.error("Engine-only mode execution failed: %s", e)
            mode_statuses["engine-only"] = ModeExecutionStatus.FAILED

    elif mode == "e2e":
        try:
            m, traces = run_system_on_scenarios(
                "MM",
                "MemoryMesh (e2e)",
                scenarios,
                base_config,
                mode="e2e",
                embedding_model=shared_emb,
            )
            metrics_by_system["MM (e2e)"] = m
            traces_by_system["MM (e2e)"] = traces
            mode_statuses["e2e"] = ModeExecutionStatus.COMPLETED
        except Exception as e:
            logger.error("E2E mode execution failed: %s", e)
            mode_statuses["e2e"] = ModeExecutionStatus.FAILED

    elif mode == "baseline":
        try:
            m, traces = run_system_on_scenarios(
                "B0",
                "Naive RAG",
                scenarios,
                base_config,
                mode="baseline",
                retrieval_method=retrieval_method,
                embedding_model=shared_emb,
            )
            metrics_by_system["B0 (naive RAG)"] = m
            traces_by_system["B0 (naive RAG)"] = traces
            mode_statuses["baseline"] = ModeExecutionStatus.COMPLETED
        except Exception as e:
            logger.error("Baseline mode execution failed: %s", e)
            mode_statuses["baseline"] = ModeExecutionStatus.FAILED

    elif mode == "ablation":
        systems_to_run = [
            ("MM (full)", "full engine", base_config),
            (
                "A1 -temporal",
                "without temporal",
                build_ablation_config(base_config, "use_temporal"),
            ),
            ("A2 -trust", "without trust", build_ablation_config(base_config, "use_trust")),
            (
                "A3 -card.",
                "without cardinality",
                build_ablation_config(base_config, "use_cardinality"),
            ),
            ("A4 -context", "without context", build_ablation_config(base_config, "use_context")),
            (
                "A5 -propag.",
                "without propagation",
                build_ablation_config(base_config, "use_propagation"),
            ),
        ]
        for sys_id, sys_name, cfg in systems_to_run:
            m, traces = run_system_on_scenarios(
                sys_id,
                sys_name,
                scenarios,
                cfg,
                mode="structured",
                embedding_model=shared_emb,
            )
            metrics_by_system[sys_id] = m
            traces_by_system[sys_id] = traces
        mode_statuses["structured"] = ModeExecutionStatus.COMPLETED

    elif mode == "all":
        # 1. Baseline B0
        logger.info("Evaluating B0 Naive RAG baseline...")
        try:
            m_b0, traces_b0 = run_system_on_scenarios(
                "B0",
                "Naive RAG",
                scenarios,
                base_config,
                mode="baseline",
                retrieval_method=retrieval_method,
                embedding_model=shared_emb,
            )
            metrics_by_system["B0 (naive RAG)"] = m_b0
            traces_by_system["B0 (naive RAG)"] = traces_b0
            mode_statuses["baseline"] = ModeExecutionStatus.COMPLETED
        except Exception as e:
            logger.error("Baseline execution failed: %s", e)
            mode_statuses["baseline"] = ModeExecutionStatus.FAILED

        # 2. Structured MM
        logger.info("Evaluating MemoryMesh structured mode...")
        try:
            m_struct, traces_struct = run_system_on_scenarios(
                "MM",
                "MemoryMesh (structured)",
                scenarios,
                base_config,
                mode="structured",
                embedding_model=shared_emb,
            )
            metrics_by_system["MM (structured)"] = m_struct
            traces_by_system["MM (structured)"] = traces_struct
            mode_statuses["structured"] = ModeExecutionStatus.COMPLETED
        except Exception as e:
            logger.error("Structured mode execution failed: %s", e)
            mode_statuses["structured"] = ModeExecutionStatus.FAILED

        # 3. Engine-only MM
        logger.info("Evaluating MemoryMesh engine-only mode...")
        try:
            m_eng, traces_eng = run_system_on_scenarios(
                "MM",
                "Engine-only",
                scenarios,
                base_config,
                mode="engine-only",
                embedding_model=shared_emb,
            )
            metrics_by_system["MM (engine-only)"] = m_eng
            traces_by_system["MM (engine-only)"] = traces_eng
            mode_statuses["engine-only"] = ModeExecutionStatus.COMPLETED
        except Exception as e:
            logger.error("Engine-only mode execution failed: %s", e)
            mode_statuses["engine-only"] = ModeExecutionStatus.FAILED

        # 4. E2E MM
        logger.info("Evaluating MemoryMesh E2E mode...")
        try:
            m_e2e, traces_e2e = run_system_on_scenarios(
                "MM",
                "MemoryMesh (e2e)",
                scenarios,
                base_config,
                mode="e2e",
                embedding_model=shared_emb,
            )
            metrics_by_system["MM (e2e)"] = m_e2e
            traces_by_system["MM (e2e)"] = traces_e2e
            mode_statuses["e2e"] = ModeExecutionStatus.COMPLETED
        except Exception as e:
            logger.error("E2E mode execution failed: %s", e)
            mode_statuses["e2e"] = ModeExecutionStatus.FAILED

        # 5. Optional Ablations
        if with_ablations:
            ablations_to_run = [
                (
                    "A1 -temporal",
                    "without temporal",
                    build_ablation_config(base_config, "use_temporal"),
                ),
                ("A2 -trust", "without trust", build_ablation_config(base_config, "use_trust")),
                (
                    "A3 -card.",
                    "without cardinality",
                    build_ablation_config(base_config, "use_cardinality"),
                ),
                (
                    "A4 -context",
                    "without context",
                    build_ablation_config(base_config, "use_context"),
                ),
                (
                    "A5 -propag.",
                    "without propagation",
                    build_ablation_config(base_config, "use_propagation"),
                ),
            ]
            for sys_id, sys_name, cfg in ablations_to_run:
                m, traces = run_system_on_scenarios(
                    sys_id,
                    sys_name,
                    scenarios,
                    cfg,
                    mode="structured",
                    embedding_model=shared_emb,
                )
                metrics_by_system[sys_id] = m
                traces_by_system[sys_id] = traces

    failures_summary: list[dict[str, Any]] = []
    for sys_name, traces in traces_by_system.items():
        for t in traces:
            reasons = []
            if t.is_current_correct is False:
                reasons.append("Incorrect current value")
            if t.is_as_of_correct is False:
                reasons.append("Incorrect as-of value")
            if t.is_history_correct is False:
                reasons.append("Incorrect history sequence")
            if t.is_stale is True:
                reasons.append("Stale / forbidden value present")
            if t.abstention_correct is False:
                reasons.append("Abstention mismatch")
            if t.provenance_correct is False:
                reasons.append("Provenance mismatch")
            if t.impact_recall is not None and t.impact_recall < 1.0:
                reasons.append("Impact recall incomplete")

            if reasons:
                failures_summary.append(
                    {
                        "system": sys_name,
                        "scenario_id": t.scenario_id,
                        "question_id": t.question_id,
                        "reasons": reasons,
                        "expected": t.expected,
                        "actual": t.actual,
                    }
                )

    return metrics_by_system, traces_by_system, failures_summary, mode_statuses


def finalize_evaluation_artifacts(
    *,
    report_file: Path,
    out_file: Path,
    provisional_gate_result: GateEvaluationResult,
    metrics_by_system: dict[str, EvaluationMetrics],
    traces_by_system: dict[str, list[QuestionEvalResult]],
    scenarios: list[dict[str, Any]],
    timestamp_str: str,
    mode_statuses: dict[str, ModeExecutionStatus],
    results_payload: dict[str, Any],
) -> tuple[GateEvaluationResult, bool]:
    """Execute authoritative 7-step artifact finalization and read-back verification sequence.

    Returns:
        tuple of (final_gate_result: GateEvaluationResult, verification_succeeded: bool)
    """
    # Step 2: Generate and persist both initial artifacts to disk
    report_content = generate_evaluation_report(
        metrics_by_system=metrics_by_system,
        traces_by_system=traces_by_system,
        scenarios=scenarios,
        gate_result=provisional_gate_result,
        timestamp=timestamp_str,
        mode_execution_statuses=mode_statuses,
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Initial evaluation report generated: %s", report_file)

    results_payload["g2_gate"] = provisional_gate_result.model_dump()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    logger.info("Initial evaluation results JSON saved: %s", out_file)

    # Step 3: Read back persisted artifacts and validate them against provisional result
    is_valid_artifacts, art_evidence, art_failures = verify_persisted_artifacts(
        report_path=report_file,
        results_path=out_file,
        expected_gate_result=provisional_gate_result,
        ignore_g2_1_status=True,
        require_exact_evidence=True,
    )

    # Step 4: Update G2-1 with the empirical read-back result
    if is_valid_artifacts:
        g2_1_status = GateStatus.PASS
        g2_1_evidence = (
            f"Artifacts verified on disk and read back successfully: {report_file.name} "
            f"({report_file.stat().st_size} bytes, Section H verified) and "
            f"{out_file.name} ({out_file.stat().st_size} bytes, all criteria match)."
        )
    else:
        g2_1_status = GateStatus.FAIL
        g2_1_evidence = f"Artifact read-back verification failed: {'; '.join(art_failures)}"

    final_criteria: list[GateCriterionResult] = []
    for c in provisional_gate_result.criteria:
        if c.criterion_id == "G2-1":
            final_criteria.append(
                GateCriterionResult(
                    criterion_id="G2-1",
                    title="Artifact Generation",
                    target_rule="Generate report.md and results.json; complete G2 gate.",
                    status=g2_1_status,
                    evidence=g2_1_evidence,
                    metrics={"report_path": str(report_file), "results_path": str(out_file)},
                )
            )
        else:
            final_criteria.append(c)

    if any(c.status == GateStatus.FAIL for c in final_criteria):
        final_overall_status = GateStatus.FAIL
        final_summary = "Gate G2 FAILED: One or more acceptance criteria failed."
    elif any(c.status == GateStatus.INCONCLUSIVE for c in final_criteria):
        final_overall_status = GateStatus.INCONCLUSIVE
        final_summary = "Gate G2 INCONCLUSIVE: Evaluation data incomplete to confirm all criteria."
    else:
        final_overall_status = GateStatus.PASS
        final_summary = (
            "Gate G2 PASSED: All 7 acceptance criteria satisfied with empirical evidence."
        )

    final_gate_result = GateEvaluationResult(
        gate_name="G2",
        timestamp=timestamp_str,
        overall_status=final_overall_status,
        criteria=final_criteria,
        summary=final_summary,
    )

    # Step 5: Persist both artifacts once more with the finalized G2-1 gate result
    results_payload["g2_gate"] = final_gate_result.model_dump()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    final_report_content = generate_evaluation_report(
        metrics_by_system=metrics_by_system,
        traces_by_system=traces_by_system,
        scenarios=scenarios,
        gate_result=final_gate_result,
        timestamp=timestamp_str,
        mode_execution_statuses=mode_statuses,
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(final_report_content)

    # Step 6: Authoritative read-back verification ensuring disk strictly matches final_gate_result
    final_ok, final_ev, final_errs = verify_persisted_artifacts(
        report_path=report_file,
        results_path=out_file,
        expected_gate_result=final_gate_result,
        ignore_g2_1_status=False,
        require_exact_evidence=True,
    )

    if not final_ok:
        logger.error(
            "Final artifact read-back consistency check failed: %s; forcing G2-1 and gate to FAIL",
            final_errs,
        )
        corrected_criteria: list[GateCriterionResult] = []
        for c in final_gate_result.criteria:
            if c.criterion_id == "G2-1":
                corrected_criteria.append(
                    GateCriterionResult(
                        criterion_id="G2-1",
                        title="Artifact Generation",
                        target_rule="Generate report.md and results.json; complete G2 gate.",
                        status=GateStatus.FAIL,
                        evidence=(
                            f"Final artifact read-back verification failed: {'; '.join(final_errs)}"
                        ),
                        metrics={"report_path": str(report_file), "results_path": str(out_file)},
                    )
                )
            else:
                corrected_criteria.append(c)

        final_gate_result = GateEvaluationResult(
            gate_name="G2",
            timestamp=timestamp_str,
            overall_status=GateStatus.FAIL,
            criteria=corrected_criteria,
            summary="Gate G2 FAILED: Final artifact verification failed.",
        )

        # Write corrected gate result back to disk
        results_payload["g2_gate"] = final_gate_result.model_dump()
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)

        corrected_report_content = generate_evaluation_report(
            metrics_by_system=metrics_by_system,
            traces_by_system=traces_by_system,
            scenarios=scenarios,
            gate_result=final_gate_result,
            timestamp=timestamp_str,
            mode_execution_statuses=mode_statuses,
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(corrected_report_content)

        # Subsequent deterministic verification of the corrected failure artifacts
        subsequent_ok, subsequent_ev, subsequent_errs = verify_persisted_artifacts(
            report_path=report_file,
            results_path=out_file,
            expected_gate_result=final_gate_result,
            ignore_g2_1_status=False,
            require_exact_evidence=True,
        )
        if not subsequent_ok:
            logger.error(
                "Subsequent read-back verification of corrected artifacts failed: %s",
                subsequent_errs,
            )
            print(
                f"\n[ARTIFACT VERIFICATION FAILURE] Final persisted artifacts could not be "
                f"verified on disk: {'; '.join(subsequent_errs)}"
            )
        else:
            logger.info(
                "Subsequent read-back successfully verified corrected failure artifacts on disk."
            )

        return final_gate_result, False

    logger.info("Final artifact read-back consistency check verified 100%% disk-to-memory match.")
    return final_gate_result, True


def main() -> int:
    parser = argparse.ArgumentParser(description="MemoryMesh Evaluation Runner (P8.1, P8.4)")
    parser.add_argument(
        "--mode",
        choices=["structured", "e2e", "baseline", "engine-only", "ablation", "all"],
        default="all",
        help="Evaluation execution mode (default: all)",
    )
    parser.add_argument(
        "--scenarios-dir",
        default="eval/scenarios",
        help="Directory holding scenario YAML files (default: eval/scenarios)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Specific scenario ID or file to run (optional)",
    )
    parser.add_argument(
        "--output",
        default="eval/results.json",
        help="Path to write raw structured evaluation results JSON (default: eval/results.json)",
    )
    parser.add_argument(
        "--report-output",
        default="eval/report.md",
        help="Path to write formatted Markdown evaluation report (default: eval/report.md)",
    )
    parser.add_argument(
        "--retrieval-method",
        choices=["fastembed", "lexical"],
        default="fastembed",
        help="Retrieval method for baseline evaluation (default: fastembed)",
    )
    parser.add_argument(
        "--with-ablations",
        action="store_true",
        help="Also execute ablations A1-A5 when running --mode all",
    )
    parser.add_argument(
        "--verify-reproducibility",
        action="store_true",
        default=True,
        help="Execute an independent second evaluation run to verify G2-7 reproducibility",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs",
    )
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    base_config = load_app_config()
    scenarios_dir = Path(args.scenarios_dir)
    scenarios = load_scenarios(scenarios_dir, args.scenario)

    if not scenarios:
        logger.error("No valid evaluation scenarios found in %s", scenarios_dir)
        return 1

    timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_file = Path(args.output)
    report_file = Path(args.report_output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize shared embedding model for fast baseline indexing across scenarios
    shared_emb = None
    if args.retrieval_method == "fastembed":
        try:
            from fastembed import TextEmbedding

            shared_emb = TextEmbedding()
        except Exception as e:
            logger.warning("Could not pre-initialize FastEmbed TextEmbedding: %s", e)

    logger.info(
        "Running MemoryMesh evaluation [Run 1] in mode '%s' across %d scenarios",
        args.mode,
        len(scenarios),
    )

    (
        metrics_by_system,
        traces_by_system,
        failures_summary,
        mode_statuses,
    ) = execute_evaluation_pass(
        mode=args.mode,
        scenarios=scenarios,
        base_config=base_config,
        retrieval_method=args.retrieval_method,
        with_ablations=args.with_ablations,
        shared_emb=shared_emb,
    )

    # Print summary table for Run 1
    ascii_table = format_ascii_table(metrics_by_system)
    print("\n" + ascii_table)

    sc_ids = [str(s.get("id")) for s in scenarios]
    cats = {str(s.get("category")) for s in scenarios if s.get("category")}
    total_q = sum(len(s.get("questions", [])) for s in scenarios)

    # Optional / Default Dual-Run Reproducibility Pass for G2-7
    reproducibility_run = None
    if args.verify_reproducibility:
        logger.info("Executing independent [Run 2] to verify G2-7 deterministic reproducibility...")
        metrics_2, traces_2, failures_2, _ = execute_evaluation_pass(
            mode=args.mode,
            scenarios=scenarios,
            base_config=base_config,
            retrieval_method=args.retrieval_method,
            with_ablations=args.with_ablations,
            shared_emb=shared_emb,
        )
        reproducibility_run = normalize_run_result(
            metrics_by_system=metrics_2,
            scenario_ids=sc_ids,
            retrieval_method=args.retrieval_method,
            failures=failures_2,
            traces_by_system=traces_2,
        )

    # Extract designed-to-lose accounting records for JSON artifact
    # Explicitly select MemoryMesh structured evaluation traces for accounting
    mm_traces_dtl = []
    for k in traces_by_system:
        if "struct" in k.lower():
            mm_traces_dtl = traces_by_system[k]
            break
    if not mm_traces_dtl:
        for k in traces_by_system:
            if "b0" not in k.lower() and "baseline" not in k.lower():
                mm_traces_dtl = traces_by_system[k]
                break

    dtl_accounting: list[dict[str, Any]] = []
    for sc in scenarios:
        sid = str(sc.get("id", ""))
        if sid.startswith("lose-"):
            sc_traces = [t for t in mm_traces_dtl if t.scenario_id == sid]
            target_trace, trace_err = select_dtl_invariant_trace(sid, sc, sc_traces)
            if target_trace is None:
                is_ok = False
                outcome_str = trace_err or "No execution traces recorded"
                status_str = "**`UNEXECUTED`**"
                ans = {}
            else:
                is_ok, outcome_str, status_str = verify_designed_to_lose_outcome(
                    sid, sc, target_trace
                )
                ans = target_trace.actual if isinstance(target_trace.actual, dict) else {}
            dtl_accounting.append(
                {
                    "scenario_id": sid,
                    "description": sc.get("description", ""),
                    "executed": len(sc_traces) > 0,
                    "verified": is_ok,
                    "observed_outcome": outcome_str,
                    "accounting_status": status_str,
                    "valid_at": ans.get("valid_at"),
                    "known_at": ans.get("known_at"),
                    "now": ans.get("now"),
                    "abstained": ans.get("abstained", False),
                    "conflict_flagged": ans.get("conflict_flagged", False),
                    "needs_review_flagged": ans.get("needs_review_flagged", False),
                    "observed_value": ans.get("value"),
                    "rules": ans.get("rules", []),
                    "dependents": ans.get("dependents", []),
                    "decision_records": ans.get("decision_records", []),
                    "dependent_records": ans.get("dependent_records", []),
                    "active_memory_records": ans.get("active_memory_records", []),
                    "memory_records": ans.get("memory_records", []),
                }
            )

    # =========================================================================
    # G2 Evaluation Gate Finalization Sequence (7-Step Read-Back Verification)
    # 1. Evaluate G2-2 through G2-7 (creating provisional gate result).
    # 2. Generate and persist both artifacts (report.md and results.json).
    # 3. Read them back and validate them against the provisional result.
    # 4. Update G2-1 with the empirical read-back result.
    # 5. Persist both artifacts once more with finalized G2-1 outcome.
    # 6. Perform a final consistency check without changing criterion results.
    # 7. Complete finalization and output verified gate summary.
    # =========================================================================

    # Step 1: Evaluate provisional gate result (G2-2 through G2-7 evaluated;
    #         G2-1 evaluated provisionally).
    provisional_gate_result = evaluate_g2_gate(
        metrics_by_system=metrics_by_system,
        executed_scenarios=scenarios,
        scenario_ids=sc_ids,
        categories_present=cats,
        total_questions=total_q,
        report_path=report_file,
        results_path=out_file,
        timestamp=timestamp_str,
        traces_by_system=traces_by_system,
        retrieval_method=args.retrieval_method,
        failures=failures_summary,
        reproducibility_run=reproducibility_run,
        mode_execution_statuses=mode_statuses,
    )

    results_payload = {
        "timestamp": timestamp_str,
        "mode": args.mode,
        "mode_execution_statuses": {k: v.value for k, v in mode_statuses.items()},
        "scenario_count": len(scenarios),
        "question_count": total_q,
        "scenarios": sc_ids,
        "categories": {cat: sum(1 for s in scenarios if s.get("category") == cat) for cat in cats},
        "retrieval_method": args.retrieval_method,
        "provider_info": {
            "e2e": "DeterministicEvaluationE2EProvider (Offline Simulation)",
            "baseline": (
                f"DeterministicEvaluationBaselineProvider "
                f"(Offline Simulation with {args.retrieval_method} top-k=5)"
            ),
        },
        "metrics": {k: v.model_dump() for k, v in metrics_by_system.items()},
        "traces": {k: [t.model_dump() for t in v] for k, v in traces_by_system.items()},
        "failures": failures_summary,
        "designed_to_lose_accounting": dtl_accounting,
        "g2_gate": provisional_gate_result.model_dump(),
    }

    final_gate_result, verification_ok = finalize_evaluation_artifacts(
        report_file=report_file,
        out_file=out_file,
        provisional_gate_result=provisional_gate_result,
        metrics_by_system=metrics_by_system,
        traces_by_system=traces_by_system,
        scenarios=scenarios,
        timestamp_str=timestamp_str,
        mode_statuses=mode_statuses,
        results_payload=results_payload,
    )

    # Step 7: Summary display
    print("\n=======================================================")
    print(f"G2 EVALUATION GATE STATUS: [{final_gate_result.overall_status.value}]")
    print(f"Summary: {final_gate_result.summary}")
    print("=======================================================")
    for c in final_gate_result.criteria:
        print(f" - [{c.status.value:<4}] {c.criterion_id}: {c.title}")
    print("=======================================================\n")

    if not verification_ok:
        print(
            "[ARTIFACT VERIFICATION FAILURE] Final persisted artifacts failed read-back validation."
        )

    return 0 if (verification_ok and final_gate_result.overall_status != GateStatus.FAIL) else 1


if __name__ == "__main__":
    sys.exit(main())
