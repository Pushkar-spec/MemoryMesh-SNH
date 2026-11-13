"""MemoryMesh Evaluation Runner (P8.1).

Governed by:
- docs/EVALUATION_AND_DEMO.md
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P8.1)

Supported execution modes:
- structured: deterministic engine evaluation with structured claims (zero LLM calls)
- e2e: full natural-language ingestion, query parsing, and answer composition
- baseline: naive RAG baseline (B0) using FastEmbed and baseline_answer prompt
- engine-only: MemoryMesh with deterministic answer composition
- ablation: runs MM (full) and feature ablations A1-A5
- all: runs baseline, full MM, and all ablations A1-A5
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
from eval.metrics import (  # noqa: E402
    EvaluationMetrics,
    QuestionEvalResult,
    compute_metrics,
    evaluate_question,
    format_ascii_table,
)
from memorymesh.config import AppConfig, load_app_config  # noqa: E402
from memorymesh.llm.client import LLMClient  # noqa: E402
from memorymesh.models import Claim, IngestMessage  # noqa: E402
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
            logger.error("Failed to load scenario %s: %s", file_path, e)

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
) -> tuple[EvaluationMetrics, list[QuestionEvalResult]]:
    """Run an evaluation run for a specified system across all loaded scenarios."""
    all_q_results: list[QuestionEvalResult] = []
    all_dec_matches: list[bool] = []
    total_messages = 0

    is_baseline = system_id.upper() in ("B0", "B0-LEXICAL")
    eff_retrieval_method = retrieval_method or ("fastembed" if is_baseline else "memorymesh")

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

                ans_dict = {
                    "value": ans.value,
                    "values": ans.values,
                    "answer_text": ans.answer_text,
                    "explanation": ans.explanation,
                    "conflict_flagged": ans.conflict_flagged,
                    "needs_review_flagged": ans.needs_review_flagged,
                    "abstained": ans.abstained,
                    "cited_memory_ids": ans.cited_memory_ids,
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
                if mode == "structured" or mode == "engine-only" or "structured_claims" in sc:
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

                    srcs: list[str] = []
                    for it in ans_result.result.items:
                        srcs.extend(getattr(it, "source_names", []))

                    ans_dict = {
                        "value": ans_result.answer.value,
                        "values": ans_result.answer.values,
                        "answer_text": ans_result.answer.answer_text,
                        "explanation": ans_result.answer.explanation,
                        "conflict_flagged": ans_result.answer.conflict_flagged,
                        "needs_review_flagged": ans_result.answer.needs_review_flagged,
                        "abstained": ans_result.answer.abstained,
                        "cited_memory_ids": ans_result.answer.cited_memory_ids,
                        "dependents": deps,
                        "cited_sources": srcs,
                        "rules": [d.rule_id for d in ans_result.result.decisions if d.rule_id],
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


def main() -> int:
    parser = argparse.ArgumentParser(description="MemoryMesh Evaluation Runner (P8.1)")
    parser.add_argument(
        "--mode",
        choices=["structured", "e2e", "baseline", "engine-only", "ablation", "all"],
        default="structured",
        help="Evaluation execution mode (default: structured)",
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
    out_file.parent.mkdir(parents=True, exist_ok=True)

    metrics_by_system: dict[str, EvaluationMetrics] = {}
    traces_by_system: dict[str, list[dict[str, Any]]] = {}

    llm_client: LLMClient | None = None
    if args.mode in ("baseline", "e2e", "all"):
        llm_client = LLMClient(config=base_config.engine.llm)

    logger.info(
        "Running MemoryMesh evaluation in mode: %s (%d scenarios)", args.mode, len(scenarios)
    )

    if args.mode == "structured":
        m, traces = run_system_on_scenarios(
            "MM", "MemoryMesh (structured)", scenarios, base_config, mode="structured"
        )
        metrics_by_system["MM (full)"] = m
        traces_by_system["MM (full)"] = [t.model_dump() for t in traces]

    elif args.mode == "engine-only":
        m, traces = run_system_on_scenarios(
            "MM", "Engine-only", scenarios, base_config, mode="engine-only"
        )
        metrics_by_system["MM (engine-only)"] = m
        traces_by_system["MM (engine-only)"] = [t.model_dump() for t in traces]

    elif args.mode == "e2e":
        m, traces = run_system_on_scenarios(
            "MM", "MemoryMesh (e2e)", scenarios, base_config, mode="e2e", llm_client=llm_client
        )
        metrics_by_system["MM (full)"] = m
        traces_by_system["MM (full)"] = [t.model_dump() for t in traces]

    elif args.mode == "baseline":
        m, traces = run_system_on_scenarios(
            "B0", "Naive RAG", scenarios, base_config, mode="baseline", llm_client=llm_client
        )
        metrics_by_system["B0 (naive RAG)"] = m
        traces_by_system["B0 (naive RAG)"] = [t.model_dump() for t in traces]

    elif args.mode == "ablation":
        # Run full MM and all 5 ablations
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
            m, traces = run_system_on_scenarios(sys_id, sys_name, scenarios, cfg, mode="structured")
            metrics_by_system[sys_id] = m
            traces_by_system[sys_id] = [t.model_dump() for t in traces]

    elif args.mode == "all":
        # 1. Baseline B0
        m_b0, traces_b0 = run_system_on_scenarios(
            "B0", "Naive RAG", scenarios, base_config, mode="baseline", llm_client=llm_client
        )
        metrics_by_system["B0 (baseline)"] = m_b0
        traces_by_system["B0 (baseline)"] = [t.model_dump() for t in traces_b0]

        # 2. Structured MM
        m_struct, traces_struct = run_system_on_scenarios(
            "MM", "structured", scenarios, base_config, mode="structured"
        )
        metrics_by_system["MM (structured)"] = m_struct
        traces_by_system["MM (structured)"] = [t.model_dump() for t in traces_struct]

        # 3. Engine-only MM
        m_eng, traces_eng = run_system_on_scenarios(
            "MM", "engine-only", scenarios, base_config, mode="engine-only"
        )
        metrics_by_system["MM (engine-only)"] = m_eng
        traces_by_system["MM (engine-only)"] = [t.model_dump() for t in traces_eng]

        # 4. E2E MM
        m_e2e, traces_e2e = run_system_on_scenarios(
            "MM", "e2e", scenarios, base_config, mode="e2e", llm_client=llm_client
        )
        metrics_by_system["MM (e2e)"] = m_e2e
        traces_by_system["MM (e2e)"] = [t.model_dump() for t in traces_e2e]

        # 5. Ablations A1-A5
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
            ("A4 -context", "without context", build_ablation_config(base_config, "use_context")),
            (
                "A5 -propag.",
                "without propagation",
                build_ablation_config(base_config, "use_propagation"),
            ),
        ]
        for sys_id, sys_name, cfg in ablations_to_run:
            m, traces = run_system_on_scenarios(sys_id, sys_name, scenarios, cfg, mode="structured")
            metrics_by_system[sys_id] = m
            traces_by_system[sys_id] = [t.model_dump() for t in traces]

    # Print summary table
    ascii_table = format_ascii_table(metrics_by_system)
    print("\n" + ascii_table)

    # Save raw structured results JSON for P8.1
    results_payload = {
        "timestamp": timestamp_str,
        "mode": args.mode,
        "scenarios": [s.get("id") for s in scenarios],
        "metrics": {k: v.model_dump() for k, v in metrics_by_system.items()},
        "traces": traces_by_system,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    logger.info("Structured evaluation results saved to: %s", out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
