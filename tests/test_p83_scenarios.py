"""Tests for P8.3 Evaluation Scenarios (Context, Ordering, Dependencies and Time).

Governed by:
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P8.3)
- docs/EVALUATION_AND_DEMO.md
- docs/SPECIFICATION.md

Covers 26 reproducible scenarios across five categories:
A. Context isolation and context switching (context-001 to context-006)
B. Out-of-order claim arrival (ordering-001 to ordering-005)
C. Dependency chains and propagation (dependency-001 to dependency-005)
D. Historical and time-travel queries (temporal-001 to temporal-005)
E. Designed-to-lose scenarios (lose-001 to lose-005)

Supported evaluation modes:
1. Structured (pure engine execution with structured claims, 0 LLM calls)
2. Engine-only (MemoryMesh with deterministic answer composition)
3. End-to-end (E2E with deterministic simulation provider)
4. Naive RAG baseline (B0 with FastEmbed top-k=5 retrieval)
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from eval.run_eval import load_scenarios, run_system_on_scenarios
from memorymesh.config import load_app_config
from memorymesh.llm.client import LLMClient
from memorymesh.llm.providers import LLMProvider
from memorymesh.models import Action, Claim, EdgeType, IngestMessage
from memorymesh.service import MemoryService

SCENARIOS_DIR = Path("eval/scenarios")

P83_CATEGORIES = {
    "context_isolation",
    "out_of_order",
    "dependency_propagation",
    "historical_time_travel",
    "designed_to_lose",
}


def test_load_all_p83_scenarios_schema_validation() -> None:
    """Verifies all 26 P8.3 scenarios load and conform to the scenario schema."""
    scenarios = load_scenarios(SCENARIOS_DIR)
    p83_scenarios = [s for s in scenarios if s.get("category") in P83_CATEGORIES]
    assert len(p83_scenarios) == 26, f"Expected 26 P8.3 scenarios, found {len(p83_scenarios)}"

    categories = {s.get("category") for s in p83_scenarios}
    assert categories == P83_CATEGORIES

    for sc in p83_scenarios:
        sc_id = sc.get("id")
        assert sc_id, "Scenario missing 'id'"
        assert "description" in sc, f"Scenario {sc_id} missing 'description'"
        assert "now" in sc, f"Scenario {sc_id} missing 'now'"
        assert "messages" in sc and len(sc["messages"]) >= 1, f"Scenario {sc_id} missing 'messages'"
        assert "questions" in sc and len(sc["questions"]) >= 1, (
            f"Scenario {sc_id} missing 'questions'"
        )

        # Check messages schema
        for m in sc["messages"]:
            assert "id" in m, f"Scenario {sc_id} message missing 'id'"
            assert "t" in m, f"Scenario {sc_id} message missing 't'"
            assert "source" in m, f"Scenario {sc_id} message missing 'source'"
            assert "text" in m, f"Scenario {sc_id} message missing 'text'"

        # Check questions schema
        for q in sc["questions"]:
            assert "id" in q, f"Scenario {sc_id} question missing 'id'"
            assert "type" in q, f"Scenario {sc_id} question missing 'type'"
            assert "text" in q, f"Scenario {sc_id} question missing 'text'"
            assert "expect" in q, f"Scenario {sc_id} question missing 'expect'"


def test_p83_context_isolation_scenarios() -> None:
    """Verifies context_isolation scenarios (context-001 to context-006) in structured mode."""
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    ctx_scenarios = [s for s in all_scenarios if s.get("category") == "context_isolation"]
    assert len(ctx_scenarios) == 6

    metrics, traces = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (structured)",
        scenarios=ctx_scenarios,
        app_config=cfg,
        mode="structured",
    )

    assert metrics.current_accuracy == 1.0
    assert metrics.engine_decision_accuracy == 1.0
    assert metrics.avg_llm_calls_per_question == 0.0


def test_p83_out_of_order_scenarios() -> None:
    """Verifies out_of_order scenarios (ordering-001 to ordering-005) in structured mode."""
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    ooo_scenarios = [s for s in all_scenarios if s.get("category") == "out_of_order"]
    assert len(ooo_scenarios) == 5

    metrics, traces = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (structured)",
        scenarios=ooo_scenarios,
        app_config=cfg,
        mode="structured",
    )

    assert metrics.current_accuracy == 1.0
    assert metrics.as_of_accuracy == 1.0
    assert metrics.history_accuracy == 1.0
    assert metrics.engine_decision_accuracy == 1.0
    assert metrics.avg_llm_calls_per_question == 0.0


def test_p83_dependency_propagation_scenarios() -> None:
    """Verifies dependency_propagation scenarios (dependency-001 to dependency-005)."""
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    dep_scenarios = [s for s in all_scenarios if s.get("category") == "dependency_propagation"]
    assert len(dep_scenarios) == 5

    metrics, traces = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (structured)",
        scenarios=dep_scenarios,
        app_config=cfg,
        mode="structured",
    )

    assert metrics.impact_recall == 1.0
    assert metrics.impact_precision == 1.0
    assert metrics.engine_decision_accuracy == 1.0
    assert metrics.avg_llm_calls_per_question == 0.0

    # Specific verification for dependency-004 cycle propagation without infinite recursion
    dep4_traces = [t for t in traces if t.scenario_id == "dependency-004"]
    assert len(dep4_traces) == 3
    # q1: Server B runs_service -> api-gateway, needs_review_flagged is True
    assert dep4_traces[0].actual["value"] == "api-gateway"
    assert dep4_traces[0].actual["needs_review_flagged"] is True
    # q2: Server A runs_service -> redis-cache, needs_review_flagged is True
    assert dep4_traces[1].actual["value"] == "redis-cache"
    assert dep4_traces[1].actual["needs_review_flagged"] is True
    # q3: Impact of Server B -> Server A runs_service
    assert dep4_traces[2].impact_recall == 1.0
    assert dep4_traces[2].actual["needs_review_flagged"] is True


def test_p83_historical_time_travel_scenarios() -> None:
    """Verifies historical_time_travel scenarios (temporal-001 to temporal-005)."""
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    hist_scenarios = [s for s in all_scenarios if s.get("category") == "historical_time_travel"]
    assert len(hist_scenarios) == 5

    metrics, traces = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (structured)",
        scenarios=hist_scenarios,
        app_config=cfg,
        mode="structured",
    )

    assert metrics.current_accuracy == 1.0
    assert metrics.as_of_accuracy == 1.0
    assert metrics.history_accuracy == 1.0
    assert metrics.engine_decision_accuracy == 1.0
    assert metrics.avg_llm_calls_per_question == 0.0


def test_p83_designed_to_lose_scenarios() -> None:
    """Verifies designed_to_lose scenarios (lose-001 to lose-005) in structured mode."""
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    lose_scenarios = [s for s in all_scenarios if s.get("category") == "designed_to_lose"]
    assert len(lose_scenarios) == 5

    metrics, traces = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (structured)",
        scenarios=lose_scenarios,
        app_config=cfg,
        mode="structured",
    )

    assert metrics.conflict_recall == 1.0
    assert metrics.conflict_false_alarm_rate == 0.0
    assert metrics.engine_decision_accuracy == 1.0
    assert metrics.avg_llm_calls_per_question == 0.0

    # Detailed scenario-level verification inspecting actual traces and state
    traces_by_id = {t.scenario_id: t for t in traces}

    # lose-001: insufficient evidence -> system abstains
    t1 = traces_by_id["lose-001"]
    assert t1.actual["abstained"] is True
    assert t1.actual["value"] is None
    assert t1.is_current_correct is True

    # lose-002: peer dispute without authority -> conflict flagged with both values
    t2 = traces_by_id["lose-002"]
    assert t2.actual["conflict_flagged"] is True
    assert sorted(t2.actual["values"]) == ["friday", "monday"]
    assert t2.is_current_correct is True

    # lose-003: unsupported historical date prior to records -> system abstains
    t3 = traces_by_id["lose-003"]
    assert t3.actual["abstained"] is True
    assert t3.actual["value"] is None
    assert t3.is_as_of_correct is True

    # lose-004: low-trust rumor rejected -> preserves valid 3.12, rejects 3.9
    t4 = traces_by_id["lose-004"]
    assert t4.actual["value"] == "3.12"
    assert "3.9" not in str(t4.actual["value"])
    assert "3.9" not in str(t4.actual.get("values", []))
    assert t4.is_current_correct is True

    # Additionally inspect the actual persisted decision log record for the low-trust claim
    tf_lose4 = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_lose4 = tf_lose4.name
    tf_lose4.close()
    try:
        svc_lose4 = MemoryService(db_lose4, cfg=cfg)
        sc_lose4 = next(s for s in lose_scenarios if s.get("id") == "lose-004")
        for msg in sc_lose4["messages"]:
            m_id = msg["id"]
            t_dt = datetime.fromisoformat(msg["t"].replace("Z", "+00:00"))
            im = IngestMessage(source_id=msg["source"], event_time=t_dt, text=msg["text"])
            claims = [Claim.model_validate(c) for c in sc_lose4["structured_claims"][m_id]]
            svc_lose4.ingest_claims(im, claims)

        dlogs = svc_lose4.get_decision_logs()
        m2_decisions = [dl for dl in dlogs if dl.claim_json.get("value") == "3.9"]
        assert len(m2_decisions) == 1
        assert m2_decisions[0].rule_id == "R-REJECT"
        assert m2_decisions[0].action == Action.REJECT
        assert "rejected" in m2_decisions[0].explanation.lower()
    finally:
        svc_lose4.repo.conn.close()
        if os.path.exists(db_lose4):
            os.remove(db_lose4)

    # lose-005: upstream dependency broken -> flags needs_review rather than assuming compatibility
    t5 = traces_by_id["lose-005"]
    assert t5.actual["needs_review_flagged"] is True
    dep_pairs = [(d[0].lower(), d[1].lower()) for d in t5.actual.get("dependents", [])]
    assert ("library x", "compatibility_note") in dep_pairs
    assert t5.impact_recall == 1.0


def test_p83_reproducible_structured_results() -> None:
    """Verifies evaluation on P8.3 scenarios produces identical deterministic results."""
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    p83_scenarios = [s for s in all_scenarios if s.get("category") in P83_CATEGORIES]

    metrics1, _ = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (structured)",
        scenarios=p83_scenarios,
        app_config=cfg,
        mode="structured",
    )
    metrics2, _ = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (structured)",
        scenarios=p83_scenarios,
        app_config=cfg,
        mode="structured",
    )

    dump1 = metrics1.model_dump(exclude={"metadata", "avg_latency_per_question"})
    dump2 = metrics2.model_dump(exclude={"metadata", "avg_latency_per_question"})
    assert dump1 == dump2


class P83DeterministicE2EProvider(LLMProvider):
    """Deterministic simulation provider for end-to-end P8.3 evaluation.

    Operates strictly on prompt strings without live network calls or YAML access.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        self.call_count += 1
        sys_low = system.lower()

        # 1. Query parser prompt
        if "query parser" in sys_low or "parse_query" in sys_low:
            q_match = re.search(r'User Question:\s*"""(.*?)"""', user, re.DOTALL)
            q_text = q_match.group(1).strip() if q_match else user
            q_low = q_text.lower()

            # Attribute detection
            if "python" in q_low or "version" in q_low:
                attr = "python_version"
            elif "os" in q_low or "ubuntu" in q_low or "debian" in q_low:
                attr = "os_release"
            elif "deploy" in q_low:
                attr = "deploy_day"
            elif any(
                w in q_low for w in ["status", "operational", "maintenance", "degraded", "up"]
            ):
                attr = "service_status"
            elif any(
                w in q_low
                for w in ["owner", "owned", "owns", "team", "who owns", "sre", "platform"]
            ):
                attr = "owner_team"
            elif "break" in q_low or "upgrade" in q_low or "impact" in q_low:
                if "service" in q_low:
                    attr = "runs_service"
                else:
                    attr = "python_version"
            elif "service" in q_low:
                attr = "runs_service"
            else:
                attr = None

            # Intent detection
            if "why" in q_low or "believe" in q_low:
                intent = "why"
            elif "break" in q_low or "impact" in q_low:
                intent = "impact"
            elif any(
                w in q_low
                for w in [
                    "history",
                    "releases",
                    "which python versions",
                    "which deploy days",
                    "which os releases",
                    "has server a run",
                    "has server a had",
                ]
            ):
                intent = "history"
            elif any(
                w in q_low
                for w in [
                    "march",
                    "february",
                    "april",
                    "january",
                    "june",
                    "2024",
                    "in feb",
                    "in jan",
                ]
            ):
                intent = "as_of"
            else:
                intent = "current"

            # Context detection
            ctx: dict[str, str] = {}
            if "production" in q_low:
                ctx["env"] = "production"
            elif "staging" in q_low:
                ctx["env"] = "staging"
            elif "prod" in q_low:
                ctx["env"] = "prod"
            elif "dev" in q_low:
                ctx["env"] = "dev"
            elif "us-east" in q_low:
                ctx["region"] = "us-east"
            elif "eu-central" in q_low:
                ctx["region"] = "eu-central"
            elif "project alpha" in q_low or "in alpha" in q_low or "alpha" in q_low:
                ctx["project"] = "alpha"
            elif "project beta" in q_low or "in beta" in q_low or "beta" in q_low:
                ctx["project"] = "beta"

            if attr is None:
                return json.dumps(
                    {
                        "entity": "Server B" if "server b" in q_low else "Server A",
                        "attribute": None,
                        "intent": intent,
                        "temporal_intent": intent,
                        "query_type": "fact_lookup",
                        "context": ctx,
                        "valid_at": None,
                        "known_at": None,
                        "confidence": 0.0,
                        "reasoning": "Unsupported attribute or ambiguous query.",
                    }
                )

            # Temporal coordinates
            valid_at = None
            if "march" in q_low:
                valid_at = "2026-03-15T00:00:00Z"
            elif "february" in q_low or "in feb" in q_low:
                valid_at = "2026-02-15T00:00:00Z"
            elif "april" in q_low:
                valid_at = "2026-04-01T00:00:00Z"
            elif "january" in q_low or "in jan" in q_low:
                valid_at = "2026-01-20T00:00:00Z"
            elif "2024" in q_low:
                valid_at = "2024-01-01T00:00:00Z"
            elif "june" in q_low and intent == "as_of":
                valid_at = "2026-06-15T00:00:00Z"

            known_at = None
            if "june 5" in q_low:
                known_at = "2026-06-05T00:00:00Z"
            elif "june 20" in q_low:
                known_at = "2026-06-20T00:00:00Z"
            elif "february" in q_low and "january" in q_low:
                known_at = "2026-02-01T00:00:00Z"
            elif "june" in q_low and "status in june" in q_low:
                known_at = "2026-06-25T00:00:00Z"

            return json.dumps(
                {
                    "entity": "Server B" if "server b" in q_low else "Server A",
                    "attribute": attr,
                    "intent": intent,
                    "temporal_intent": intent,
                    "query_type": "fact_lookup",
                    "context": ctx,
                    "valid_at": valid_at,
                    "known_at": known_at,
                    "confidence": 0.95,
                }
            )

        # 2. Relation classifier prompt
        elif "relation classifier" in sys_low or "classify_relation" in sys_low:
            return json.dumps(
                {
                    "relation": "conflicts",
                    "confidence": 0.95,
                    "reasoning": "Value upgrade detected.",
                }
            )

        # 3. Answer composer prompt
        elif "answer composer" in sys_low or "compose_answer" in sys_low:
            mem_ids = [int(m) for m in re.findall(r"Memory M(\d+):", user)]
            has_dispute = "dispute" in user.lower() or "disputed" in user.lower()
            needs_review = "needs_review" in user.lower()

            if "No active memories found" in user or "No memories valid" in user or not mem_ids:
                if not has_dispute:
                    return json.dumps(
                        {
                            "answer_text": "I do not have enough information to answer.",
                            "value": None,
                            "values": [],
                            "abstained": True,
                            "conflict_flagged": False,
                            "needs_review_flagged": False,
                            "referenced_memory_ids": [],
                        }
                    )

            val_match = re.search(r'=\s*"([^"]+)"', user)
            val = val_match.group(1) if val_match else None
            all_vals = re.findall(r'=\s*"([^"]+)"', user)

            if has_dispute and len(all_vals) >= 2:
                return json.dumps(
                    {
                        "answer_text": f"Dispute between {', '.join(all_vals)}.",
                        "value": all_vals[0],
                        "values": all_vals,
                        "abstained": False,
                        "conflict_flagged": True,
                        "needs_review_flagged": False,
                        "referenced_memory_ids": mem_ids,
                    }
                )

            return json.dumps(
                {
                    "answer_text": (f"Server A is {val}." if val else "Information retrieved."),
                    "value": val,
                    "values": all_vals if len(all_vals) > 1 else ([val] if val else []),
                    "abstained": False,
                    "conflict_flagged": False,
                    "needs_review_flagged": needs_review,
                    "referenced_memory_ids": mem_ids,
                }
            )

        return "{}"


class P83NaiveRAGBaselineProvider(LLMProvider):
    """Deterministic simulation of naive RAG baseline answer generation.

    Receives ONLY the question and retrieved top-k messages.
    Does not implement bitemporal valid/known times or dependency propagation.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        self.call_count += 1

        # 1. Extract question
        q_match = re.search(r'User Question:\s*"""(.*?)"""', user, re.DOTALL)
        q_text = q_match.group(1).strip() if q_match else ""
        q_low = q_text.lower()

        # 2. Extract retrieved messages block
        msgs_match = re.search(r"Retrieved Messages:\s*(.*?)\s*Instructions:", user, re.DOTALL)
        msgs_text = msgs_match.group(1).strip() if msgs_match else ""

        if not msgs_text or "No messages retrieved" in msgs_text:
            return json.dumps(
                {
                    "value": None,
                    "values": [],
                    "conflict_flagged": False,
                    "needs_review_flagged": False,
                    "abstained": True,
                    "explanation": "No messages were retrieved to answer the question.",
                    "cited_memory_ids": [],
                    "answer_text": "No messages retrieved.",
                }
            )

        # 3. Detect attribute
        attr = None
        if "python" in q_low or "version" in q_low:
            attr = "python_version"
        elif "os" in q_low or "ubuntu" in q_low or "debian" in q_low:
            attr = "os_release"
        elif "deploy" in q_low:
            attr = "deploy_day"
        elif any(w in q_low for w in ["status", "operational", "maintenance", "degraded", "up"]):
            attr = "service_status"
        elif any(w in q_low for w in ["team", "owner", "owned", "who owns", "sre", "platform"]):
            attr = "owner_team"
        elif "service" in q_low:
            attr = "runs_service"

        # 4. Extract candidate facts in retrieval rank order
        candidates: list[str] = []
        for line in msgs_text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            line_low = line_str.lower()

            if attr == "python_version":
                for m in re.finditer(r"\bpython\s*([0-9]+\.[0-9]+)\b", line_low):
                    val = m.group(1)
                    if val not in candidates:
                        candidates.append(val)
                if "python" in line_low:
                    for m in re.finditer(r"\b(3\.[0-9]+)\b", line_low):
                        val = m.group(1)
                        if val not in candidates:
                            candidates.append(val)

            elif attr == "os_release":
                for m in re.finditer(
                    r"\b(ubuntu\s+[0-9]+\.[0-9]+(?:\s+lts)?|debian\s+[0-9]+)\b",
                    line_low,
                ):
                    val = m.group(1)
                    if val not in candidates:
                        candidates.append(val)

            elif attr == "deploy_day":
                for d in [
                    "monday",
                    "tuesday",
                    "wednesday",
                    "thursday",
                    "friday",
                    "saturday",
                    "sunday",
                ]:
                    if d in line_low and d not in candidates:
                        candidates.append(d)

            elif attr == "service_status":
                for st in ["maintenance", "degraded", "down", "up"]:
                    if re.search(r"\b" + st + r"\b", line_low) and st not in candidates:
                        candidates.append(st)

            elif attr == "owner_team":
                for tm in [
                    "sre core",
                    "core platform team",
                    "infra team",
                    "data engineering",
                    "platform ops",
                ]:
                    if tm in line_low and tm not in candidates:
                        candidates.append(tm)

            elif attr == "runs_service":
                for svc in ["auth-service", "api-gateway", "redis-cache"]:
                    if svc in line_low and svc not in candidates:
                        candidates.append(svc)

        if not candidates:
            return json.dumps(
                {
                    "value": None,
                    "values": [],
                    "conflict_flagged": False,
                    "needs_review_flagged": False,
                    "abstained": True,
                    "explanation": (
                        f"Retrieved context contains no information about {attr or 'query'}."
                    ),
                    "cited_memory_ids": [],
                    "answer_text": "I do not have enough information to answer.",
                }
            )

        is_list_query = any(
            w in q_low
            for w in [
                "history",
                "releases",
                "which python versions",
                "which deploy days",
                "which os releases",
                "has server a run",
                "has server a had",
            ]
        )
        has_conflict = len(candidates) > 1 and not is_list_query

        if is_list_query:
            val = candidates[0]
            vals = candidates
            ans_text = f"Retrieved context indicates: {', '.join(candidates)}."
        else:
            val = candidates[0]
            vals = [val]
            ans_text = f"Retrieved context indicates {val}."

        return json.dumps(
            {
                "value": val,
                "values": vals,
                "conflict_flagged": has_conflict,
                "needs_review_flagged": False,
                "abstained": False,
                "explanation": (
                    f"Extracted from retrieved messages ({len(candidates)} candidate(s) found)."
                ),
                "cited_memory_ids": [],
                "answer_text": ans_text,
            }
        )


def test_p83_all_scenarios_engine_only_mode() -> None:
    """Verifies all 26 P8.3 scenarios execute in engine-only mode with 0 LLM calls."""
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    p83_scenarios = [s for s in all_scenarios if s.get("category") in P83_CATEGORIES]
    assert len(p83_scenarios) == 26

    metrics, traces = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (engine-only)",
        scenarios=p83_scenarios,
        app_config=cfg,
        mode="engine-only",
    )

    assert len(traces) == 50
    assert metrics.current_accuracy == 1.0
    assert metrics.as_of_accuracy == 1.0
    assert metrics.history_accuracy == 1.0
    assert metrics.engine_decision_accuracy == 1.0
    assert metrics.conflict_recall == 1.0
    assert metrics.conflict_false_alarm_rate == 0.0
    assert metrics.impact_recall == 1.0
    assert metrics.impact_precision == 1.0
    assert metrics.provenance_correctness == 1.0
    assert metrics.avg_llm_calls_per_question == 0.0

    for t in traces:
        if t.question_type == "current":
            assert t.is_current_correct is True
            assert t.is_stale is not True
        elif t.question_type == "as_of":
            assert t.is_as_of_correct is True
        elif t.question_type == "history":
            assert t.is_history_correct is True
        elif t.question_type == "why" and t.provenance_correct is not None:
            assert t.provenance_correct is True
        elif t.question_type == "impact" and t.impact_recall is not None:
            assert t.impact_recall == 1.0


def test_p83_all_scenarios_e2e_mode() -> None:
    """Verifies all 26 P8.3 scenarios execute in E2E mode with simulated provider."""
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    p83_scenarios = [s for s in all_scenarios if s.get("category") in P83_CATEGORIES]
    assert len(p83_scenarios) == 26

    provider = P83DeterministicE2EProvider()
    llm_client = LLMClient(provider=provider, config=cfg.engine.llm)

    metrics, traces = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (e2e)",
        scenarios=p83_scenarios,
        app_config=cfg,
        mode="e2e",
        llm_client=llm_client,
    )

    assert len(traces) == 50
    assert metrics.current_accuracy == 1.0
    assert metrics.as_of_accuracy == 1.0
    assert metrics.history_accuracy == 1.0
    assert metrics.conflict_recall == 1.0
    assert metrics.impact_recall == 1.0
    assert metrics.avg_llm_calls_per_question > 0.0

    for t in traces:
        if t.question_type == "current":
            assert t.is_current_correct is True
            assert t.is_stale is not True
        elif t.question_type == "as_of":
            assert t.is_as_of_correct is True
        elif t.question_type == "history":
            assert t.is_history_correct is True

    # Assert individual designed-to-lose traces in E2E mode
    e2e_by_id = {t.scenario_id: t for t in traces}
    assert e2e_by_id["lose-001"].actual["abstained"] is True
    assert e2e_by_id["lose-002"].actual["conflict_flagged"] is True
    assert e2e_by_id["lose-003"].actual["abstained"] is True
    assert e2e_by_id["lose-004"].actual["value"] == "3.12"
    assert e2e_by_id["lose-005"].actual["needs_review_flagged"] is True


def test_p83_all_scenarios_naive_rag_baseline_mode() -> None:
    """Verifies all 26 P8.3 scenarios execute in FastEmbed naive RAG baseline mode."""
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    p83_scenarios = [s for s in all_scenarios if s.get("category") in P83_CATEGORIES]
    assert len(p83_scenarios) == 26

    b0_provider = P83NaiveRAGBaselineProvider()
    llm_client = LLMClient(provider=b0_provider, config=cfg.engine.llm)

    metrics, traces = run_system_on_scenarios(
        system_id="B0",
        system_name="Naive RAG (B0)",
        scenarios=p83_scenarios,
        app_config=cfg,
        mode="baseline",
        llm_client=llm_client,
    )

    assert len(traces) == 50
    assert metrics.system_id == "B0"
    assert metrics.retrieval_method == "fastembed"
    # FastEmbed baseline reports None (n/a) for graph and engine metrics
    assert metrics.engine_decision_accuracy is None
    assert metrics.impact_recall is None
    assert metrics.provenance_correctness is None
    # Baseline shows significant failure rates across temporal and context scenarios
    assert metrics.current_accuracy is not None and metrics.current_accuracy < 1.0
    assert metrics.stale_answer_rate is not None and metrics.stale_answer_rate > 0.0
    assert metrics.conflict_false_alarm_rate is not None and metrics.conflict_false_alarm_rate > 0.0


def test_p83_bitemporal_time_travel_regression() -> None:
    """Verifies bitemporal transaction-time regression behavior using temporal-002.

    Demonstrates that with identical valid_at (2026-03-01):
    - When queried as known_at June 5 (before m2 backfill arrives on June 15),
      the system correctly abstains (knowledge did not exist then).
    - When queried as known_at June 20 (after m2 backfill arrives),
      the system correctly resolves to backfilled value '3.10'.
    """
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    t002 = [s for s in all_scenarios if s.get("id") == "temporal-002"]
    assert len(t002) == 1

    metrics, traces = run_system_on_scenarios(
        system_id="MM",
        system_name="MemoryMesh (bitemporal)",
        scenarios=t002,
        app_config=cfg,
        mode="structured",
    )

    assert len(traces) == 2
    q1_trace = traces[0]
    q2_trace = traces[1]

    # q1: valid_at=2026-03-01, known_at=2026-06-05 (prior to June 15 backfill)
    assert q1_trace.question_id == "q1"
    assert q1_trace.actual["abstained"] is True
    assert q1_trace.actual["value"] is None
    assert q1_trace.is_as_of_correct is True

    # q2: valid_at=2026-03-01, known_at=2026-06-20 (post June 15 backfill)
    assert q2_trace.question_id == "q2"
    assert q2_trace.actual["abstained"] is False
    assert q2_trace.actual["value"] == "3.10"
    assert q2_trace.is_as_of_correct is True

    # Regression comparison: identical valid_at, divergent belief outcomes
    assert q1_trace.actual["value"] != q2_trace.actual["value"]
    assert q1_trace.actual["abstained"] != q2_trace.actual["abstained"]


def test_p83_deterministic_e2e_provider_safety() -> None:
    """Verifies P83DeterministicE2EProvider safe failure and parsing behavior."""
    provider = P83DeterministicE2EProvider()
    sys_prompt = "You are a query parser."

    # 1. Unsupported attribute does not silently fallback to python_version
    user_unsupported = 'User Question:\n"""\nWhat is the chassis color of Server A?\n"""'
    res_unsupported = json.loads(provider.complete(sys_prompt, user_unsupported, model="test"))
    assert res_unsupported["attribute"] is None
    assert res_unsupported["confidence"] == 0.0

    # 2. Ambiguous query without recognized attribute
    user_ambiguous = 'User Question:\n"""\nTell me something interesting about Server A.\n"""'
    res_ambiguous = json.loads(provider.complete(sys_prompt, user_ambiguous, model="test"))
    assert res_ambiguous["attribute"] is None

    # 3. Project context extraction
    user_project = (
        'User Question:\n"""\nWhat Python version does Server A use in Project Alpha?\n"""'
    )
    res_project = json.loads(provider.complete(sys_prompt, user_project, model="test"))
    assert res_project["attribute"] == "python_version"
    assert res_project["context"] == {"project": "alpha"}


def test_p83_naive_rag_baseline_provider_prompt_grounding() -> None:
    """Verifies P83NaiveRAGBaselineProvider derives answers strictly from prompt text."""
    provider = P83NaiveRAGBaselineProvider()
    sys_prompt = "You are a helpful assistant answering from retrieved context."

    # 1. Grounding in Python version from prompt
    user_p1 = (
        'User Question:\n"""\nWhat Python version does Server A use in production?\n"""\n\n'
        "Retrieved Messages:\n"
        "[m1] (2026-01-10T09:00:00Z, source: priya): In production, Server A runs Python 3.12.\n\n"
        "Instructions:\n..."
    )
    ans1 = json.loads(provider.complete(sys_prompt, user_p1, model="test"))
    assert ans1["value"] == "3.12"
    assert ans1["abstained"] is False

    # 2. Context change in prompt changes answer
    user_p2 = (
        'User Question:\n"""\nWhat Python version does Server A use in staging?\n"""\n\n'
        "Retrieved Messages:\n"
        "[m1] (2026-01-12T11:00:00Z, source: dev): In staging, Server A runs Python 3.11.\n\n"
        "Instructions:\n..."
    )
    ans2 = json.loads(provider.complete(sys_prompt, user_p2, model="test"))
    assert ans2["value"] == "3.11"
    assert ans2["value"] != ans1["value"]

    # 3. Empty or irrelevant context produces abstention
    user_empty = (
        'User Question:\n"""\nWhich day does Server A deploy?\n"""\n\n'
        "Retrieved Messages:\n"
        "No messages retrieved.\n\n"
        "Instructions:\n..."
    )
    ans_empty = json.loads(provider.complete(sys_prompt, user_empty, model="test"))
    assert ans_empty["abstained"] is True
    assert ans_empty["value"] is None


def test_p83_lose_004_explicit_r_reject_decision_log() -> None:
    """Verifies that the low-trust claim in lose-004 is rejected via explicit R-REJECT record.

    Inspects the persisted decision log in the database rather than merely inferring rejection
    from the final question answer.
    """
    cfg = load_app_config()
    all_scenarios = load_scenarios(SCENARIOS_DIR)
    sc = next(s for s in all_scenarios if s.get("id") == "lose-004")

    tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tf.name
    tf.close()
    try:
        svc = MemoryService(db_path, cfg=cfg)
        for msg in sc["messages"]:
            m_id = msg["id"]
            t_dt = datetime.fromisoformat(msg["t"].replace("Z", "+00:00"))
            im = IngestMessage(source_id=msg["source"], event_time=t_dt, text=msg["text"])
            claims = [Claim.model_validate(c) for c in sc["structured_claims"][m_id]]
            svc.ingest_claims(im, claims)

        # Inspect persisted decision log
        dlogs = svc.get_decision_logs()
        assert len(dlogs) == 2

        # First claim from sysadmin_priya was inserted
        assert dlogs[0].rule_id == "R-INSERT"
        assert dlogs[0].claim_json.get("value") == "3.12"

        # Second claim from intern_chat (low trust) was explicitly rejected
        low_trust_dl = dlogs[1]
        assert low_trust_dl.rule_id == "R-REJECT"
        assert low_trust_dl.action == Action.REJECT
        assert low_trust_dl.claim_json.get("value") == "3.9"
        assert "below reject threshold" in low_trust_dl.explanation
    finally:
        svc.repo.conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_p83_dependency_cycle_propagation_proven() -> None:
    """Demonstrates and proves dependency propagation through an actual directed cycle.

    Data Model Limitation Documented:
    Supported sequential claim ingestion in MemoryMesh creates strictly acyclic memory
    DAGs because each new memory record is assigned a monotonic ID strictly greater than
    any pre-existing target memory returned by retrieve_current (src_memory_id > dst_memory_id).
    Consequently, memory-level directed edge cycles cannot be constructed through supported
    message ingestion alone.

    However, the underlying dependency resolution and propagation engine (propagate.py)
    is designed to be cycle-safe. This test establishes real DEPENDS_ON edges in the database
    that form an actual directed cycle between distinct memory nodes, inspects and proves the
    directed cycle before triggering propagation, and verifies that propagation terminates
    without infinite recursion, processing each affected memory at most once.
    """
    cfg = load_app_config()
    tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tf.name
    tf.close()

    try:
        svc = MemoryService(db_path, cfg=cfg)

        # Ingest M1: Server A runs auth-service
        r1 = svc.ingest_claims(
            IngestMessage(
                source_id="sysadmin_priya",
                event_time=datetime.fromisoformat("2026-01-10T09:00:00+00:00"),
                text="Server A runs auth-service.",
            ),
            [
                Claim(
                    subject="Server A",
                    attribute="runs_service",
                    value="auth-service",
                    polarity="assert",
                    valid_from=datetime.fromisoformat("2026-01-10T09:00:00+00:00"),
                    extract_confidence=0.95,
                )
            ],
        )
        m1_id = r1.decisions[0].memory_id
        assert m1_id is not None

        # Ingest M2: Server B runs api-gateway depending on Server A runs_service
        r2 = svc.ingest_claims(
            IngestMessage(
                source_id="sysadmin_priya",
                event_time=datetime.fromisoformat("2026-01-15T10:00:00+00:00"),
                text="Server B runs api-gateway depending on Server A.",
            ),
            [
                Claim(
                    subject="Server B",
                    attribute="runs_service",
                    value="api-gateway",
                    depends_on=[{"entity": "Server A", "attribute": "runs_service"}],
                    polarity="assert",
                    valid_from=datetime.fromisoformat("2026-01-15T10:00:00+00:00"),
                    extract_confidence=0.95,
                )
            ],
        )
        m2_id = r2.decisions[0].memory_id
        assert m2_id is not None

        # Ingest M3: Server C runs cache-service depending on Server B runs_service
        r3 = svc.ingest_claims(
            IngestMessage(
                source_id="sysadmin_priya",
                event_time=datetime.fromisoformat("2026-01-20T10:00:00+00:00"),
                text="Server C runs cache-service depending on Server B.",
            ),
            [
                Claim(
                    subject="Server C",
                    attribute="runs_service",
                    value="cache-service",
                    depends_on=[{"entity": "Server B", "attribute": "runs_service"}],
                    polarity="assert",
                    valid_from=datetime.fromisoformat("2026-01-20T10:00:00+00:00"),
                    extract_confidence=0.95,
                )
            ],
        )
        m3_id = r3.decisions[0].memory_id
        assert m3_id is not None

        # Ingest created edges:
        # M2 --DEPENDS_ON--> M1  (src=2, dst=1) => 1 -> 2 (2 is downstream dependent of 1)
        # M3 --DEPENDS_ON--> M2  (src=3, dst=2) => 2 -> 3 (3 is downstream dependent of 2)
        # To close the directed cycle, we insert edge:
        # M1 --DEPENDS_ON--> M3  (src=1, dst=3) => 3 -> 1 (1 is downstream dependent of 3)
        svc.repo.insert_edge(
            src_memory_id=m1_id,
            dst_memory_id=m3_id,
            edge_type=EdgeType.DEPENDS_ON,
            reason="Establish mutual circular dependency",
            created_at=datetime.fromisoformat("2026-01-21T10:00:00+00:00"),
        )

        # Inspect and prove the actual dependency edges before triggering propagation
        dep_edges = svc.repo.get_transitive_dependency_edges([m1_id], direction="downstream")
        edge_set = {(e.src_memory_id, e.dst_memory_id, e.type) for e in dep_edges}

        assert (m2_id, m1_id, EdgeType.DEPENDS_ON) in edge_set  # M2 depends on M1
        assert (m3_id, m2_id, EdgeType.DEPENDS_ON) in edge_set  # M3 depends on M2
        assert (m1_id, m3_id, EdgeType.DEPENDS_ON) in edge_set  # M1 depends on M3

        # Prove directed path in both directions between distinct memory nodes:
        # Directed downstream path M1 -> M2: (src=2, dst=1)
        # Directed downstream path M2 -> M1: M2 -> M3 -> M1 via (src=3, dst=2) and (src=1, dst=3)
        # Directed downstream path M1 -> M3: M1 -> M2 -> M3
        # Directed downstream path M3 -> M1: M3 -> M1 via (src=1, dst=3)

        # Trigger an upstream mutation / retraction of M1
        retract_report = svc.ingest_claims(
            IngestMessage(
                source_id="sysadmin_priya",
                event_time=datetime.fromisoformat("2026-02-01T09:00:00+00:00"),
                text="Server A auth-service is decommissioned.",
            ),
            [
                Claim(
                    subject="Server A",
                    attribute="runs_service",
                    value="auth-service",
                    polarity="retract",
                    valid_from=datetime.fromisoformat("2026-02-01T09:00:00+00:00"),
                    extract_confidence=0.95,
                )
            ],
        )

        # Assert traversal terminates successfully without infinite recursion
        assert retract_report.success is True
        decisions = retract_report.decisions
        assert len(decisions) >= 1
        assert decisions[0].rule_id == "R-RETRACT"
        assert decisions[0].memory_id == m1_id

        # Assert each affected memory is processed at most once
        prop_decisions = [d for d in decisions if d.rule_id == "R-PROPAGATE"]
        assert len(prop_decisions) == 2

        affected_ids = [d.memory_id for d in prop_decisions]
        assert affected_ids == [m2_id, m3_id]
        assert len(affected_ids) == len(set(affected_ids)), "Duplicate memory processing detected!"

        # Assert resulting review flags are set
        mem2 = svc.repo.get_memory(m2_id)
        mem3 = svc.repo.get_memory(m3_id)
        assert mem2 is not None and mem2.needs_review is True
        assert mem3 is not None and mem3.needs_review is True

        # Assert propagation decision records contain correct causal paths
        assert any(
            f"M{m1_id} -> M{m2_id}" in d.explanation and d.memory_id == m2_id
            for d in prop_decisions
        )
        assert any(
            f"M{m1_id} -> M{m2_id} -> M{m3_id}" in d.explanation and d.memory_id == m3_id
            for d in prop_decisions
        )
    finally:
        svc.repo.conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
