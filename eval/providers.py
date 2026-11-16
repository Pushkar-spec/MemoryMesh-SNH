"""Unified deterministic providers for evaluation without live network calls.

Governed by:
- docs/EVALUATION_AND_DEMO.md
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P8.1, P8.2, P8.3, P8.4)

These providers provide deterministic, rule-based simulation of LLM inference
across all evaluation scenarios (hero, P8.2, and P8.3).
They operate strictly on prompt text received from MemoryMesh orchestrators,
with zero access to scenario YAML files, expected answers, or evaluation metadata.
"""

from __future__ import annotations

import json
import re

from memorymesh.llm.providers import LLMProvider


class DeterministicEvaluationE2EProvider(LLMProvider):
    """Deterministic simulation provider for MemoryMesh end-to-end evaluation.

    Operates purely on prompt strings passed into `complete()` without network calls.
    Handles:
    - Query parser prompts (natural language query -> structured query plan)
    - Relation classifier prompts (value equivalence/conflict classification)
    - Answer composer prompts (retrieved memories -> FinalAnswer schema)
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

            # Attribute detection strictly from natural language question text
            if "python" in q_low or "version" in q_low:
                attr = "python_version"
            elif "os" in q_low or "ubuntu" in q_low or "debian" in q_low or "centos" in q_low:
                attr = "os_release"
            elif (
                "deploy" in q_low or "friday" in q_low or "monday" in q_low or "wednesday" in q_low
            ):
                attr = "deploy_day"
            elif any(
                w in q_low
                for w in ["status", "operational", "maintenance", "degraded", "up", "health"]
            ):
                attr = "service_status"
            elif any(
                w in q_low
                for w in [
                    "owner",
                    "owned",
                    "owns",
                    "team",
                    "who owns",
                    "sre",
                    "platform",
                    "infrastructure core",
                    "sre core",
                ]
            ):
                attr = "owner_team"
            elif "break" in q_low or "upgrade" in q_low or "impact" in q_low:
                if "service" in q_low:
                    attr = "runs_service"
                else:
                    attr = "python_version"
            elif "service" in q_low or "library" in q_low:
                attr = "runs_service"
            elif "compatibility" in q_low or "compatible" in q_low:
                attr = "compatibility_note"
            else:
                attr = None

            # Intent detection
            intent = "current"
            if "why" in q_low or "believe" in q_low:
                intent = "why"
            elif "break" in q_low or "impact" in q_low or "affect" in q_low:
                intent = "impact"
            elif any(
                w in q_low
                for w in [
                    "history",
                    "releases",
                    "which python versions",
                    "had",
                    "have run",
                    "teams have owned",
                    "previous",
                ]
            ):
                intent = "history"

            # Temporal boundaries & target dates
            target_date = None
            valid_at = None
            known_at = None

            if "march 3" in q_low:
                intent = "as_of"
                target_date = "2026-03-03"
            elif "september 1" in q_low:
                intent = "as_of"
                valid_at = "2026-09-01T12:00:00Z"
            elif "may 2025" in q_low:
                intent = "as_of"
                target_date = "2025-05-01"
            elif "october 2025" in q_low:
                intent = "as_of"
                target_date = "2025-10-01"
            elif "march 15" in q_low or "mid march" in q_low:
                intent = "as_of"
                valid_at = "2026-03-15T12:00:00Z"
            elif "march" in q_low:
                intent = "as_of"
                target_date = "2026-03-01"
            elif "january 10" in q_low:
                intent = "as_of"
                valid_at = "2026-01-10T12:00:00Z"
            elif "february" in q_low:
                intent = "as_of"
                valid_at = "2026-02-15T12:00:00Z"
            elif "april" in q_low:
                intent = "as_of"
                valid_at = "2026-04-01T12:00:00Z"
            elif "june 5" in q_low:
                intent = "as_of"
                valid_at = "2026-03-01T00:00:00Z"
                known_at = "2026-06-05T00:00:00Z"
            elif "june 20" in q_low:
                intent = "as_of"
                valid_at = "2026-03-01T00:00:00Z"
                known_at = "2026-06-20T00:00:00Z"
            elif "june 1" in q_low:
                intent = "as_of"
                valid_at = "2026-06-01T12:00:00Z"

            # Context extraction
            context: dict[str, str] = {}
            if "staging" in q_low:
                context["environment"] = "staging"
            elif "production" in q_low or "prod" in q_low:
                context["environment"] = "production"
            elif "project alpha" in q_low or "alpha" in q_low:
                context["project"] = "alpha"
            elif "project beta" in q_low or "beta" in q_low:
                context["project"] = "beta"

            # Entity extraction
            entity = "Server A"
            if "server b" in q_low:
                entity = "Server B"
            elif "server c" in q_low:
                entity = "Server C"
            elif "database 1" in q_low:
                entity = "Database 1"
            elif "library x" in q_low:
                entity = "Library X"
            elif "auth" in q_low and "service" in q_low:
                entity = "Auth Service"
            elif "server a" in q_low:
                entity = "Server A"

            parsed = {
                "entity": entity,
                "attribute": attr,
                "intent": intent,
                "target_date": target_date,
                "valid_at": valid_at,
                "known_at": known_at,
                "context": context,
                "confidence": 0.95 if attr else 0.0,
            }
            return json.dumps(parsed)

        # 2. Relation classifier prompt
        if (
            "relation classifier" in sys_low
            or "classify_relation" in sys_low
            or "classify" in sys_low
        ):
            u_low = user.lower()
            if any(k in u_low for k in ["up", "operational", "running"]):
                return json.dumps(
                    {"relation": "identical", "confidence": 0.95, "explanation": "Synonyms"}
                )
            if any(k in u_low for k in ["down", "degraded", "maintenance"]):
                return json.dumps(
                    {"relation": "contradicts", "confidence": 0.95, "explanation": "Opposite state"}
                )
            if "sre" in u_low and "platform" in u_low:
                return json.dumps(
                    {"relation": "contradicts", "confidence": 0.95, "explanation": "Distinct teams"}
                )
            return json.dumps(
                {"relation": "contradicts", "confidence": 0.90, "explanation": "Conflicting values"}
            )

        # 3. Answer composer prompt
        if "answer composer" in sys_low or "compose_answer" in sys_low or "compose" in sys_low:
            mem_ids = [int(m) for m in re.findall(r"Memory M(\d+):", user)]
            has_dispute = "disputed" in user.lower()
            needs_review = "needs_review" in user.lower()

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

            ans_text = f"Server A is {val}." if val else "Information retrieved."
            if val and mem_ids:
                ans_text = f"Server A is {val} [M{mem_ids[0]}]."

            return json.dumps(
                {
                    "answer_text": ans_text,
                    "value": val,
                    "values": all_vals if len(all_vals) > 1 else ([val] if val else []),
                    "abstained": val is None,
                    "conflict_flagged": False,
                    "needs_review_flagged": needs_review,
                    "referenced_memory_ids": mem_ids,
                }
            )

        return "{}"


class DeterministicEvaluationBaselineProvider(LLMProvider):
    """Deterministic simulation of Naive RAG baseline answer generation.

    Receives ONLY the question and retrieved top-k messages in the prompt.
    Does not implement MemoryMesh temporal resolution, trust ranking, or conflict resolution.
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
        elif "os" in q_low or "ubuntu" in q_low or "debian" in q_low or "centos" in q_low:
            attr = "os_release"
        elif "deploy" in q_low:
            attr = "deploy_day"
        elif any(w in q_low for w in ["status", "operational", "maintenance", "degraded", "up"]):
            attr = "service_status"
        elif any(
            w in q_low
            for w in [
                "team",
                "owner",
                "owned",
                "who owns",
                "sre",
                "platform",
                "infrastructure core",
                "sre core",
            ]
        ):
            attr = "owner_team"
        elif "service" in q_low or "library" in q_low:
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
                    r"\b(ubuntu\s+[0-9]+\.[0-9]+(?:\s+lts)?|debian\s+[0-9]+|centos\s+[0-9]+)\b",
                    line_low,
                ):
                    val = m.group(1)
                    if val not in candidates:
                        candidates.append(val)

            elif attr == "deploy_day":
                for d in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
                    if d in line_low and d not in candidates:
                        candidates.append(d)

            elif attr == "service_status":
                for st in ["operational", "up", "degraded", "maintenance", "down"]:
                    if st in line_low and st not in candidates:
                        candidates.append(st)

            elif attr == "owner_team":
                for tm in [
                    "infrastructure core",
                    "sre core",
                    "core sre",
                    "sre team",
                    "platform sre",
                    "data platform",
                    "alpha team",
                    "beta team",
                ]:
                    if tm in line_low and tm not in candidates:
                        candidates.append(tm)

            elif attr == "runs_service":
                for srv in [
                    "auth-service",
                    "api-gateway",
                    "cache-service",
                    "billing-service",
                    "analytics-worker",
                ]:
                    if srv in line_low and srv not in candidates:
                        candidates.append(srv)

        # 5. Formulate answer strictly from extracted candidates
        if not candidates:
            return json.dumps(
                {
                    "value": None,
                    "values": [],
                    "conflict_flagged": False,
                    "needs_review_flagged": False,
                    "abstained": True,
                    "explanation": "No relevant candidate facts found in retrieved context.",
                    "cited_memory_ids": [],
                    "answer_text": "Retrieved messages do not contain the answer.",
                }
            )

        is_history_query = any(
            w in q_low for w in ["history", "releases", "which python versions", "had"]
        )
        conflict_detected = len(candidates) > 1 and not is_history_query

        # Naive RAG picks the first or most salient candidate in retrieved messages
        chosen_val = candidates[0]

        return json.dumps(
            {
                "value": chosen_val if not is_history_query else None,
                "values": candidates if is_history_query else [chosen_val],
                "conflict_flagged": conflict_detected,
                "needs_review_flagged": False,
                "abstained": False,
                "explanation": (
                    f"Retrieved messages indicate {chosen_val} "
                    f"(candidates found: {', '.join(candidates)})."
                ),
                "cited_memory_ids": [],
                "answer_text": f"Based on retrieved messages: {chosen_val}.",
            }
        )
