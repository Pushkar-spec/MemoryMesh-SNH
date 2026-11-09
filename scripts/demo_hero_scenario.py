"""CLI demonstration of MemoryMesh's G1 structured hero scenario (P5.3).

Demonstrates an end-to-end deterministic scenario showcasing:
  1. Initial administrator claim ingestion (R-INSERT).
  2. Administrator update with trust-aware temporal replacement (R-REPLACE).
  3. Lower-trust conflicting claim rejection from an intern (R-REJECT).
  4. Temporal retrieval (current 3.12 vs historical 3.10).
  5. Decision explainability, provenance, evidence, and transition edges.
  6. Multi-level dependency creation and downstream change propagation (R-PROPAGATE).
  7. Downstream impact analysis and blast-radius path discovery.
  8. Human review acknowledgement (R-ACKNOWLEDGE) with factual immutability
     and downstream independent isolation.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import Any

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.models import (
    Action,
    Claim,
    EdgeType,
    IngestMessage,
    MemoryStatus,
    format_datetime_utc,
)
from memorymesh.service import MemoryService

# Fixed reference timestamp for deterministic demo output
DEMO_CLOCK_REF = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)


def run_demo(stream: Any = sys.stdout) -> int:
    """Execute the complete G1 structured hero scenario and print formatted results."""
    conn = None
    try:
        # Initialize an isolated in-memory SQLite database
        conn = init_db(":memory:")
        cfg = load_app_config()
        service = MemoryService(db_target=conn, cfg=cfg, clock=lambda: DEMO_CLOCK_REF)

        print("=" * 80, file=stream)
        print("MemoryMesh: G1 Structured Hero Scenario Demonstration (P5.3)", file=stream)
        print("=" * 80, file=stream)
        print("Isolated database initialized: SQLite (:memory:)", file=stream)
        print("Configuration catalog loaded from engine, sources, and attributes.", file=stream)
        print(file=stream)

        # ---------------------------------------------------------------------
        # STEP 1: Initial Administrator Claim
        # ---------------------------------------------------------------------
        t1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
        t1_str = format_datetime_utc(t1)
        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A provisioned with Python 3.10 runtime",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            valid_from=t1,
            extract_confidence=0.95,
        )

        print("-" * 80, file=stream)
        print("STEP 1: Initial Administrator Claim", file=stream)
        print("-" * 80, file=stream)
        print(
            f'Message : "{msg1.text}" (source: {msg1.source_id}, event_time: {t1_str})',
            file=stream,
        )
        print(
            f'Claim   : {claim1.subject} -> {claim1.attribute} = "{claim1.value}" '
            f"(valid_from: {t1_str})",
            file=stream,
        )
        print(file=stream)

        report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
        if not report1.decisions:
            print("[ERROR] Expected a resolution decision for claim 1, got none.", file=stream)
            return 1

        dec1 = report1.decisions[0]
        m1_id = dec1.memory_id
        mem1 = service.repo.get_memory(m1_id) if m1_id else None
        if mem1 is None:
            print("[ERROR] Failed to retrieve persisted memory 1.", file=stream)
            return 1

        print("Resolution Decision:", file=stream)
        print(f"  Action      : {dec1.action.value}", file=stream)
        print(f"  Rule ID     : {dec1.rule_id}", file=stream)
        print(
            f'  Memory ID   : #{mem1.id} (status: {mem1.status.value}, value: "{mem1.value_norm}")',
            file=stream,
        )
        print(
            f"  Validity    : [{format_datetime_utc(mem1.valid_from)}, "
            f"{format_datetime_utc(mem1.valid_to) if mem1.valid_to else 'OPEN'})",
            file=stream,
        )
        print(f"  Confidence  : base={mem1.confidence:.4f}", file=stream)
        print(f"  Explanation : {dec1.explanation}", file=stream)
        print(file=stream)

        if dec1.rule_id != "R-INSERT":
            print(
                f"[ERROR] Step 1 failed: expected 'R-INSERT', got '{dec1.rule_id}'.",
                file=stream,
            )
            return 1

        # ---------------------------------------------------------------------
        # STEP 2: Administrator Update (Newer Conflicting Claim)
        # ---------------------------------------------------------------------
        t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        t2_str = format_datetime_utc(t2)
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A upgraded to Python 3.12",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            valid_from=t2,
            extract_confidence=0.95,
        )

        print("-" * 80, file=stream)
        print("STEP 2: Administrator Update (Trust-Aware Temporal Replacement)", file=stream)
        print("-" * 80, file=stream)
        print(
            f'Message : "{msg2.text}" (source: {msg2.source_id}, event_time: {t2_str})',
            file=stream,
        )
        print(
            f'Claim   : {claim2.subject} -> {claim2.attribute} = "{claim2.value}" '
            f"(valid_from: {t2_str})",
            file=stream,
        )
        print(file=stream)

        report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        if not report2.decisions:
            print("[ERROR] Expected a resolution decision for claim 2, got none.", file=stream)
            return 1

        dec2 = report2.decisions[0]
        m2_id = dec2.memory_id
        mem2 = service.repo.get_memory(m2_id) if m2_id else None
        if mem2 is None:
            print("[ERROR] Failed to retrieve persisted memory 2.", file=stream)
            return 1

        mem1_updated = service.repo.get_memory(mem1.id)
        if mem1_updated is None:
            print("[ERROR] Failed to retrieve updated memory 1.", file=stream)
            return 1

        vto_str = format_datetime_utc(mem1_updated.valid_to) if mem1_updated.valid_to else "OPEN"
        print("Resolution Decision:", file=stream)
        print(f"  Action      : {dec2.action.value}", file=stream)
        print(f"  Rule ID     : {dec2.rule_id}", file=stream)
        print(
            f'  New Memory  : #{mem2.id} (status: {mem2.status.value}, value: "{mem2.value_norm}")',
            file=stream,
        )
        print(
            f"  Validity    : [{format_datetime_utc(mem2.valid_from)}, "
            f"{format_datetime_utc(mem2.valid_to) if mem2.valid_to else 'OPEN'})",
            file=stream,
        )
        print(
            f"  Closed Old  : #{mem1_updated.id} (status: {mem1_updated.status.value}, "
            f"valid_to: {vto_str})",
            file=stream,
        )
        print(f"  Explanation : {dec2.explanation}", file=stream)
        print(file=stream)

        if dec2.rule_id != "R-REPLACE" or mem1_updated.status != MemoryStatus.HISTORICAL:
            print(
                "[ERROR] Step 2 failed: expected 'R-REPLACE' and historical status.",
                file=stream,
            )
            return 1

        # ---------------------------------------------------------------------
        # STEP 3: Lower-Trust Conflicting Claim Rejection
        # ---------------------------------------------------------------------
        t3 = datetime(2026, 6, 5, 0, 0, tzinfo=UTC)
        t3_str = format_datetime_utc(t3)
        msg3 = IngestMessage(
            source_id="intern_chat",
            text="Pretty sure Server A runs Python 3.9",
            event_time=t3,
        )
        claim3 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.9",
            valid_from=t3,
            extract_confidence=0.70,
        )

        print("-" * 80, file=stream)
        print("STEP 3: Lower-Trust Conflicting Claim Rejection", file=stream)
        print("-" * 80, file=stream)
        print(
            f'Message : "{msg3.text}" (source: {msg3.source_id}, reliability: 0.30, '
            f"event_time: {t3_str})",
            file=stream,
        )
        print(
            f'Claim   : {claim3.subject} -> {claim3.attribute} = "{claim3.value}" '
            f"(extract_confidence: {claim3.extract_confidence})",
            file=stream,
        )
        print(file=stream)

        report3 = service.ingest_claims(msg3, [claim3], recorded_at=t3)
        if not report3.decisions:
            print("[ERROR] Expected a resolution decision for claim 3, got none.", file=stream)
            return 1

        dec3 = report3.decisions[0]
        print("Resolution Decision:", file=stream)
        print(f"  Action      : {dec3.action.value}", file=stream)
        print(f"  Rule ID     : {dec3.rule_id}", file=stream)
        print(f"  Explanation : {dec3.explanation}", file=stream)
        print(file=stream)

        if dec3.rule_id != "R-REJECT" or dec3.action != Action.REJECT:
            print(
                f"[ERROR] Step 3 failed: expected 'R-REJECT', got '{dec3.rule_id}'.",
                file=stream,
            )
            return 1

        # ---------------------------------------------------------------------
        # STEP 4: Temporal Retrieval (Current vs As-Of March 15)
        # ---------------------------------------------------------------------
        t_curr = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
        t_hist = datetime(2026, 3, 15, 0, 0, tzinfo=UTC)

        print("-" * 80, file=stream)
        print("STEP 4: Temporal Retrieval (Point-in-Time Consistency)", file=stream)
        print("-" * 80, file=stream)

        res_curr = service.current("Server A", "python_version", now=t_curr)
        res_hist = service.current("Server A", "python_version", valid_at=t_hist, now=t_curr)

        curr_item = res_curr.items[0] if res_curr.items else None
        hist_item = res_hist.items[0] if res_hist.items else None

        if curr_item is None or hist_item is None:
            print("[ERROR] Failed to retrieve current or historical memory.", file=stream)
            return 1

        curr_vto = (
            format_datetime_utc(curr_item.memory.valid_to) if curr_item.memory.valid_to else "OPEN"
        )
        print(f"Query [Current at {format_datetime_utc(t_curr)}]:", file=stream)
        print(f"  Value               : {curr_item.memory.value_display}", file=stream)
        print(f"  Status              : {curr_item.memory.status.value}", file=stream)
        print(
            f"  Valid From          : {format_datetime_utc(curr_item.memory.valid_from)}",
            file=stream,
        )
        print(f"  Valid To            : {curr_vto}", file=stream)
        print(f"  Effective Confidence: {curr_item.effective_confidence:.4f}", file=stream)
        print(file=stream)

        hist_vto = (
            format_datetime_utc(hist_item.memory.valid_to) if hist_item.memory.valid_to else "OPEN"
        )
        print(f"Query [Historical as-of {format_datetime_utc(t_hist)}]:", file=stream)
        print(f"  Value               : {hist_item.memory.value_display}", file=stream)
        print(f"  Status              : {hist_item.memory.status.value}", file=stream)
        print(
            f"  Valid From          : {format_datetime_utc(hist_item.memory.valid_from)}",
            file=stream,
        )
        print(f"  Valid To            : {hist_vto}", file=stream)
        print(f"  Effective Confidence: {hist_item.effective_confidence:.4f}", file=stream)
        print(file=stream)

        if curr_item.memory.value_norm != "3.12" or hist_item.memory.value_norm != "3.10":
            print("[ERROR] Step 4 validation failed: incorrect values retrieved.", file=stream)
            return 1

        # ---------------------------------------------------------------------
        # STEP 5: Explainability and Provenance
        # ---------------------------------------------------------------------
        print("-" * 80, file=stream)
        print("STEP 5: Decision Explainability and Provenance", file=stream)
        print("-" * 80, file=stream)

        exp2 = service.explain(mem2.id)
        print(f"Explanation for Memory M{mem2.id} (Python 3.12):", file=stream)
        for note in exp2.notes:
            print(f"  - {note}", file=stream)

        print("  Decision History:", file=stream)
        for d in exp2.decisions:
            print(f"    * [{d.rule_id}] Action={d.action.value}: {d.explanation}", file=stream)

        print("  Connected Graph Edges:", file=stream)
        for e in exp2.edges:
            print(
                f"    * Edge: M{e.src_memory_id} --{e.type.value}--> M{e.dst_memory_id} "
                f"(reason: '{e.reason}')",
                file=stream,
            )
        print(file=stream)

        # ---------------------------------------------------------------------
        # STEP 6: Multi-Level Dependency Ingestion and Change Propagation
        # ---------------------------------------------------------------------
        print("-" * 80, file=stream)
        print("STEP 6: Dependency Setup and Downstream Propagation", file=stream)
        print("-" * 80, file=stream)

        # Ingest direct dependent: api_gateway
        msg_dep1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Deploying API Gateway on Server A",
            event_time=t1,
        )
        claim_dep1 = Claim(
            subject="Server A",
            attribute="runs_service",
            value="api_gateway",
            valid_from=t1,
            extract_confidence=0.95,
        )
        rep_dep1 = service.ingest_claims(msg_dep1, [claim_dep1], recorded_at=t1)
        m_dep1_id = rep_dep1.decisions[0].memory_id
        assert m_dep1_id is not None

        # Ingest transitive dependent: service_status
        msg_dep2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Service status is operational up",
            event_time=t1,
        )
        claim_dep2 = Claim(
            subject="Server A",
            attribute="service_status",
            value="up",
            valid_from=t1,
            extract_confidence=0.95,
        )
        rep_dep2 = service.ingest_claims(msg_dep2, [claim_dep2], recorded_at=t1)
        m_dep2_id = rep_dep2.decisions[0].memory_id
        assert m_dep2_id is not None

        # Establish dependency edges:
        # M_dep1 --DEPENDS_ON--> M2 (api_gateway depends on Python 3.12 runtime)
        # M_dep2 --DEPENDS_ON--> M_dep1 (service_status depends on api_gateway)
        service.repo.insert_edge(
            src_memory_id=m_dep1_id,
            dst_memory_id=mem2.id,
            edge_type=EdgeType.DEPENDS_ON,
            reason="API Gateway requires Python 3.12 runtime",
            created_at=t2,
        )
        service.repo.insert_edge(
            src_memory_id=m_dep2_id,
            dst_memory_id=m_dep1_id,
            edge_type=EdgeType.DEPENDS_ON,
            reason="Service status depends on API Gateway uptime",
            created_at=t2,
        )

        print(
            f"Configured dependency graph: M{m_dep2_id} (service_status) -> "
            f"M{m_dep1_id} (api_gateway) -> M{mem2.id} (python_version 3.12)",
            file=stream,
        )

        # Upstream event: Security advisory triggers review propagation
        t4 = datetime(2026, 6, 20, 0, 0, tzinfo=UTC)
        audit_msg_id = service.repo.insert_message(
            source_id="sysadmin_priya",
            text="Security scan alert: CVE-2026-4444 vulnerability advisory in runtime",
            event_time=t4,
        )
        cause_text = "CVE-2026-4444 runtime security advisory"
        prop_logs = service.propagate_change(
            changed_ids=[mem2.id],
            cause_text=cause_text,
            message_id=audit_msg_id,
            recorded_at=t4,
            trigger_mode="manual",
        )

        m_dep1_after_prop = service.repo.get_memory(m_dep1_id)
        m_dep2_after_prop = service.repo.get_memory(m_dep2_id)
        assert m_dep1_after_prop is not None and m_dep2_after_prop is not None

        print(f"Triggered upstream change propagation for M{mem2.id}: '{cause_text}'", file=stream)
        print(f"Propagation logs generated: {len(prop_logs)}", file=stream)
        for log in prop_logs:
            print(f"  - [{log.rule_id}] Target M{log.memory_id}: {log.explanation}", file=stream)

        print(
            f"  Memory #{m_dep1_id} needs_review={m_dep1_after_prop.needs_review} "
            f"(reason: '{m_dep1_after_prop.review_reason}')",
            file=stream,
        )
        print(
            f"  Memory #{m_dep2_id} needs_review={m_dep2_after_prop.needs_review} "
            f"(reason: '{m_dep2_after_prop.review_reason}')",
            file=stream,
        )
        print(file=stream)

        if not (m_dep1_after_prop.needs_review and m_dep2_after_prop.needs_review):
            print("[ERROR] Step 6 failed: dependents were not flagged for review.", file=stream)
            return 1

        # ---------------------------------------------------------------------
        # STEP 7: Downstream Impact Analysis
        # ---------------------------------------------------------------------
        print("-" * 80, file=stream)
        print("STEP 7: Downstream Impact Analysis (Blast Radius)", file=stream)
        print("-" * 80, file=stream)

        impact_items = service.impact(mem2.id)
        print(
            f"Downstream blast radius from M{mem2.id} ({len(impact_items)} affected memories):",
            file=stream,
        )
        for item in impact_items:
            m = item.memory.memory
            path_str = " -> ".join(f"M{x}" for x in item.path)
            print(
                f"  - Dependent M{m.id} [{m.attribute} = '{m.value_display}']",
                file=stream,
            )
            print(f"    Path        : {path_str}", file=stream)
            print(f"    Status      : {m.status.value}, needs_review={m.needs_review}", file=stream)
            print(f"    Explanation : {item.reason}", file=stream)

        print(file=stream)

        if len(impact_items) != 2:
            print(
                f"[ERROR] Step 7 failed: expected 2 impact items, got {len(impact_items)}.",
                file=stream,
            )
            return 1

        # ---------------------------------------------------------------------
        # STEP 8: Human Review Acknowledgement & Independent Isolation
        # ---------------------------------------------------------------------
        print("-" * 80, file=stream)
        print("STEP 8: Review Acknowledgement and Independent Isolation", file=stream)
        print("-" * 80, file=stream)

        t5 = datetime(2026, 6, 21, 0, 0, tzinfo=UTC)
        ack_log = service.acknowledge_review(
            m_dep1_id,
            recorded_at=t5,
            reviewed_by="SecOps_Alice",
            explanation="Verified CVE-2026-4444 patch applied and runtime is compatible",
        )

        m_dep1_ack = service.repo.get_memory(m_dep1_id)
        m_dep2_ack = service.repo.get_memory(m_dep2_id)
        assert m_dep1_ack is not None and m_dep2_ack is not None

        print(f"Review Acknowledged for M{m_dep1_id}:", file=stream)
        print(f"  Decision Rule       : {ack_log.rule_id}", file=stream)
        print(f"  Action              : {ack_log.action.value}", file=stream)
        print(f"  Reviewer            : {ack_log.claim_json.get('reviewed_by')}", file=stream)
        print(f"  Explanation         : {ack_log.explanation}", file=stream)
        print(
            f"  Needs Review State  : M{m_dep1_id}.needs_review = {m_dep1_ack.needs_review} "
            f"(cleared)",
            file=stream,
        )
        print(
            f"  Factual Data        : value='{m_dep1_ack.value_display}', "
            f"confidence={m_dep1_ack.confidence:.4f} (immutable)",
            file=stream,
        )
        print(
            f"  Downstream Isolation: M{m_dep2_id}.needs_review = {m_dep2_ack.needs_review} "
            f"(retains review flag)",
            file=stream,
        )
        print(file=stream)

        if m_dep1_ack.needs_review is not False or m_dep2_ack.needs_review is not True:
            print("[ERROR] Step 8 failed: review flags inconsistent.", file=stream)
            return 1

        # ---------------------------------------------------------------------
        # SUMMARY OF RESULTS
        # ---------------------------------------------------------------------
        print("=" * 80, file=stream)
        print("Summary of Results:", file=stream)
        print(f"  1. Initial Claim Ingestion : R-INSERT -> M{mem1.id} (Python 3.10)", file=stream)
        print(
            f"  2. Administrator Update    : R-REPLACE -> M{mem2.id} (Python 3.12) "
            f"supersedes M{mem1.id}",
            file=stream,
        )
        print(
            "  3. Low-Trust Intern Claim  : R-REJECT -> Action.REJECT "
            "(score 0.210 < threshold 0.428)",
            file=stream,
        )
        curr_time_str = format_datetime_utc(t_curr)
        hist_time_str = format_datetime_utc(t_hist)
        print(
            f"  4. Current Value           : Python 3.12 (ACTIVE, as-of {curr_time_str})",
            file=stream,
        )
        print(
            f"  5. Historical Value        : Python 3.10 (HISTORICAL, as-of {hist_time_str})",
            file=stream,
        )
        print(
            "  6. Decision Explainability : Fully audited with provenance and supersession graph",
            file=stream,
        )
        print(
            f"  7. Dependency Propagation  : Upstream change correctly flagged M{m_dep1_id} "
            f"and M{m_dep2_id}",
            file=stream,
        )
        print(
            f"  8. Blast Radius Impact     : Identified {len(impact_items)} dependents "
            "with exact paths",
            file=stream,
        )
        print(
            f"  9. Review Acknowledgement  : M{m_dep1_id} acknowledged; "
            f"M{m_dep2_id} remains under review",
            file=stream,
        )
        print("=" * 80, file=stream)
        print("Hero scenario demonstration completed successfully.", file=stream)
        return 0

    except Exception as exc:
        print(f"[ERROR] Exception during hero scenario execution: {exc}", file=stream)
        import traceback

        traceback.print_exc(file=stream)
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    sys.exit(run_demo())
