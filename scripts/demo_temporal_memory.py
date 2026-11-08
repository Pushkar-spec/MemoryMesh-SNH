"""CLI demonstration of structured temporal memory ingestion and retrieval.

Governed by docs/SPECIFICATION.md Section 2, docs/ARCHITECTURE.md, and docs/AGENTS.md.
Implements canonical task P1.3 (roadmap task T1.5).
Demonstrates:
  1. Ingestion of an initial structured claim (R-INSERT).
  2. Ingestion of a newer conflicting claim (R-REPLACE).
  3. Querying the current memory (Python 3.12).
  4. Querying a historical memory as-of March 15, 2026 (Python 3.10).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import Any

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.models import Claim, IngestMessage, format_datetime_utc
from memorymesh.service import MemoryService


def run_demo(stream: Any = sys.stdout) -> int:
    """Execute the structured temporal memory flow and print deterministic results."""
    conn = None
    try:
        # Initialize an isolated in-memory SQLite database
        conn = init_db(":memory:")
        cfg = load_app_config()
        service = MemoryService(db_target=conn, cfg=cfg)

        print("=" * 80, file=stream)
        print("MemoryMesh: Structured Temporal Memory Demonstration (P1.3)", file=stream)
        print("=" * 80, file=stream)
        print("Isolated database initialized: SQLite (:memory:)", file=stream)
        print("Configuration catalog loaded from engine, sources, and attributes.", file=stream)
        print(file=stream)

        # ---------------------------------------------------------------------
        # STEP 1: First Claim
        # ---------------------------------------------------------------------
        t1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
        t1_str = format_datetime_utc(t1)
        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs Python 3.10",
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
        print("STEP 1: Ingest Initial Claim", file=stream)
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
        mem1 = service.repo.get_memory(dec1.memory_id) if dec1.memory_id else None
        if mem1 is None:
            print("[ERROR] Failed to retrieve persisted memory 1.", file=stream)
            return 1

        mem1_vfrom = format_datetime_utc(mem1.valid_from)
        mem1_vto = format_datetime_utc(mem1.valid_to) if mem1.valid_to else "OPEN"

        print("Resolution Decision:", file=stream)
        print(f"  Action      : {dec1.action.value}", file=stream)
        print(f"  Rule ID     : {dec1.rule_id}", file=stream)
        print(
            f'  Memory ID   : #{mem1.id} (status: {mem1.status.value}, value: "{mem1.value_norm}")',
            file=stream,
        )
        print(f"  Validity    : [{mem1_vfrom}, {mem1_vto})", file=stream)
        print(f"  Explanation : {dec1.explanation}", file=stream)
        print(file=stream)

        if dec1.rule_id != "R-INSERT":
            print(
                f"[ERROR] Step 1 validation failed: expected 'R-INSERT', got '{dec1.rule_id}'.",
                file=stream,
            )
            return 1

        # ---------------------------------------------------------------------
        # STEP 2: Second (Conflicting) Claim
        # ---------------------------------------------------------------------
        t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        t2_str = format_datetime_utc(t2)
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A migrated to Python 3.12",
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
        print("STEP 2: Ingest Newer Conflicting Claim", file=stream)
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
        mem2 = service.repo.get_memory(dec2.memory_id) if dec2.memory_id else None
        if mem2 is None:
            print("[ERROR] Failed to retrieve persisted memory 2.", file=stream)
            return 1

        mem1_updated = service.repo.get_memory(mem1.id)
        if mem1_updated is None:
            print("[ERROR] Failed to retrieve updated memory 1.", file=stream)
            return 1

        mem2_vfrom = format_datetime_utc(mem2.valid_from)
        mem2_vto = format_datetime_utc(mem2.valid_to) if mem2.valid_to else "OPEN"
        mem1_vto_updated = (
            format_datetime_utc(mem1_updated.valid_to) if mem1_updated.valid_to else "None"
        )

        print("Resolution Decision:", file=stream)
        print(f"  Action      : {dec2.action.value}", file=stream)
        print(f"  Rule ID     : {dec2.rule_id}", file=stream)
        print(
            f'  New Memory  : #{mem2.id} (status: {mem2.status.value}, value: "{mem2.value_norm}")',
            file=stream,
        )
        print(f"  Validity    : [{mem2_vfrom}, {mem2_vto})", file=stream)
        print(
            f"  Closed Old  : #{mem1_updated.id} (status: {mem1_updated.status.value}, "
            f"valid_to: {mem1_vto_updated})",
            file=stream,
        )
        print(f"  Explanation : {dec2.explanation}", file=stream)
        print(file=stream)

        if dec2.rule_id != "R-REPLACE":
            print(
                f"[ERROR] Step 2 validation failed: expected 'R-REPLACE', got '{dec2.rule_id}'.",
                file=stream,
            )
            return 1

        # ---------------------------------------------------------------------
        # STEP 3: Current Retrieval
        # ---------------------------------------------------------------------
        t_now = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
        t_now_str = format_datetime_utc(t_now)
        print("-" * 80, file=stream)
        print(f"STEP 3: Query Current Memory (reference time: {t_now_str})", file=stream)
        print("-" * 80, file=stream)

        cur_result = service.current("Server A", "python_version", now=t_now)
        if not cur_result.items:
            print("[ERROR] No current memory found.", file=stream)
            return 1

        active_mem = cur_result.items[0].memory
        act_vfrom = format_datetime_utc(active_mem.valid_from)
        act_vto = (
            format_datetime_utc(active_mem.valid_to) if active_mem.valid_to else "None (open-ended)"
        )

        print("Query: Current python_version for Server A", file=stream)
        print(f"  Value               : {active_mem.value_norm}", file=stream)
        print(f"  Status              : {active_mem.status.value}", file=stream)
        print(f"  Valid From          : {act_vfrom}", file=stream)
        print(f"  Valid To            : {act_vto}", file=stream)
        print(f"  Base Confidence     : {active_mem.confidence}", file=stream)
        print(
            f"  Effective Confidence: {cur_result.items[0].effective_confidence}",
            file=stream,
        )
        print(file=stream)

        if active_mem.value_norm != "3.12":
            print(
                f"[ERROR] Step 3 validation failed: expected current version '3.12', "
                f"got '{active_mem.value_norm}'.",
                file=stream,
            )
            return 1

        # ---------------------------------------------------------------------
        # STEP 4: Historical Retrieval
        # ---------------------------------------------------------------------
        t_hist = datetime(2026, 3, 15, 0, 0, tzinfo=UTC)
        t_hist_str = format_datetime_utc(t_hist)
        print("-" * 80, file=stream)
        print(f"STEP 4: Query Historical Memory As-Of {t_hist_str}", file=stream)
        print("-" * 80, file=stream)

        hist_result = service.current("Server A", "python_version", valid_at=t_hist, now=t_now)
        if not hist_result.items:
            print("[ERROR] No historical memory found for requested timestamp.", file=stream)
            return 1

        hist_mem = hist_result.items[0].memory
        hist_vfrom = format_datetime_utc(hist_mem.valid_from)
        hist_vto = format_datetime_utc(hist_mem.valid_to) if hist_mem.valid_to else "None"

        print(f"Query: python_version for Server A valid on {t_hist_str}", file=stream)
        print(f"  Value               : {hist_mem.value_norm}", file=stream)
        print(f"  Status              : {hist_mem.status.value}", file=stream)
        print(f"  Valid From          : {hist_vfrom}", file=stream)
        print(f"  Valid To            : {hist_vto}", file=stream)
        print(f"  Base Confidence     : {hist_mem.confidence}", file=stream)
        print(
            f"  Effective Confidence: {hist_result.items[0].effective_confidence}",
            file=stream,
        )
        print(file=stream)

        if hist_mem.value_norm != "3.10":
            print(
                f"[ERROR] Step 4 validation failed: expected historical version '3.10', "
                f"got '{hist_mem.value_norm}'.",
                file=stream,
            )
            return 1

        # ---------------------------------------------------------------------
        # Summary & Final Runtime Validation
        # ---------------------------------------------------------------------
        print("=" * 80, file=stream)
        print("Summary of Results:", file=stream)
        print(
            f"  Initial Decision   : {dec1.rule_id} -> Memory #{mem1.id} "
            f"(Python {mem1.value_norm})",
            file=stream,
        )
        print(
            f"  Conflict Decision  : {dec2.rule_id} -> Memory #{mem2.id} "
            f"(Python {mem2.value_norm}) supersedes #{mem1.id}",
            file=stream,
        )
        print(
            f"  Current Version    : Python {active_mem.value_norm} (expected: 3.12)",
            file=stream,
        )
        print(f"  As-Of March 15     : Python {hist_mem.value_norm} (expected: 3.10)", file=stream)
        print("=" * 80, file=stream)

        # Final invariant validation check
        if (
            dec1.rule_id != "R-INSERT"
            or dec2.rule_id != "R-REPLACE"
            or active_mem.value_norm != "3.12"
            or hist_mem.value_norm != "3.10"
        ):
            print("[ERROR] Runtime result validation failed.", file=stream)
            return 1

        print("Demonstration completed successfully.", file=stream)
        return 0

    except Exception as exc:
        print(f"[FATAL] CLI demonstration encountered an error: {exc}", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    sys.exit(run_demo())
