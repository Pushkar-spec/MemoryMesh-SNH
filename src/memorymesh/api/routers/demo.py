"""SSE scenario replay routes for MemoryMesh.

Governed by docs/SPECIFICATION.md Section 7.2 and docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from fastapi import APIRouter, Depends, Query, Request
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from memorymesh.api.deps import get_service
from memorymesh.models import Claim, IngestMessage
from memorymesh.service import MemoryService

router = APIRouter(tags=["demo"])


class ReplayLock:
    """Atomic non-blocking lock to guarantee a single active replay session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Atomically attempt to acquire the lock without blocking.

        Returns True if acquired; False if another replay session already holds it.
        """
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        """Release the lock if currently held."""
        if self._lock.locked():
            try:
                self._lock.release()
            except RuntimeError:
                pass

    @property
    def is_locked(self) -> bool:
        """Return True if the replay lock is currently held."""
        return self._lock.locked()


# Session-level atomic concurrency lock to prevent overlapping replay sessions
_replay_lock = ReplayLock()


def _find_scenarios_dir() -> Path:
    """Locate the evaluation scenarios directory."""
    cwd_scenarios = Path("eval/scenarios")
    if cwd_scenarios.is_dir():
        return cwd_scenarios.resolve()
    module_scenarios = Path(__file__).resolve().parents[4] / "eval" / "scenarios"
    if module_scenarios.is_dir():
        return module_scenarios.resolve()
    return cwd_scenarios.resolve()


def _load_scenario(scenarios_dir: Path, target_id: str) -> dict[str, Any] | None:
    """Load a specific scenario definition from YAML files by ID or filename."""
    if not scenarios_dir.is_dir():
        return None
    for file_path in sorted(scenarios_dir.glob("*.yaml")) + sorted(scenarios_dir.glob("*.yml")):
        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                sc_id = str(data.get("id", ""))
                if target_id in (sc_id, file_path.name, file_path.stem):
                    data["_file_name"] = file_path.name
                    return data
        except Exception:
            continue
    return None


@router.get(
    "/demo/replay",
    response_class=EventSourceResponse,
    summary="Stream scenario replay decisions via SSE",
)
async def replay_scenario(
    request: Request,
    service: Annotated[MemoryService, Depends(get_service)],
    scenario: str = Query(default="hero-001", description="Scenario ID or filename"),
    mode: Literal["structured", "e2e"] = Query(
        default="structured", description="Replay execution mode"
    ),
    delay_ms: int = Query(default=800, ge=0, description="Delay between messages in milliseconds"),
    reset: bool = Query(
        default=False,
        deprecated=True,
        description=(
            "Deprecated legacy parameter. GET replay is strictly read-only and never "
            "resets data. Use POST /api/demo/reset to perform database reset."
        ),
    ),
) -> EventSourceResponse:
    """Stream scenario replay decisions sequentially as Server-Sent Events (SSE).

    Emits:
      - `message_ingested`: IngestReport JSON payload for each processed message.
      - `done`: Final summary payload {"messages": int} once replay finishes.
      - `error`: Error payload {"error": str} on failure, missing scenario, or collision.
    """

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        if not _replay_lock.try_acquire():
            yield ServerSentEvent(
                event="error",
                data=json.dumps(
                    {
                        "error": (
                            "A replay session is already active. Concurrent replays are blocked."
                        )
                    }
                ),
            )
            return

        try:
            scenarios_dir = _find_scenarios_dir()
            scenario_data = _load_scenario(scenarios_dir, scenario)
            if scenario_data is None:
                yield ServerSentEvent(
                    event="error",
                    data=json.dumps({"error": f"Scenario '{scenario}' not found."}),
                )
                return

            messages: list[dict[str, Any]] = scenario_data.get("messages", [])
            structured_claims: dict[str, Any] = scenario_data.get("structured_claims", {})
            total_messages = 0

            for msg in messages:
                if await request.is_disconnected():
                    break

                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000.0)

                if await request.is_disconnected():
                    break

                msg_id = str(msg.get("id", ""))
                source_id = str(msg.get("source", "unknown"))
                t_raw = msg.get("t")
                if t_raw:
                    try:
                        event_time = datetime.fromisoformat(str(t_raw))
                    except Exception:
                        event_time = datetime.now(UTC)
                else:
                    event_time = datetime.now(UTC)

                text = str(msg.get("text", ""))
                im = IngestMessage(source_id=source_id, event_time=event_time, text=text)

                if mode == "structured" and structured_claims:
                    raw_claims = structured_claims.get(msg_id, [])
                    claims = [Claim.model_validate(c) for c in raw_claims]
                    report = service.ingest_claims(message=im, claims=claims)
                else:
                    report = service.ingest(message=im)

                total_messages += 1
                yield ServerSentEvent(
                    event="message_ingested",
                    data=report.model_dump_json(),
                )

            yield ServerSentEvent(
                event="done",
                data=json.dumps({"messages": total_messages}),
            )
        except Exception as exc:
            yield ServerSentEvent(
                event="error",
                data=json.dumps({"error": str(exc)}),
            )
        finally:
            _replay_lock.release()

    return EventSourceResponse(event_generator())
