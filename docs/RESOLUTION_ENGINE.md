# MemoryMesh: Resolution Engine Specification

**This document governs behavior.** If code and this document disagree, the code is wrong (or this document must be updated in the same commit).

The engine is the heart of the project. It is a **pure function**:

```python
def resolve(
    rc: ResolvedClaim,
    chain: list[Memory],            # all non-rejected memories for (entity, attribute, context_key), by valid_from
    schema: AttributeDef,
    cfg: EngineConfig,
    classify: Callable[[ResolvedClaim, Memory], Relation],   # injected; LLM-backed only for text values
    dep_targets: list[int],         # memory IDs the claim depends on (resolved by pipeline)
    recorded_at: datetime,          # ingestion time (becomes recorded_at / closed_at)
) -> Decision: ...
```

---

## 1. Core Concepts

| Term | Meaning |
| :--- | :--- |
| **Chain** | The ordered history of one fact slot: `(entity, attribute, context_key)` |
| **Covers(m, t)** | `m.valid_from <= t` and (`m.valid_to` is null or `t < m.valid_to`) |
| **Live** | Status `active` or `disputed` |
| **Strength** | `reliability(source) * extract_confidence` of one piece of evidence |
| **Score** | Effective confidence of a memory, or the strength of the new claim |

---

## 2. Formulas (`engine/confidence.py`)

```text
strength(e)           = source.reliability * claim.extract_confidence

base_confidence(m)    = min(max_confidence,
                            1 - PRODUCT over DISTINCT sources s of (1 - max strength of s's evidence))

decay(m, at)          = 0.5 ^ (age_days / half_life_days)     if attribute has half_life_days
                      = 1                                      otherwise
                        where age_days = max(0, (at - m.valid_from) in days)

effective_conf(m, at) = base_confidence(m) * decay(m, at) * (review_penalty if m.needs_review else 1)

new_score             = strength(new claim)
old_score             = max over live covering memories of effective_conf(m, rc.valid_from)
```

Rules for evidence:
- Evidence from the **same source** on the same memory does **not** increase confidence (a source cannot corroborate itself). It is still recorded, and the decision is `R-DUP-SAME-SOURCE`.
- `base_confidence` is **recomputed** from `evidence` rows after any evidence change. It is never incremented in place.

---

## 3. Relation Classification (`engine/classifier.py`)

Compares a new claim with one existing memory in the same chain.

| Value type | Rule |
| :--- | :--- |
| any | `value_norm` equal -> `IDENTICAL` |
| `version` | new starts with `old + "."` -> `REFINES` (e.g. `3.12.4` vs `3.12`) |
| `version` | old starts with `new + "."` -> `IDENTICAL` (new is less specific but compatible) |
| `version`, `enum`, `number`, `date` | otherwise single-valued -> `CONFLICTS`; multi-valued -> `COMPLEMENTARY` |
| `text`, single-valued | fuzzy ratio < `text_duplicate_candidate` -> `CONFLICTS`; else ask LLM (`classify_relation`) |
| `text`, multi-valued | fuzzy ratio < `text_duplicate_candidate` -> `COMPLEMENTARY`; else ask LLM |
| `text`, LLM unavailable | single -> `CONFLICTS`; multi -> `COMPLEMENTARY` (explanation notes this) |

Context is **not** compared here. Different contexts never meet, because chains are keyed by `context_key`.

---

## 4. Decision Procedure

Rules are evaluated **in this order**; the first match decides. Every outcome returns a `Decision` with a `rule_id`, numeric `scores`, and a human-readable `explanation`.

```text
resolve(rc, chain, schema, cfg, classify, dep_targets, recorded_at):

  s_new = strength(rc)                         # 1.0-based if ablation.use_trust is false

  R-SKIP-LOWCONF   if rc.claim.extract_confidence < min_extract_confidence -> SKIP
  (R-SKIP-ATTR / R-SKIP-VALUE / R-SKIP-LLMFAIL are produced earlier by the pipeline, same log format)

  if ablation.use_temporal is false:           -> R-INSERT always (append-only baseline)

  if rc.polarity == RETRACT:                   -> R-RETRACT (section 4.4)

  if chain is empty:                           -> R-INSERT

  if schema.cardinality == MULTI (and ablation.use_cardinality): -> multi path (4.2)

  single-valued path (4.1)
```

### 4.1 Single-valued path

```text
cover = [m in chain if covers(m, rc.valid_from)]
live  = [m in cover if m.status in (ACTIVE, DISPUTED)]
hist  = [m in cover if m.status == HISTORICAL]

A. Agreement with a live memory
   for m in live:
       rel = classify(rc, m)
       IDENTICAL -> if rc.source already in m.evidence sources:  R-DUP-SAME-SOURCE (add evidence row, no confidence change)
                    else:                                         R-CORROBORATE   (add evidence, recompute confidence)
       REFINES   -> R-REFINE
                    new memory ACTIVE, valid_from = rc.valid_from;
                    old: status HISTORICAL, valid_to = rc.valid_from, closed_at = recorded_at;
                    edge refines (new -> old). Dependents are NOT flagged.

B. Conflict with live memories
   old_score = max effective_conf over live
   newer     = all(rc.valid_from > m.valid_from for m in live)
   margin    = replace_margin
   if rc.value_norm equals the value of any HISTORICAL memory in chain that was superseded:
       margin = revert_margin            # guard against stale sources reverting to old values (default 1.00)

   R-REPLACE  if newer AND s_new >= old_score * margin
              new memory ACTIVE (valid_from = rc.valid_from);
              every live m: status HISTORICAL, valid_to = rc.valid_from, closed_at = recorded_at;
              edges supersedes (new -> m); affected_memory_ids = [live ids]
   R-REJECT   elif s_new < old_score * reject_ratio
              new memory stored with status REJECTED (kept for audit); edge contradicts (new -> live); no change to live
   R-DISPUTE  else
              new memory DISPUTED; every live m -> DISPUTED; edges contradicts (new -> m);
              affected_memory_ids = [live ids]

   (When not `newer`, i.e. equal valid_from, REPLACE is not allowed: outcome is REJECT or DISPUTE.)

C. Conflict with history only (new claim reaches back into a closed window)
   if hist and not live:
       rel = classify(rc, hist[0])
       IDENTICAL/REFINES -> R-CORROBORATE the historical memory (adds evidence)
       CONFLICTS         -> R-REJECT if s_new < hist_score * reject_ratio else R-DISPUTE
       (the engine never rewrites history automatically)

D. No memory covers rc.valid_from
   later = first m in chain with m.valid_from > rc.valid_from
   if later is None:  R-INSERT   (open-ended ACTIVE, previous windows already closed)
   else:              R-BACKFILL
        gate: if s_new < score(later) * reject_ratio -> R-REJECT
        else new memory HISTORICAL, valid_to = rc.valid_to or later.valid_from, closed_at = recorded_at
```

### 4.2 Multi-valued path

```text
same = [m in chain if classify(rc, m) == IDENTICAL]
live_same = [m in same if m.status in (ACTIVE, DISPUTED) and covers(m, rc.valid_from)]

if live_same: R-CORROBORATE (or R-DUP-SAME-SOURCE) on live_same[0]
else:         R-COEXIST    new memory ACTIVE; nothing else changes
```
Multi-valued memories never conflict, so they are never replaced, rejected, or disputed. They end only through `RETRACT`.

### 4.3 Insert behavior (`R-INSERT`, `R-COEXIST`, `R-BACKFILL`)
- Confidence = `strength` (single evidence row).
- `recorded_at = recorded_at` parameter; `closed_at` set only if `valid_to` is set.
- For every `dep_targets` ID: `AddEdge(src="$new", dst=target, type=DEPENDS_ON)` (not for REJECT).

### 4.4 Retract (`R-RETRACT`)
Claim: "Server A no longer runs service Y." (`polarity = retract`).
```text
target = live memory with IDENTICAL value that covers rc.valid_from
if none:  SKIP (rule R-SKIP-NOTARGET, explanation "nothing to retract")
else:     if s_new >= effective_conf(target) * reject_ratio:
              target -> HISTORICAL, valid_to = rc.valid_from, closed_at = recorded_at; affected = [target]
          else: R-REJECT-RETRACT (retraction too weak; logged, no change)
```

### 4.5 Dispute resolution by a human (`R-DISPUTE-RESOLVE`)
`MemoryService.resolve_dispute(winner_id)` explicitly resolves an active dispute in favor of a designated winner memory.

### Dispute Identification
- Active disputes are identified as connected components in the undirected graph formed by `contradicts` edges where all member memories have `status = 'disputed'` and share identical `entity_id`, `attribute`, and canonical `context_key`.
- Requests specifying non-existent, non-disputed, isolated, or already-resolved memories are rejected.

### Status Transitions & Validity Boundaries
- Winner memory -> `ACTIVE`.
- Each losing member of the dispute group:
  - If the loser is older than the winner (`loser.valid_from < winner.valid_from`):
    - `loser.status = HISTORICAL`
    - `loser.valid_to = winner.valid_from`
    - `loser.closed_at = eff_recorded_at`
    - An edge `src=winner.id, dst=loser.id, type='supersedes'` is inserted.
  - If the loser is contemporaneous or newer (`loser.valid_from >= winner.valid_from`):
    - `loser.status = REJECTED`
    - `loser.closed_at = eff_recorded_at`
- All original evidence records and earlier decision logs remain intact.

### Auditability & Atomicity
- An independent audit record is added to `decision_log` with `rule_id = "R-DISPUTE-RESOLVE"` and `action = Action.DISPUTE`. The original `R-DISPUTE` ingestion decisions are preserved.
- All mutations (winner activation, loser updates, graph edges, and decision log) are executed atomically in a single database transaction. Any failure rolls back the entire resolution.

### Retrieval Behavior
- **Before resolution:** Unresolved disputes populate `QueryResult.disputes` with structured `DisputeView(a, b, edge_reason)`. `get_current()` safely returns `None` to avoid serving disputed facts as authoritative.
- **After resolution:** Resolved disputes no longer appear in `service.disputes()` or `QueryResult.disputes`. `get_current()` safely returns the winner as the single active authoritative memory, while historical queries (`get_as_of`) return the older loser for its validity window.

### 4.6 Stretch: dispute auto-resolution (S3)
After any `R-CORROBORATE` on a `DISPUTED` memory, recompute both sides. If `loser_conf < winner_conf * dispute_autoresolve_ratio`, apply 4.5 automatically and log `R-DISPUTE-AUTO`.

---

## 5. Dependency Propagation (`engine/propagate.py`)

Runs **after** a Decision is applied, when `affected_memory_ids` is non-empty and `ablation.use_propagation` is true.

```text
propagate(graph, changed_ids, cause_text):
    flagged = []
    queue = [(id, [id]) for id in changed_ids]
    visited = set(changed_ids)
    while queue:
        node, path = queue.pop(0)
        for dep in graph.predecessors(node) via edges type=depends_on:   # dependents point AT foundations
            if dep in visited: continue
            visited.add(dep)
            if dep.status in (ACTIVE, DISPUTED) and not already flagged:
                emit UpdateMemory(dep, needs_review=True,
                                  review_reason=f"depends on M{node}, which was {cause_text}")
                flagged.append((dep, path + [dep]))
            queue.append((dep, path + [dep]))      # transitive, cycle-safe via visited
    return mutations, flagged                       # flagged = the "blast radius"
```

### 5.1 Direction and Semantics
- **Convention:** If memory B depends on memory A: `B --DEPENDS_ON--> A`.
  - `src_memory_id = B` (dependent / downstream).
  - `dst_memory_id = A` (foundation / upstream).
- **Direct resolution:**
  - `service.get_dependencies(memory_id)` returns upstream foundations (`src=memory_id`).
  - `service.get_dependents(memory_id)` returns downstream dependents (`dst=memory_id`).
- **Transitive resolution:**
  - `service.get_transitive_dependents(memory_id)` resolves all downstream dependents using cycle-safe BFS, returning `(dependent_memory, path)`.
  - Cycles, self-referential edges, and diamond duplicates are handled safely without infinite loops or duplicate results.

### 5.2 Propagation Rules
- Triggered by `R-REPLACE`, `R-DISPUTE`, `R-RETRACT`, and `R-DISPUTE-RESOLVE` (for losers).
- **Not** triggered by `R-REFINE`, `R-CORROBORATE`, `R-REJECT`, or `R-COEXIST`.
- **What Propagation Does:**
  - Identifies active or disputed downstream memories reachable via `DEPENDS_ON`.
  - Sets `needs_review = True` and records an explainable `review_reason` referencing the original changed root memory and cause (transitive dependents do not falsely claim their intermediate predecessors changed).
  - Traverses each changed root independently; different roots reaching the same dependent produce separate audit records, while diamond paths from the same root are deduplicated.
  - Generates an audit `DecisionLogEntry` with `rule_id = "R-PROPAGATE"`, `action = Action.PROPAGATE`, recording the original root upstream ID, dependency path, and cause.
  - Lowers effective confidence dynamically via `review_penalty` during retrieval (without mutating stored base confidence).
  - Event-level idempotency: A propagation event identity is defined as `(memory_id, upstream_id, message_id, trigger_mode, event_id/cause)`. Repeating the identical propagation event does not duplicate decision log entries. Distinct events, different roots, or different trigger modes (e.g. automatic pipeline vs manual maintenance invocation) generate separate audit records in `decision_log` and preserve multiple active reasons in `review_reason`.
  - Attribution: automatic propagation (from ingestion or dispute resolution) is marked `trigger_mode = "automatic"`; manual propagation is marked `trigger_mode = "manual"` (even when an audit message is supplied) and attributes to the changed root foundation message if no message is provided.
  - Runs atomically inside the triggering operation's transaction. All explicit message validations, graph reads, review-state mutations, and decision-log insertions execute inside `self.repo.transaction()`.
- **What Propagation Deliberately Does NOT Do:**
  - Does NOT overwrite or mutate factual values (`value_norm`, `value_display`).
  - Does NOT auto-delete, auto-reject, or auto-retract dependent memories.
  - Does NOT modify stored base confidence or supporting evidence records.
  - Does NOT fabricate new claims.

---

## 6. Retrieval Semantics (`retrieval/`)

### 6.1 Scope matching and specificity
A memory with context `C_m` **matches** a query with context `C_q` if `C_m ⊆ C_q` (every key/value of the memory appears in the query). Global memories (`{}`) match every query.

Among matching memories for the same `(entity, attribute)`, group by `len(C_m)`. The **highest** specificity group that has at least one memory valid at `valid_at` is the answer (`scope_match = "exact"`); lower-specificity matches are returned as inherited fallback (`scope_match = "inherited_global"`) only if no more specific match exists.

Example: query `{environment: staging}` returns the staging memory (3.11), not the global one (3.12). Query `{}` returns the global one.

### 6.2 Ranking and flags
- Items sorted by `effective_conf` descending, then `valid_from` descending.
- `disputed` memories are always returned **together** with their counterpart in `QueryResult.disputes`; the composer must state the disagreement.
- `needs_review` and `is_stale` are surfaced on every item.
- `rejected` memories are returned only for `history` and `why` intents, under `rejected_claims`.

### 6.3 Why and Impact
- **Why (`service.explain(memory_id)`):** Answers "Why does this memory exist in its current state?" deterministically and without mutation:
  - Memory view with effective confidence evaluated at current reference time (including decay and review penalties).
  - Evidence records supporting the memory with sources and strengths.
  - Decision log entries where `memory_id` matches or is contained in `affected_memory_ids`, sorted chronologically (`created_at ASC, id ASC`).
  - Graph relationship edges (`supersedes`, `contradicts`, `refines`, `depends_on`).
  - Active dispute pairs if the memory is currently under dispute.
  - Explanatory notes summarizing state, confidence, evidence sources, rule history, and review flags.
- **Decision Log Retrieval (`service.get_decision_logs(memory_id, limit, offset)`):** Provides chronological audit history. When `memory_id` is supplied, matches records where the memory was either the primary subject or an affected member, with optional limit and pagination offset.
- **Invariant Checks (`service.check_invariants()`):** Strictly read-only audit procedure returning structured `InvariantReport` (with `total_checked` reflecting the count of domain records across memories, messages, edges, decisions, and evidence evaluated) verifying 8 core invariants:
  1. `INV-CLOSED-AT-REQUIRED`: Memories with `valid_to` must have `closed_at` set.
  2. `INV-VALID-INTERVAL`: Valid interval must satisfy `valid_to > valid_from`.
  3. `INV-NO-ACTIVE-OVERLAP`: Single-valued attributes have non-overlapping active validity intervals for the same entity, attribute, and context.
  4. `INV-REJECTED-EXCLUDED`: Rejected memories are excluded from normal current-state and as-of retrieval.
  5. `INV-CONFIDENCE-CONSISTENT`: Stored base confidence matches noisy-OR recomputation from supporting evidence within 1e-4 tolerance.
  6. `INV-IDEMPOTENT-INGEST`: Passive duplicate-key consistency across committed messages (behavioral idempotency—that duplicate ingestion creates no additional memories and returns duplicate=True—is verified dynamically by the test suite).
  7. `INV-FOREIGN-KEY-INTEGRITY`: Edges, decision log references, and evidence records point to existing memory records.
  8. `INV-SOURCE-ATTRIBUTE-INTEGRITY`: References to entities, attribute definitions, and sources exist in the database catalogs.
- **Impact:** `propagate`-style traversal without writing, returning `ImpactItem` with the dependency path.

---

## 7. Ablation Flags (used by the evaluation, see `EVALUATION_AND_DEMO.md`)

| Flag off | Behavior | What it should break |
| :--- | :--- | :--- |
| `use_temporal` | Everything is inserted as active; no windows closed; retrieval returns all non-rejected memories | Current-state and as-of accuracy |
| `use_trust` | Every source has reliability 1.0 | Low-trust rejection, dispute detection |
| `use_cardinality` | Every attribute treated as single-valued | Multi-valued attributes get overwritten |
| `use_context` | `context_key` forced to `""` | Staging vs production facts collide |
| `use_propagation` | No dependency flagging | Impact recall |

---

## 8. Worked Example: The Hero Scenario

Assumed extractor confidence: `0.95` for direct statements, `0.50` for hedged ones ("I think ...?").

| # | Message (source, time) | Slot | Computation | Outcome |
| :-: | :--- | :--- | :--- | :--- |
| m1 | "Server A runs Python 3.10." (priya 0.90, 10 Jan) | python_version | empty chain | `R-INSERT`, conf = 0.90*0.95 = **0.855** |
| m2 | "Runtime for Server A: Python 3.10" (wiki 0.75, 2 Feb) | python_version | IDENTICAL, different source; strength 0.7125; conf = 1 - (1-0.855)(1-0.7125) = **0.958** | `R-CORROBORATE` |
| m3 | "Server A deploys happen on Fridays." (priya, 10 Feb) | deploy_day | empty chain | `R-INSERT`, conf 0.855 |
| m4 | "Library X is deployed on Server A and works fine with its Python 3.10 runtime." (priya, 5 Mar) | compatibility_note (Library X) | empty chain; `dep_targets = [M1]` | `R-INSERT` + `depends_on` edge to M1 |
| m5 | "We migrated Server A to Python 3.12 today." (priya, 1 Jun) | python_version | conflict; newer; s_new = 0.855; old_score = 0.958; threshold = 0.958*0.80 = 0.766; 0.855 >= 0.766 | `R-REPLACE`: M1 -> historical (valid_to 1 Jun); M4 flagged `needs_review` (effective conf 0.855*0.5 = 0.4275) |
| m6 | "I think Server A is still on Python 3.9?" (intern 0.30, 3 Jun) | python_version | conflict; s_new = 0.30*0.50 = 0.15; old_score = 0.855; reject threshold = 0.855*0.50 = 0.4275; 0.15 < 0.4275 | `R-REJECT` (stored `rejected`, `contradicts` edge) |
| m7 | "FYI deploys for Server A are on Mondays now." (dev_chat 0.50, 15 Jun) | deploy_day | conflict; s_new = 0.50*0.95 = 0.475; old 0.855; replace threshold 0.684; reject threshold 0.4275; 0.4275 <= 0.475 < 0.684 | `R-DISPUTE` (both `disputed`) |
| m8 | "In staging, Server A still runs Python 3.11." (dev_chat, 20 Jun) | python_version, context `environment=staging` | different `context_key` -> empty chain | `R-INSERT` (separate scope) |

**Expected answers (query time 25 Jun 2026):**
- Python now (global) -> `3.12`, conf 0.855, source priya.
- Python on 15 Mar -> `3.10`, conf 0.958, sources priya + wiki.
- Why 3.12? -> evidence m5, decision `R-REPLACE` with scores, supersedes M1.
- What may break after the migration? -> M4 "Library X works with 3.10" flagged.
- Deploy day -> **disputed**: friday (priya) vs monday (dev_chat).
- Python in staging -> `3.11`.
- Was 3.9 ever the version? -> No. One low-trust claim was rejected (`R-REJECT`).

---

## 9. Edge Cases & Required Behavior

| Case | Required behavior |
| :--- | :--- |
| Same message ingested twice | No-op, `IngestReport.duplicate = True` |
| Stale source repeats an old value after a migration | Treated as conflict with the active memory; `revert_margin` makes REPLACE harder; typically `R-DISPUTE` or `R-REJECT` |
| Late message with an earlier effective date and an empty slot before an existing memory | `R-BACKFILL` as historical |
| Late message contradicting a closed historical window | `R-REJECT` or `R-DISPUTE`; never rewrites history |
| Equal `valid_from`, equal trust | Not REPLACE; `R-DISPUTE` |
| Claim for an unknown attribute | `R-SKIP-ATTR`; logged; no memory created |
| Hedged claim (`extract_confidence < 0.40`) | `R-SKIP-LOWCONF`; logged |
| Claim with a version value the normalizer cannot parse | `R-SKIP-VALUE` |
| Dependency target not found | Claim inserted without edge; explanation notes "dependency target not found" |
| Dependency cycle | Propagation is cycle-safe (visited set) |
| Source missing from `sources.yaml` | Reject ingestion with a typed error (`UnknownSourceError`) |
| Known limitation | A genuinely correct claim from a low-trust source is rejected. This is by design and shown in the evaluation as a designed-to-lose case |

---

## 10. Rule ID Registry

| Rule ID | Action | Summary |
| :--- | :--- | :--- |
| `R-SKIP-LOWCONF` | SKIP | Extraction confidence below minimum |
| `R-SKIP-ATTR` | SKIP | Attribute not in catalog |
| `R-SKIP-VALUE` | SKIP | Value cannot be normalized |
| `R-SKIP-LLMFAIL` | SKIP | LLM output invalid after retry |
| `R-SKIP-NOTARGET` | SKIP | Retraction with nothing to retract |
| `R-INSERT` | INSERT | First claim in slot, or open-ended after closed windows |
| `R-CORROBORATE` | CORROBORATE | Same value from a different source: confidence rises |
| `R-DUP-SAME-SOURCE` | CORROBORATE | Same value, same source: evidence logged, confidence unchanged |
| `R-REFINE` | REFINE | More specific compatible value |
| `R-COEXIST` | COEXIST | Multi-valued new value |
| `R-REPLACE` | REPLACE | Newer, sufficiently trusted conflicting claim |
| `R-BACKFILL` | BACKFILL | Older fact filling an earlier gap |
| `R-REJECT` | REJECT | Conflicting claim far weaker than existing |
| `R-REJECT-RETRACT` | REJECT | Retraction too weak |
| `R-DISPUTE` | DISPUTE | Conflict in the ambiguous middle band |
| `R-RETRACT` | RETRACT | Valid memory closed by a retraction |
| `R-DISPUTE-RESOLVE` | (manual) | Human picks a winner |
| `R-DISPUTE-AUTO` | (stretch) | Automatic dispute resolution |
| `R-PROPAGATE` | (side effect) | Dependent flagged `needs_review` |

---

## 11. Required Tests (one per rule minimum)

| Test name | Asserts |
| :--- | :--- |
| `test_r_insert_first_claim` | Empty chain yields active memory with strength confidence |
| `test_r_corroborate_noisy_or` | m1+m2 numbers give 0.958 (±0.001) |
| `test_r_dup_same_source_no_boost` | Confidence unchanged, evidence row added |
| `test_r_replace_closes_window` | Old `valid_to` equals new `valid_from`; `closed_at` set; supersedes edge |
| `test_r_replace_blocked_by_margin` | Score just under threshold becomes dispute, not replace |
| `test_r_reject_low_trust` | Intern claim rejected, stored with `rejected` status |
| `test_r_dispute_band` | m7 numbers give both memories `disputed` |
| `test_r_refine_version_prefix` | `3.12.4` refines `3.12`; old historical |
| `test_less_specific_is_identical` | `3.12` against `3.12.4` corroborates |
| `test_r_coexist_multi` | Two `runs_service` values both active |
| `test_r_backfill_gap` | Older fact inserted as historical with correct `valid_to` |
| `test_history_conflict_never_rewrites` | Claim contradicting closed window cannot become active |
| `test_revert_guard` | Stale source repeating superseded value does not replace |
| `test_r_retract_closes_multi` | Retraction closes matching memory |
| `test_context_separates_chains` | Staging and global never conflict |
| `test_propagate_transitive_cycle_safe` | A->B->C chain flagged; cycle terminates |
| `test_as_of_boundary` | `valid_at == valid_to` returns the new memory, not the old |
| `test_known_at_ignores_later_closure` | Belief-time query before `closed_at` still shows old fact as open |
| `test_idempotent_ingest` | Duplicate message changes nothing |
| `test_invariants_after_scenarios` | AGENTS.md section 3D invariants hold after every eval scenario |
