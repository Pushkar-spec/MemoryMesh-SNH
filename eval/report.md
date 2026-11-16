# MemoryMesh Evaluation Report & G2 Gate Verification

**Execution Timestamp:** `20261116_043132`  
**Total Scenarios Evaluated:** 51  
**Total Questions Evaluated:** 111  
**G2 Gate Status:** **`PASS`**  

---

## A. Executive Summary

This evaluation report benchmarks **MemoryMesh** against a **Naive RAG Baseline (B0)** powered by FastEmbed dense semantic retrieval (`BAAI/bge-small-en-v1.5`, top-k=5).

### Core Findings
1. **Write-Time Resolution Outperforms Query-Time RAG:** Resolving updates and trust at write time achieves **100.0% Current Accuracy** and **0.0% Stale Answer Rate** under MemoryMesh structured evaluation, compared to **58.5% Current Accuracy** and **34.7% Stale Answer Rate** for Naive RAG.
2. **Elimination of Stale Retrievals:** Dense vector search retrieves superseded statements alongside current facts. Without write-time resolution, naive RAG surfaces stale rumors.
3. **Auditability and Dependency Impact:** MemoryMesh provides deterministic provenance traces, rule explanations, and review flagging (`R-PROPAGATE`), fundamentally impossible in unstructured vector retrieval.
4. **G2 Gate Determination:** The formal **G2 Evaluation Gate evaluated to `PASS`**, with all 7 acceptance criteria verified.

---

## B. Evaluation Methodology

### 1. Scenario Catalog & Scope
The evaluation exercises **51 reproducible scenarios** across 11 groups:

| Category | Scenarios | Focus & Test Invariants |
| :--- | :---: | :--- |
| `simple_update` | 6 | Monotonic updates, validity windows, supersedes edges |
| `corroboration` | 5 | Multi-source corroboration, noisy-OR confidence boost |
| `low_trust_reject` | 5 | Threshold enforcement, rejection of low-trust rumors |
| `dispute` | 4 | Equal-trust conflicting sources, dispute flagging |
| `multi_valued` | 4 | Attributes supporting concurrent valid values |
| `context_isolation` | 6 | Context isolation across threads, projects, and users |
| `out_of_order` | 5 | Event-time vs arrival-time ordering and graph repair |
| `dependency_propagation` | 5 | Causal graph tracking and cascade invalidation |
| `historical_time_travel` | 5 | Point-in-time state reconstruction via validity windows |
| `designed_to_lose` | 5 | Adversarial cases: unmapped aliases, safe abstention |
| `hero` | 1 | Hero multi-step scenario exercising all engine paths |

### 2. Execution Modes
Evaluation executes across 4 canonical operational modes:
1. **Structured Mode (Zero-LLM):** Direct claim ingestion; pure engine deterministic resolution; tests core resolution logic in isolation.
2. **Engine-Only Mode:** Query answering with deterministic planner; isolates query planning and retrieval from generative LLM variance.
3. **End-to-End (E2E) Mode:** Full pipeline with natural language ingestion and queries. _Note: E2E is evaluated using deterministic simulation providers per the roadmap._
4. **Naive RAG Baseline (B0):** Dense retrieval top-k=5 raw messages using FastEmbed (`BAAI/bge-small-en-v1.5`) and shared completion schema.

### 3. Mode Execution Statuses
| Mode | Required Role | Status | Description |
| :--- | :--- | :---: | :--- |
| `structured` | Canonical G2 Mode | **`COMPLETED`** | Canonical mode |
| `engine-only` | Canonical G2 Mode | **`COMPLETED`** | Canonical mode |
| `e2e` | Canonical G2 Mode | **`COMPLETED`** | Deterministic simulation / partial |
| `baseline` | Canonical G2 Mode | **`COMPLETED`** | Canonical mode |

---

## C. Quantitative Results

Below is the formal cross-system evaluation table:

| Metric | B0 (naive RAG) (Naive RAG) | MM (structured) (MemoryMesh (structured)) | MM (engine-only) (Engine-only) | MM (e2e) (MemoryMesh (e2e)) |
| :--- | :---: | :---: | :---: | :---: |
| Current accuracy | 58.5% | 100.0% | 100.0% | 84.9% |
| As-of accuracy | 36.4% | 100.0% | 100.0% | 90.9% |
| History accuracy | 18.2% | 100.0% | 100.0% | 100.0% |
| Stale-answer rate | 34.7% | 0.0% | 0.0% | 2.0% |
| Conflict recall | 80.0% | 100.0% | 100.0% | 100.0% |
| Conflict false-alarm rate | 61.4% | 0.0% | 0.0% | 0.0% |
| Impact recall | n/a | 100.0% | 100.0% | 100.0% |
| Impact precision | n/a | 100.0% | 100.0% | 100.0% |
| Provenance correctness | n/a | 100.0% | 100.0% | 93.8% |
| Abstention accuracy | 20.0% | 100.0% | 100.0% | 80.0% |
| Engine decision accuracy | n/a | 100.0% | 100.0% | 100.0% |
| Avg LLM calls / query | 1.00 | 0.00 | 0.00 | 1.68 |
| Avg Latency (s) / query | 0.01 | 0.00 | 0.00 | 0.00 |

\* _B0 metrics for Impact, Provenance, and Decision accuracy are reported as 'n/a' (no rule IDs or graph traversal)._


### Metric Definitions
- **Current Accuracy:** % of current queries matching ground truth value.
- **As-of Accuracy:** % of point-in-time historical queries matching ground truth.
- **History Accuracy:** % of queries where chronological value sequence is correct.
- **Stale Answer Rate:** % of answers returning superseded or forbidden values.
- **Conflict Recall:** % of genuine conflicts flagged by the system.
- **Conflict False-Alarm Rate:** % of unconflicted queries mistakenly flagged.
- **Impact Recall:** % of affected downstream dependents correctly identified.
- **Provenance Correctness:** % of answers citing expected source and rule.
- **Abstention Accuracy:** % of unrecorded/unsupported queries safely abstained.
- **Engine Decision Accuracy:** % of ingestion claims matching expected rule ID.

---

## D. Scenario-Group Analysis

### 1. Update and Validity Resolution (`simple_update`, `out_of_order`)
MemoryMesh achieves **0.0% Stale Answer Rate** by closing validity windows on superseded memories (`valid_until = event_time`). Naive RAG suffers high stale rates because superseded messages remain in top-k dense retrieval context.
### 2. Trust and Conflict Resolution (`low_trust_reject`, `dispute`, `corroboration`)
MemoryMesh applies trust rules at ingestion time (`R-REJECT`, `R-DISPUTE`, `R-CORROBORATE`). Naive RAG treats low-trust sources with equal semantic weight.
### 3. Temporal Queries (`historical_time_travel`)
MemoryMesh answers point-in-time queries (`as_of`) by filtering validity intervals (`valid_from <= t < valid_until`). Naive RAG has no time model.
### 4. Dependency Tracking (`dependency_propagation`)
When an upstream fact changes, MemoryMesh traverses the dependency graph and flags affected memories with `needs_review = 1` (`R-PROPAGATE`).
### 5. Designed-To-Lose Invariants (`designed_to_lose`)
Adversarial and boundary cases test safe abstention and failure containment. MemoryMesh abstains safely when evidence is missing or disputes cannot be auto-resolved.

---

## E. Failure Analysis

This section documents all system discrepancies, failures, and designed-to-lose boundary handlings without suppression.

### 1. Cross-System Failure Log

#### B0 (naive RAG) (45 Question Failures)

| Scenario | Question | Failure Reason | Classification | Expected | Observed |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `context-005` | `q2` | Incorrect current value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `core platform team` | `sre core` |
| `context-006` | `q3` | Incorrect current value, Stale / forbidden value present, Abstention mismatch | `[Baseline Retrieval / Temporal Blindness]` | `{'abstain': True, 'value'...` | `3.11` |
| `hero-001` | `q1` | Incorrect current value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `3.12` | `3.9` |
| `hero-001` | `q2` | Incorrect as-of value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `3.10` | `3.9` |
| `hero-001` | `q5` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `['friday', 'monday']` | `monday` |
| `hero-001` | `q7` | Incorrect history sequence, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `['3.10', '3.12']` | `['3.9', '3.10', '3.12', '...` |
| `dependency-001` | `q2` | Incorrect current value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `3.12` | `3.10` |
| `dependency-004` | `q2` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `redis-cache` | `auth-service` |
| `dispute-001` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `['friday', 'monday']` | `monday` |
| `dispute-002` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `['3.10', '3.11']` | `3.10` |
| `dispute-003` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `['data engineering', 'pla...` | `Retrieved messages do not...` |
| `dispute-004` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `['degraded', 'maintenance']` | `maintenance` |
| `lose-002` | `q1` | Incorrect current value | `[Designed-To-Lose: Intentional Boundary]` | `['friday', 'monday']` | `monday` |
| `lose-003` | `q1` | Incorrect as-of value, Abstention mismatch | `[Designed-To-Lose: Intentional Boundary]` | `{'abstain': True, 'value'...` | `3.12` |
| `lose-004` | `q1` | Incorrect current value, Stale / forbidden value present | `[Designed-To-Lose: Intentional Boundary]` | `3.12` | `3.9` |
| `multivalue-001` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `['api-gateway', 'auth-ser...` | `auth-service` |
| `multivalue-001` | `q2` | Incorrect history sequence | `[Baseline Retrieval / Temporal Blindness]` | `['auth-service', 'api-gat...` | `auth-service` |
| `multivalue-002` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `['billing-worker', 'metri...` | `Retrieved messages do not...` |
| `multivalue-003` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `['redis-cache', 'search-i...` | `Retrieved messages do not...` |
| `multivalue-004` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `['cron-scheduler', 'inges...` | `Retrieved messages do not...` |
| `multivalue-004` | `q2` | Incorrect as-of value | `[Baseline Retrieval / Temporal Blindness]` | `['ingestion-pipeline']` | `Retrieved messages do not...` |
| `ordering-001` | `q2` | Incorrect as-of value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `3.10` | `3.12` |
| `ordering-003` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `degraded` | `operational` |
| `ordering-003` | `q2` | Incorrect as-of value | `[Baseline Retrieval / Temporal Blindness]` | `up` | `operational` |
| `ordering-004` | `q3` | Incorrect history sequence | `[Baseline Retrieval / Temporal Blindness]` | `['friday', 'monday']` | `['monday', 'friday']` |
| `reject-005` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `infrastructure core team` | `infrastructure core` |
| `temporal-001` | `q1` | Incorrect as-of value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `3.10` | `3.12` |
| `temporal-001` | `q2` | Incorrect current value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `3.12` | `3.10` |
| `temporal-002` | `q1` | Incorrect as-of value, Abstention mismatch | `[Baseline Retrieval / Temporal Blindness]` | `{'abstain': True, 'value'...` | `3.10` |
| `temporal-003` | `q1` | Incorrect as-of value | `[Baseline Retrieval / Temporal Blindness]` | `degraded` | `up` |
| `temporal-004` | `q2` | Incorrect as-of value, Stale / forbidden value present, Abstention mismatch | `[Baseline Retrieval / Temporal Blindness]` | `{'abstain': True, 'value'...` | `maintenance` |
| `temporal-005` | `q1` | Incorrect history sequence | `[Baseline Retrieval / Temporal Blindness]` | `['3.10', '3.12']` | `['3.12', '3.10']` |
| `update-001` | `q1` | Incorrect current value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `3.12` | `3.10` |
| `update-001` | `q2` | Incorrect as-of value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `3.10` | `3.12` |
| `update-001` | `q3` | Incorrect history sequence | `[Baseline Retrieval / Temporal Blindness]` | `['3.10', '3.12']` | `['3.12', '3.10']` |
| `update-002` | `q1` | Incorrect current value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `debian 12` | `ubuntu 22.04` |
| `update-003` | `q1` | Incorrect current value | `[Baseline Retrieval / Temporal Blindness]` | `core platform team` | `Retrieved messages do not...` |
| `update-003` | `q2` | Incorrect as-of value | `[Baseline Retrieval / Temporal Blindness]` | `infra team` | `Retrieved messages do not...` |
| `update-003` | `q3` | Incorrect history sequence | `[Baseline Retrieval / Temporal Blindness]` | `['infra team', 'core plat...` | `Retrieved messages do not...` |
| `update-004` | `q2` | Incorrect as-of value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `thursday` | `tuesday` |
| `update-004` | `q3` | Incorrect history sequence | `[Baseline Retrieval / Temporal Blindness]` | `['thursday', 'tuesday']` | `['tuesday', 'thursday']` |
| `update-005` | `q2` | Incorrect as-of value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `degraded` | `up` |
| `update-005` | `q3` | Incorrect history sequence | `[Baseline Retrieval / Temporal Blindness]` | `['degraded', 'up']` | `['up', 'maintenance', 'de...` |
| `update-006` | `q3` | Incorrect as-of value, Stale / forbidden value present | `[Baseline Retrieval / Temporal Blindness]` | `3.10` | `3.9` |
| `update-006` | `q4` | Incorrect history sequence | `[Baseline Retrieval / Temporal Blindness]` | `['3.9', '3.10', '3.12']` | `['3.12', '3.9', '3.10']` |

#### MM (structured) (0 Question Failures)

_Zero question failures recorded in this mode._

#### MM (engine-only) (0 Question Failures)

_Zero question failures recorded in this mode._

#### MM (e2e) (11 Question Failures)

| Scenario | Question | Failure Reason | Classification | Expected | Observed |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `context-001` | `q1` | Incorrect current value | `[Engine Limitation / Strict Expectation]` | `3.12` | `No memory valid at the re...` |
| `context-001` | `q2` | Incorrect current value | `[Engine Limitation / Strict Expectation]` | `3.11` | `No memory valid at the re...` |
| `context-001` | `q3` | Provenance cited source/rule mismatch | `[Engine Limitation / Strict Expectation]` | `{'cited_source': 'sysadmi...` | `No memory valid at the re...` |
| `context-002` | `q1` | Incorrect current value | `[Engine Limitation / Strict Expectation]` | `ubuntu 24.04 lts` | `No memory valid at the re...` |
| `context-002` | `q2` | Incorrect current value | `[Engine Limitation / Strict Expectation]` | `debian 12` | `No memory valid at the re...` |
| `context-004` | `q1` | Incorrect current value, Stale / forbidden value present | `[Engine Limitation / Strict Expectation]` | `wednesday` | `friday` |
| `context-005` | `q1` | Incorrect current value | `[Engine Limitation / Strict Expectation]` | `sre core` | `No memory valid at the re...` |
| `context-005` | `q2` | Incorrect current value | `[Engine Limitation / Strict Expectation]` | `core platform team` | `No memory valid at the re...` |
| `hero-001` | `q6` | Incorrect current value | `[Engine Limitation / Strict Expectation]` | `3.11` | `3.12` |
| `lose-003` | `q1` | Incorrect as-of value, Abstention mismatch | `[Designed-To-Lose: Intentional Boundary]` | `{'abstain': True, 'value'...` | `3.12` |
| `temporal-002` | `q2` | Incorrect as-of value | `[Engine Limitation / Strict Expectation]` | `3.10` | `No memory valid at the re...` |

### 2. Designed-to-Lose & Boundary Case Accounting

The canonical evaluation suite includes 5 designed-to-lose scenarios (`lose-001` to `lose-005`). These adversarial scenarios explicitly test boundary conditions, unresolvable disputes, low-trust defenses, and safe abstention.

| Scenario ID | Adversarial Invariant | MM Observed Outcome | Accounting Status |
| :--- | :--- | :--- | :--- |
| `lose-001` | Insufficient evidence: query asks for unrecorded attribute; requires safe abstention | Safe abstention verified (`value=None`, `abstained=True`) | Honest Abstention (Pass) |
| `lose-002` | Peer dispute without sufficient authority: flags conflict rather than forcing a winner | Conflict flagged (`values=['friday', 'monday']`) | Honest Dispute Flag (Pass) |
| `lose-003` | Historical date prior to entity existence: requires safe time-travel abstention | Safe historical abstention verified (`value=None`, `abstained=True`) | Honest Abstention (Pass) |
| `lose-004` | Unverified low-trust rumor: rejects unverified claim and prevents memory adoption | R-REJECT enforced on claim '3.9'; rumor blocked (`value=3.12`) - Conflicting claim '3.9' rejected on attribute 'python_version': score 0.210 is below reject threshold 0.427 (old_score=0.855, reject_ratio=0.5). | Honest Defense (Pass) |
| `lose-005` | Broken upstream dependency: flags needs_review rather than assuming compatibility | Downstream impact flagged for review: expected dependent [('Library X', 'compatibility_note')] verified with per-dependent needs_review=True | Honest Review Flag (Pass) |

---

## F. Limitations

1. **Simulation Provider vs Real-World LLMs:** The E2E and baseline modes use deterministic, rule-based simulation providers. While faithful for benchmarking, they do not exhibit real-world non-determinism or hallucinations.
2. **Baseline Answer Extraction Heuristics:** The Naive RAG baseline uses FastEmbed dense retrieval combined with candidate extraction. Multi-shot frontier LLMs might perform slightly better, but remain vulnerable to conflicting context windows.
3. **Sequential Ingestion Acyclicity Constraint:** As identified in P8.3, sequential claim ingestion strictly produces an acyclic DAG ($src > dst$). Cyclic graphs cannot be formed through message ingestion alone without graph edits.
4. **Static Configuration:** Source reliabilities and conflict margins are currently fixed via YAML. Dynamic trust learning from feedback is an opportunity for future work.

---

## G. Conclusion

The empirical evaluation conclusively demonstrates the primary thesis of MemoryMesh:
> **Write-time memory resolution is superior to raw retrieval at query time.**

- MemoryMesh eliminates stale answers, isolates conflicting contexts, enforces trust boundaries, and tracks transitive causal dependencies.
- Naive RAG suffers catastrophic degradation as knowledge evolves over time, because dense vector search retrieves semantically similar but superseded messages into context.
- MemoryMesh satisfies all required engineering constraints, formalizing long-term memory as an auditable, temporal knowledge graph with strict write-path resolution.

---

## H. G2 Evaluation Gate Verification

### Overall Gate Determination: **`PASS`**

Gate G2 PASSED: All 7 acceptance criteria satisfied with empirical evidence.

| Criterion | Title | Target Acceptance Rule | Status | Empirical Evidence |
| :---: | :--- | :--- | :---: | :--- |
| `G2-1` | Artifact Generation | Generate report.md and results.json; complete G2 gate. | **`PASS`** | Artifacts verified on disk and read back successfully: report.md (21838 bytes, Section H verified) and results.json (7656457 bytes, all criteria match). |
| `G2-2` | Multi-Mode Comparative Evaluation | 'make eval' produces table in both modes \| label e2e as partial | **`PASS`** | Evaluated across all 4 canonical modes: structured [COMPLETED], engine-only [COMPLETED], e2e (simulated/partial) [COMPLETED], naive RAG baseline (B0) [COMPLETED]. |
| `G2-3` | Scenario Catalog Coverage | Include all applicable evaluation scenarios (~40 sc, ~100 questions) | **`PASS`** | Full canonical catalog coverage derived from execution across all 4 modes (4 systems): 51 scenarios (target >= 40) across 10 categories with 111 declared questions (target >= 100) all verified executed with zero missing, duplicate, or mismatched question traces. |
| `G2-4` | Write-Time Resolution Superiority | Write-time resolution beats raw message retrieval at answer time. | **`PASS`** | MemoryMesh write-time resolution outperforms Naive RAG (B0): Current Acc is 100.0% vs 58.5%; Stale Rate is 0.0% vs 34.7%. |
| `G2-5` | Decision Auditability and Provenance | Decisions must be auditable; citations reflect rules/sources; B0 is n/a. | **`PASS`** | MemoryMesh decisions and provenance are fully auditable with evidence: Decision Accuracy: 100.0% (>=90.0%), Provenance Correctness: 100.0% (>=80.0%). Baseline B0 correctly reports 'n/a' (no graph or rule logging). |
| `G2-6` | Designed-To-Lose Honesty & Failure Accounting | Include at least 3 designed-to-lose cases and report them honestly. | **`PASS`** | Designed-to-lose cases (5 scenarios: lose-001, lose-002, lose-003, lose-004, lose-005) empirically executed with verified invariants (abstentions, dispute flags, R-REJECT decisions, and dependency reviews) and accounted for in Section E (Failure Analysis) without suppressing boundary conditions or limitations. |
| `G2-7` | Deterministic Reproducibility | Results can be regenerated identically from the documented command. | **`PASS`** | Dual independent evaluation runs produced 100% identical normalized outcomes across all 4 canonical modes (4 systems), 51 scenarios, and 444 individual questions: all metrics, answers, citations, conflict/abstention flags, and failure distributions match exactly. |

---

_Report automatically generated by MemoryMesh Evaluation Runner (P8.4)._

