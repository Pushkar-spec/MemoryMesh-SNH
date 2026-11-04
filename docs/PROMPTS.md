# MemoryMesh: LLM Prompt Specifications

LLMs are used for **language tasks only**: extraction, text-conflict classification, query parsing, and answer writing. They never choose replace/reject/dispute (see `AGENTS.md`).

**Global settings for every prompt:** temperature `0`; pure JSON output; validated by Pydantic; one retry on failure; every call logged to `llm_calls`.

| Prompt ID | Used by | Model tier | Output model |
| :--- | :--- | :--- | :--- |
| `extract_claims` | `ingest/extractor.py` | fast | `ExtractionResult` |
| `classify_relation` | `engine/classifier.py` | fast | `RelationResult` |
| `parse_query` | `llm/query_parser.py` | fast | `QueryPlan` |
| `compose_answer` | `llm/composer.py` | answer | `FinalAnswer` |
| `baseline_answer` | `baseline/naive_rag.py` | answer | `FinalAnswer` |
| `entity_tiebreak` (stretch) | `ingest/entity_resolver.py` | fast | `TiebreakResult` |

Templates live in `src/memorymesh/llm/prompts/<prompt_id>.md`. Placeholders use `{{double_braces}}`.

---

## 1. `extract_claims`

**Inputs:** message text, source name, `reference_time` (message event time, ISO), attribute catalog, known entities.

### Template
```text
You extract structured factual claims from one message for an IT-infrastructure memory system.

### REFERENCE TIME
The message was written at: {{reference_time}}
Resolve relative dates ("yesterday", "last Tuesday", "today") against this time and output ISO 8601 UTC.

### SOURCE
{{source_name}}

### ALLOWED ATTRIBUTES (use ONLY these names)
{{attribute_catalog}}
Each line: name | cardinality | value_type | allowed values | description | phrases that hint at it

### KNOWN ENTITIES (prefer these exact names when the message refers to them)
{{known_entities}}

### MESSAGE
"""
{{message_text}}
"""

### INSTRUCTIONS
1. Extract every distinct factual claim about an entity that maps to an allowed attribute. If none, return an empty list.
2. subject: the entity name. Use a known entity name verbatim when it matches (including nicknames).
3. value: the bare value only. For versions output "3.12", not "Python 3.12". For enums use the allowed lowercase value.
4. polarity: "assert" normally. Use "retract" when the message says something is NO LONGER true ("no longer runs", "decommissioned").
5. valid_from: the time the fact became true, if the message says so ("migrated yesterday"). If the message states the fact as simply true now, output null.
6. valid_to: only when the message says the fact stopped being true at a specific time; otherwise null.
7. context: scope qualifiers as key/value strings, e.g. {"environment": "staging"}. Use {} when the claim is general. Never invent a context.
8. depends_on: if the claim is only true because of another fact (e.g. "Library X works with Server A's Python 3.10"), list the foundation as {"entity": ..., "attribute": ...}. Otherwise [].
9. extract_confidence in [0,1]:
   - 0.90-1.00: direct, unambiguous statement
   - 0.60-0.85: slightly vague or inferred
   - 0.30-0.55: hedged ("I think", "maybe", "?", "not sure", "I heard")
   - below 0.30: speculation
10. span: copy the exact words from the message that support the claim.
11. Do NOT invent attributes, entities, or values. Do NOT include questions as claims unless they assert something with hedging.

### OUTPUT (strictly valid JSON, no markdown, no commentary)
{
  "claims": [
    {
      "subject": "string",
      "attribute": "string",
      "value": "string",
      "polarity": "assert" | "retract",
      "valid_from": "ISO8601 or null",
      "valid_to": "ISO8601 or null",
      "context": {"key": "value"},
      "depends_on": [{"entity": "string", "attribute": "string"}],
      "extract_confidence": float,
      "span": "string"
    }
  ]
}
```

### Few-shot examples (embedded in the template)

**Example A**: reference `2026-06-01T08:00:00Z`
Message: `We migrated Server A to Python 3.12 today.`
```json
{"claims":[{"subject":"Server A","attribute":"python_version","value":"3.12","polarity":"assert","valid_from":"2026-06-01T00:00:00Z","valid_to":null,"context":{},"depends_on":[],"extract_confidence":0.95,"span":"migrated Server A to Python 3.12 today"}]}
```

**Example B**: reference `2026-06-03T16:20:00Z`
Message: `I think Server A is still on Python 3.9?`
```json
{"claims":[{"subject":"Server A","attribute":"python_version","value":"3.9","polarity":"assert","valid_from":null,"valid_to":null,"context":{},"depends_on":[],"extract_confidence":0.5,"span":"I think Server A is still on Python 3.9?"}]}
```

**Example C**: reference `2026-03-05T14:00:00Z`
Message: `Library X is deployed on Server A and works fine with its Python 3.10 runtime.`
```json
{"claims":[{"subject":"Library X","attribute":"compatibility_note","value":"works with Python 3.10 on Server A","polarity":"assert","valid_from":null,"valid_to":null,"context":{},"depends_on":[{"entity":"Server A","attribute":"python_version"}],"extract_confidence":0.9,"span":"works fine with its Python 3.10 runtime"}]}
```

**Example D** (retraction): reference `2026-04-10T09:00:00Z`
Message: `Server A no longer hosts the billing service.`
```json
{"claims":[{"subject":"Server A","attribute":"runs_service","value":"billing","polarity":"retract","valid_from":null,"valid_to":null,"context":{},"depends_on":[],"extract_confidence":0.9,"span":"no longer hosts the billing service"}]}
```

**Post-processing (code, not LLM):** drop claims whose attribute is not in the catalog (`R-SKIP-ATTR`), normalize values (`R-SKIP-VALUE`), cap `valid_from` at `reference_time` (future-dated facts are a non-goal).

---

## 2. `classify_relation` (text-valued attributes only)

**Inputs:** attribute description, cardinality, existing value, new value, optional surrounding raw texts.

```text
You compare two statements about the same entity and attribute.

### ATTRIBUTE
{{attribute}}: {{attribute_description}}  (cardinality: {{cardinality}})

### EXISTING STATEMENT
"{{old_value}}"   (original text: "{{old_raw_text}}")

### NEW STATEMENT
"{{new_value}}"   (original text: "{{new_raw_text}}")

### DEFINITIONS
- identical: they say the same thing (paraphrase, or the new one is less specific but consistent).
- refines: the new statement is the same claim with more detail, not contradicting the old.
- conflicts: they cannot both be true at the same time for a single-valued attribute.
- complementary: both can be true at once (different aspects, or additive items).

### OUTPUT (strictly valid JSON only)
{"relation": "identical" | "refines" | "conflicts" | "complementary", "confidence": float, "reason": "one sentence"}
```

Rules: if `confidence < 0.5`, code falls back to `conflicts` (single-valued) or `complementary` (multi-valued) and notes the fallback in the decision explanation.

---

## 3. `parse_query`

**Inputs:** question, `now` (ISO), attribute catalog, known entities.

```text
You convert a question about stored infrastructure knowledge into a query plan.

### CURRENT TIME
{{now}}

### ALLOWED ATTRIBUTES
{{attribute_catalog}}

### KNOWN ENTITIES
{{known_entities}}

### QUESTION
"{{question}}"

### INTENTS
- current: what is true now
- as_of: what was true at a specific past time (resolve "in March" to the middle of that month in the most recent such March before now, e.g. 15th, 12:00Z)
- history: how it changed over time ("what versions has it had")
- why: why does the system believe something / where did it come from
- impact: what might break or need review because of a change
- conflicts: where do sources disagree

### INSTRUCTIONS
- entity and attribute must come from the lists; use null if the question does not identify them.
- valid_at: ISO time for as_of questions, else null.
- known_at: set ONLY if the question asks what was believed/known at a past time ("what did we think on June 2nd"), else null.
- context: e.g. {"environment": "staging"} if the question names one, else {}.
- memory_id: only if the question cites an ID like "M12", else null.

### OUTPUT (strictly valid JSON only)
{"intent": "...", "entity": "string or null", "attribute": "string or null",
 "valid_at": "ISO or null", "known_at": "ISO or null", "context": {}, "memory_id": integer or null}
```

If `entity` or `attribute` is null for an intent that requires it, the service returns `AskResult` with `answer.abstained = true` and explanation "Could not identify what you are asking about", plus suggestions from the catalog.

---

## 4. `compose_answer`

**Inputs:** question, `QueryResult` serialized as compact JSON (items with IDs, values, validity, confidence, sources, flags; disputes; impact; decisions).

```text
You answer a question using ONLY the retrieved memory records below. You may not use outside knowledge.

### QUESTION
"{{question}}"

### RETRIEVED RECORDS (JSON)
{{query_result_json}}

### RULES
1. Every factual statement must be supported by a record; cite it as [M<id>] in the explanation.
2. If "disputes" is non-empty, set conflict_flagged=true and state BOTH positions with their sources and confidences. Do not pick one.
3. If any item has needs_review=true, set needs_review_flagged=true and say why (use review_reason).
4. If any item is_stale=true, say the information may be outdated.
5. If there are no usable records, set abstained=true and say what is missing. Never guess.
6. For history/multi-valued questions put all values in "values" in chronological order; for single answers use "value".
7. Mention rejected claims only when the question is about history or "why", and say they were not believed and why.
8. Keep the explanation under 90 words, plain language.

### OUTPUT (strictly valid JSON only)
{"value": "string or null", "values": ["string"], "conflict_flagged": boolean, "needs_review_flagged": boolean,
 "abstained": boolean, "explanation": "string with [M<id>] citations", "cited_memory_ids": [integer]}
```

**Post-validation (code):** any ID in `cited_memory_ids` not present in the provided records is removed and logged as `hallucinated_citation`. `value` must equal a `value_display` from the records (case-insensitive), otherwise it is nulled and `abstained` is set.

---

## 5. `baseline_answer` (naive RAG, for fair evaluation)

Same output schema, same model, same rules wording where applicable, but the context is the **top-k raw messages** (k = 5) retrieved by dense embedding similarity with their source names and timestamps. No resolution, no validity, no confidence.

```text
Answer the question using ONLY the messages below.

### QUESTION
"{{question}}"

### RETRIEVED MESSAGES
{{messages_with_source_and_time}}

### RULES
- If messages disagree, say so and set conflict_flagged=true.
- If the messages are insufficient, set abstained=true.
- Put a single answer in "value", or several in "values" for history questions.
- Keep the explanation under 90 words.

### OUTPUT (strictly valid JSON only)
{"value": "string or null", "values": ["string"], "conflict_flagged": boolean, "needs_review_flagged": boolean,
 "abstained": boolean, "explanation": "string", "cited_memory_ids": []}
```

The baseline gets the **same instruction to flag conflicts** so it is not handicapped.

---

## 6. `entity_tiebreak` (stretch)

```text
Decide whether two names refer to the same IT entity.
Candidate A: "{{name_a}}" (known aliases: {{aliases_a}})
Candidate B: "{{name_b}}"
Message context: "{{message_text}}"
Output JSON only: {"same_entity": boolean, "confidence": float, "reason": "string"}
```
Used only for fuzzy scores between `entity_tiebreak_min` and `entity_auto_match`. If `same_entity` with confidence >= 0.8, add `name_b` as an alias; otherwise create a new entity with `needs_review = 1`.

---

## 7. Robust JSON Extraction (`llm/client.py`)

```python
import json, re

def extract_json_payload(raw_text: str) -> dict:
    """Return a JSON object even if wrapped in code fences or surrounded by prose."""
    raw_text = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if brace:
        return json.loads(brace.group(1))
    return json.loads(raw_text)
```

Failure protocol:
1. `json.JSONDecodeError` or Pydantic `ValidationError` -> log status `retry`; resend the same prompt with the suffix: `"Your previous reply was invalid: {error}. Return pure JSON matching the schema, no markdown."`
2. Second failure -> status `error`, raise `LLMOutputError`. Callers handle it as in `ARCHITECTURE.md` section 8.

---

## 8. Fake Provider Contract (tests and offline demo)

`FakeProvider` loads `tests/fixtures/llm_script.yaml`:

```yaml
extract_claims:
  - match: "We migrated Server A to Python 3.12 today."
    response: {claims: [{subject: "Server A", attribute: python_version, value: "3.12", ...}]}
classify_relation: []
compose_answer:
  default: deterministic   # builds FinalAnswer from QueryResult without a model
```

- Matching is by exact message text (extract) or by `(old_value, new_value)` (classify).
- `compose_answer: deterministic` generates `value`, flags, and a templated explanation directly from `QueryResult`, so the whole app and the engine-only eval run with no API key.
- An unmatched call raises `FakeScriptMissError` (fail loudly; never return junk).
