PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sources (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    reliability REAL NOT NULL CHECK (reliability BETWEEN 0 AND 1),
    notes       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS entities (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    type           TEXT NOT NULL DEFAULT 'thing',
    needs_review   INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_norm TEXT PRIMARY KEY,                    -- lowercase, punctuation stripped
    entity_id  INTEGER NOT NULL REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS attribute_schema (
    attribute           TEXT PRIMARY KEY,
    description         TEXT NOT NULL,
    cardinality         TEXT NOT NULL CHECK (cardinality IN ('single','multi')),
    value_type          TEXT NOT NULL CHECK (value_type IN ('version','enum','text','number','date')),
    half_life_days      REAL,
    allowed_values_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL REFERENCES sources(id),
    text        TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    event_time  TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    UNIQUE (source_id, text_hash, event_time)
);

CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id     INTEGER NOT NULL REFERENCES entities(id),
    attribute     TEXT NOT NULL REFERENCES attribute_schema(attribute),
    value_norm    TEXT NOT NULL,
    value_display TEXT NOT NULL,
    context_json  TEXT NOT NULL DEFAULT '{}',
    context_key   TEXT NOT NULL DEFAULT '',
    valid_from    TEXT NOT NULL,
    valid_to      TEXT,
    recorded_at   TEXT NOT NULL,
    closed_at     TEXT,
    status        TEXT NOT NULL CHECK (status IN ('active','historical','rejected','disputed')),
    confidence    REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    needs_review  INTEGER NOT NULL DEFAULT 0,
    review_reason TEXT NOT NULL DEFAULT '',
    raw_text      TEXT NOT NULL DEFAULT '',
    message_id    INTEGER NOT NULL REFERENCES messages(id),
    source_id     TEXT NOT NULL REFERENCES sources(id),
    CHECK (valid_to IS NULL OR closed_at IS NOT NULL),
    CHECK (valid_to IS NULL OR valid_to > valid_from)
);
CREATE INDEX IF NOT EXISTS idx_mem_lookup ON memories(entity_id, attribute, context_key, status);
CREATE INDEX IF NOT EXISTS idx_mem_valid  ON memories(valid_from, valid_to);

CREATE TABLE IF NOT EXISTS evidence (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    source_id   TEXT NOT NULL REFERENCES sources(id),
    message_id  INTEGER NOT NULL REFERENCES messages(id),
    strength    REAL NOT NULL CHECK (strength BETWEEN 0 AND 1),
    observed_at TEXT NOT NULL,
    span        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_ev_mem ON evidence(memory_id);

CREATE TABLE IF NOT EXISTS edges (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    src_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    dst_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    type          TEXT NOT NULL CHECK (type IN ('supersedes','refines','contradicts','depends_on')),
    reason        TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    UNIQUE (src_memory_id, dst_memory_id, type)
);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON edges(dst_memory_id, type);
CREATE INDEX IF NOT EXISTS idx_edge_src ON edges(src_memory_id, type);

CREATE TABLE IF NOT EXISTS decision_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id          INTEGER NOT NULL REFERENCES messages(id),
    claim_json          TEXT NOT NULL,
    action              TEXT NOT NULL,
    rule_id             TEXT NOT NULL,
    memory_id           INTEGER REFERENCES memories(id),
    affected_ids_json   TEXT NOT NULL DEFAULT '[]',
    scores_json         TEXT NOT NULL DEFAULT '{}',
    explanation         TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id             TEXT PRIMARY KEY,
    timestamp      TEXT NOT NULL,
    prompt_id      TEXT NOT NULL,           -- extract_claims | classify_relation | parse_query | compose_answer | ...
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_summary TEXT NOT NULL,
    response_raw   TEXT NOT NULL,
    parsed_json    TEXT,
    latency_ms     REAL NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('success','retry','error','cache_hit')),
    error          TEXT
);

