CREATE TABLE IF NOT EXISTS dialogues (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    step            INTEGER NOT NULL,
    stage           INTEGER NOT NULL,
    question        TEXT    NOT NULL,
    answer          TEXT    NOT NULL,
    curiosity_score REAL    NOT NULL,
    clusters_active TEXT    NOT NULL,
    delta_summary   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    REAL    NOT NULL,
    step         INTEGER NOT NULL,
    event_type   TEXT    NOT NULL,
    cluster_a    TEXT,
    cluster_b    TEXT,
    metadata     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS latent_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     REAL    NOT NULL,
    step          INTEGER NOT NULL,
    node_count    INTEGER NOT NULL,
    cluster_count INTEGER NOT NULL,
    edge_count    INTEGER NOT NULL,
    graph_json    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     REAL    NOT NULL,
    step          INTEGER NOT NULL,
    weights_path  TEXT    NOT NULL,
    node_count    INTEGER NOT NULL,
    cluster_count INTEGER NOT NULL,
    edge_count    INTEGER NOT NULL,
    stage         INTEGER NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'complete',
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS human_chat (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    step            INTEGER NOT NULL,
    role            TEXT    NOT NULL,
    message         TEXT    NOT NULL,
    clusters_active TEXT
);

CREATE TABLE IF NOT EXISTS cluster_cofiring (
    cluster_a    TEXT    NOT NULL,
    cluster_b    TEXT    NOT NULL,
    count        INTEGER DEFAULT 0,
    last_updated INTEGER DEFAULT 0,
    PRIMARY KEY (cluster_a, cluster_b)
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    image_id     INTEGER PRIMARY KEY,
    image_emb    BLOB    NOT NULL,
    caption_emb  BLOB    NOT NULL,
    caption_text TEXT    NOT NULL,
    image_url    TEXT
);

CREATE TABLE IF NOT EXISTS category_performance (
    category    TEXT    PRIMARY KEY,
    total       INTEGER DEFAULT 0,
    positive    INTEGER DEFAULT 0,
    avg_sim     REAL    DEFAULT 0.0,
    last_step   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS episodic_memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    step            INTEGER NOT NULL,
    category        TEXT,
    input_vec       BLOB    NOT NULL,
    expected_vec    BLOB    NOT NULL,
    error_magnitude REAL    NOT NULL,
    trigger         TEXT    NOT NULL,
    replay_count    INTEGER DEFAULT 0,
    last_replayed   INTEGER DEFAULT 0,
    created_at      REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodic_category ON episodic_memories(category);
