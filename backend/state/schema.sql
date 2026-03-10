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
