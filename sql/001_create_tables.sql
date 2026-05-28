-- ==========================================
-- Issue Cracker - 전체 스키마 (Full DDL)
-- 기존 테이블 DROP 후 재생성
-- ==========================================

DROP MATERIALIZED VIEW IF EXISTS issue_cracker.hourly_snapshots;
DROP TABLE IF EXISTS issue_cracker.analysis_results;
DROP TABLE IF EXISTS issue_cracker.comments;
DROP TABLE IF EXISTS issue_cracker.posts;
DROP TABLE IF EXISTS issue_cracker.crises;
DROP TYPE IF EXISTS issue_cracker.content_type;
DROP TYPE IF EXISTS issue_cracker.platform_type;

CREATE SCHEMA IF NOT EXISTS issue_cracker;

-- ==========================================
-- ENUM
-- ==========================================
CREATE TYPE issue_cracker.platform_type AS ENUM (
    'youtube', 'twitter', 'community', 'news', 'instagram', 'tiktok'
);

CREATE TYPE issue_cracker.content_type AS ENUM (
    'video', 'article', 'post', 'tweet', 'reel', 'short'
);

-- ==========================================
-- 🚨 Crises (위기 건) - 사용자 입력 단위
-- 파이프라인 1회 실행 = 1개 crisis
-- ==========================================
CREATE TABLE issue_cracker.crises (
    crisis_id       TEXT        PRIMARY KEY,
    title           TEXT        NOT NULL,
    description     TEXT,                           -- → PipelineState.crisis_context
    crisis_type     TEXT        NOT NULL
                    CHECK (crisis_type IN ('victim', 'accidental', 'preventable')),
    status          TEXT        NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'analyzing', 'resolved', 'archived')),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by      TEXT,

    train_csv_path  TEXT,
    input_csv_path  TEXT,
    extra           JSONB      DEFAULT '{}'
);

-- ==========================================
-- 📄 Posts (원문)
-- ==========================================
CREATE TABLE issue_cracker.posts (
    post_id         TEXT        PRIMARY KEY,
    crisis_id       TEXT        NOT NULL REFERENCES issue_cracker.crises(crisis_id),
    created_at      TIMESTAMPTZ NOT NULL,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    platform        issue_cracker.platform_type NOT NULL,
    content_type    issue_cracker.content_type  NOT NULL,
    url             TEXT,

    title           TEXT        NOT NULL,
    body            TEXT,

    author_id       TEXT        NOT NULL,
    author_name     TEXT,
    author_followers INTEGER   DEFAULT 0,

    view_count      INTEGER    DEFAULT 0,
    like_count      INTEGER    DEFAULT 0,
    share_count     INTEGER    DEFAULT 0,
    comment_count   INTEGER    DEFAULT 0,

    extra           JSONB      DEFAULT '{}'
);

-- ==========================================
-- 💬 Comments (댓글)
-- ==========================================
CREATE TABLE issue_cracker.comments (
    comment_id          TEXT        PRIMARY KEY,
    post_id             TEXT        NOT NULL REFERENCES issue_cracker.posts(post_id),
    crisis_id           TEXT        NOT NULL REFERENCES issue_cracker.crises(crisis_id),
    parent_comment_id   TEXT        REFERENCES issue_cracker.comments(comment_id),
    created_at          TIMESTAMPTZ NOT NULL,
    collected_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    body                TEXT        NOT NULL,

    author_id           TEXT        NOT NULL,
    author_name         TEXT,
    author_followers    INTEGER,

    like_count          INTEGER    DEFAULT 0,
    reply_count         INTEGER    DEFAULT 0,

    extra               JSONB      DEFAULT '{}'
);

-- ==========================================
-- 📊 Analysis Results (통합 분석 결과)
-- post / comment 공통
-- ==========================================
CREATE TABLE issue_cracker.analysis_results (
    id              BIGSERIAL   PRIMARY KEY,
    crisis_id       TEXT        NOT NULL REFERENCES issue_cracker.crises(crisis_id),

    target_type     TEXT        NOT NULL CHECK (target_type IN ('post', 'comment')),
    target_id       TEXT        NOT NULL,

    -- 공통
    sentiment       TEXT        CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    sentiment_score REAL,

    -- 댓글 특화
    is_mockery      BOOLEAN     DEFAULT FALSE,
    is_advocate     BOOLEAN     DEFAULT FALSE,

    -- 포스트 특화
    influence_score SMALLINT,

    -- 메타
    model_version   TEXT,
    analyzed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (target_type, target_id, model_version)
);

-- ==========================================
-- 📋 hourly_snapshots (Materialized View)
-- ==========================================
CREATE MATERIALIZED VIEW issue_cracker.hourly_snapshots AS
SELECT
    c.crisis_id,
    date_trunc('hour', c.created_at)                            AS hour_bucket,

    COUNT(*)                                                    AS total_mentions,
    COUNT(ar.id)                                                AS analyzed_count,

    COUNT(*) FILTER (WHERE ar.sentiment = 'negative')           AS negative_mentions,
    COUNT(*) FILTER (WHERE ar.is_mockery = TRUE)                AS mockery_mentions,
    COUNT(*) FILTER (WHERE ar.is_advocate = TRUE)               AS advocate_mentions,

    ROUND(
        COUNT(*) FILTER (WHERE ar.sentiment = 'negative')::NUMERIC
        / NULLIF(COUNT(ar.id), 0), 3
    )                                                           AS negative_ratio,
    ROUND(
        COUNT(*) FILTER (WHERE ar.is_mockery = TRUE)::NUMERIC
        / NULLIF(COUNT(ar.id), 0), 3
    )                                                           AS mockery_index,
    ROUND(
        COUNT(*) FILTER (WHERE ar.is_advocate = TRUE)::NUMERIC
        / NULLIF(COUNT(ar.id), 0), 3
    )                                                           AS advocate_ratio,

    COALESCE(
        MAX(par.influence_score) FILTER (WHERE par.influence_score IS NOT NULL), 0
    )                                                           AS influencer_impact

FROM issue_cracker.comments c
LEFT JOIN issue_cracker.analysis_results ar
    ON ar.target_type = 'comment' AND ar.target_id = c.comment_id
JOIN issue_cracker.posts p ON c.post_id = p.post_id
LEFT JOIN issue_cracker.analysis_results par
    ON par.target_type = 'post' AND par.target_id = p.post_id
GROUP BY c.crisis_id, date_trunc('hour', c.created_at)
ORDER BY c.crisis_id, hour_bucket;

-- ==========================================
-- 📊 인덱스
-- ==========================================
CREATE INDEX idx_crises_status          ON issue_cracker.crises (status, created_at DESC);
CREATE INDEX idx_posts_crisis_time      ON issue_cracker.posts (crisis_id, created_at);
CREATE INDEX idx_posts_platform         ON issue_cracker.posts (platform, created_at);
CREATE INDEX idx_comments_post          ON issue_cracker.comments (post_id, created_at);
CREATE INDEX idx_comments_crisis_time   ON issue_cracker.comments (crisis_id, created_at);
CREATE INDEX idx_analysis_crisis_type   ON issue_cracker.analysis_results (crisis_id, target_type);
CREATE INDEX idx_analysis_sentiment     ON issue_cracker.analysis_results (crisis_id, target_type, sentiment)
    WHERE sentiment IS NOT NULL;
CREATE UNIQUE INDEX idx_hourly_snapshots_pk
    ON issue_cracker.hourly_snapshots (crisis_id, hour_bucket);
