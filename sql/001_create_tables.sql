-- ==========================================
-- Issue Cracker — Schema DDL (간소화 버전)
-- PostgreSQL 16+ / pgvector 확장 필요
--
-- 테이블 구성 (5개):
--   1. issues             — 사용자 입력 단위 (분석 요청)
--   2. posts              — 원문 (전체 수집, issue 독립)
--   3. comments           — 댓글 (전체 수집, issue 독립)
--   4. analysis_results   — 감성 분류 결과 (issue 연결)
--   5. pipeline_runs      — 파이프라인 실행 이력 & 결과
--
-- 설계 원칙:
--   - posts/comments는 이슈와 무관하게 전체 수집
--   - analysis_results만 issue_id로 연결 (어떤 이슈 관점의 분석인지)
--   - posts/comments에 pgvector 임베딩 → 이슈별 관련 콘텐츠 검색
-- ==========================================

-- 🗑️ DROP (역순)
DROP TABLE IF EXISTS issue_cracker.pipeline_runs;
DROP TABLE IF EXISTS issue_cracker.analysis_results;
DROP TABLE IF EXISTS issue_cracker.comments;
DROP TABLE IF EXISTS issue_cracker.posts;
DROP TABLE IF EXISTS issue_cracker.issues;

DROP TYPE IF EXISTS issue_cracker.content_type;
DROP TYPE IF EXISTS issue_cracker.platform_type;

CREATE SCHEMA IF NOT EXISTS issue_cracker;
CREATE EXTENSION IF NOT EXISTS vector;

-- ==========================================
-- 📦 ENUM Types
-- ==========================================
CREATE TYPE issue_cracker.platform_type AS ENUM (
    'youtube', 'twitter', 'community', 'news', 'instagram', 'tiktok'
);

CREATE TYPE issue_cracker.content_type AS ENUM (
    'video', 'article', 'post', 'tweet', 'reel', 'short'
);


-- ==========================================
-- [1] 🚨 Issues — 사용자 입력 단위
-- ==========================================
CREATE TABLE issue_cracker.issues (
    issue_id        TEXT        PRIMARY KEY,
    user_input      TEXT        NOT NULL,               -- 사용자가 입력한 위기 상황 설명
    issue_type      TEXT        NOT NULL                -- SCCT 위기 유형
                    CHECK (issue_type IN ('victim', 'accidental', 'preventable')),
    status          TEXT        NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'analyzing', 'resolved', 'archived')),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    extra           JSONB       DEFAULT '{}'
);


-- ==========================================
-- [2] 📄 Posts — 원문 (전체 수집, issue 독립)
-- ==========================================
CREATE TABLE issue_cracker.posts (
    post_id         TEXT        PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    platform        issue_cracker.platform_type NOT NULL,
    content_type    issue_cracker.content_type  NOT NULL,
    url             TEXT,

    title           TEXT        NOT NULL,
    body            TEXT,

    author_id       TEXT        NOT NULL,
    author_name     TEXT,
    author_followers INTEGER    DEFAULT 0,

    view_count      INTEGER     DEFAULT 0,
    like_count      INTEGER     DEFAULT 0,
    share_count     INTEGER     DEFAULT 0,
    comment_count   INTEGER     DEFAULT 0,

    -- pgvector 임베딩 (title + body)
    embedding       vector(384),                        -- all-MiniLM-L6-v2 차원

    extra           JSONB       DEFAULT '{}'
);


-- ==========================================
-- [3] 💬 Comments — 댓글 (전체 수집, issue 독립)
-- ==========================================
CREATE TABLE issue_cracker.comments (
    comment_id          TEXT        PRIMARY KEY,
    post_id             TEXT        NOT NULL,
    parent_comment_id   TEXT,
    created_at          TIMESTAMPTZ NOT NULL,
    collected_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    body                TEXT        NOT NULL,

    author_id           TEXT        NOT NULL,
    author_name         TEXT,
    author_followers    INTEGER,

    like_count          INTEGER     DEFAULT 0,
    reply_count         INTEGER     DEFAULT 0,

    -- pgvector 임베딩 (body)
    embedding           vector(384),                    -- all-MiniLM-L6-v2 차원

    extra               JSONB       DEFAULT '{}'
);


-- ==========================================
-- [4] 📊 Analysis Results — 감성 분류 + 영향력 스코어
--     issue_id 연결: 같은 댓글도 이슈마다 다른 관점으로 분석 가능
-- ==========================================
CREATE TABLE issue_cracker.analysis_results (
    id              BIGSERIAL   PRIMARY KEY,
    issue_id        TEXT        NOT NULL,

    target_type     TEXT        NOT NULL CHECK (target_type IN ('post', 'comment')),
    target_id       TEXT        NOT NULL,

    -- 감성 분류
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

    UNIQUE (issue_id, target_type, target_id, model_version)
);


-- ==========================================
-- [5] 🔄 Pipeline Runs — 파이프라인 실행 이력 & 결과
-- ==========================================
CREATE TABLE issue_cracker.pipeline_runs (
    run_id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id            TEXT        NOT NULL,

    -- 입력 파라미터
    issue_type          TEXT        NOT NULL,
    crisis_context      TEXT,

    -- 실행 상태
    status              TEXT        NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    loop_count          SMALLINT    DEFAULT 0,

    -- 에이전트 산출물
    planner_instructions    TEXT,
    strategist_draft        TEXT,
    strategist_timeline     JSONB,
    analyst_draft           TEXT,
    draft_report            JSONB,
    review_feedback         TEXT,
    is_approved             BOOLEAN,

    -- NVI 예측 결과
    actual_nvi_history      JSONB,
    nvi_baseline_forecast   JSONB,
    nvi_mitigated_forecast  JSONB,

    -- 요약 지표
    alert_level             TEXT,
    baseline_nvi_bottom     REAL,
    mitigated_nvi_bottom    REAL,
    defense_effect          REAL,

    -- 타임스탬프
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);


-- ==========================================
-- 📊 Indexes
-- ==========================================

-- issues
CREATE INDEX idx_issues_status
    ON issue_cracker.issues (status, created_at DESC);

-- posts
CREATE INDEX idx_posts_platform_time
    ON issue_cracker.posts (platform, created_at DESC);
CREATE INDEX idx_posts_collected
    ON issue_cracker.posts (collected_at DESC);

-- posts: pgvector (코사인 유사도 검색)
CREATE INDEX idx_posts_embedding
    ON issue_cracker.posts
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- comments
CREATE INDEX idx_comments_post
    ON issue_cracker.comments (post_id, created_at);
CREATE INDEX idx_comments_collected
    ON issue_cracker.comments (collected_at DESC);

-- comments: pgvector (코사인 유사도 검색)
CREATE INDEX idx_comments_embedding
    ON issue_cracker.comments
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- analysis_results
CREATE INDEX idx_analysis_issue_type
    ON issue_cracker.analysis_results (issue_id, target_type);
CREATE INDEX idx_analysis_sentiment
    ON issue_cracker.analysis_results (issue_id, target_type, sentiment)
    WHERE sentiment IS NOT NULL;

-- pipeline_runs
CREATE INDEX idx_pipeline_runs_issue
    ON issue_cracker.pipeline_runs (issue_id, started_at DESC);
CREATE INDEX idx_pipeline_runs_status
    ON issue_cracker.pipeline_runs (status, started_at DESC);
