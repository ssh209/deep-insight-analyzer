-- ==========================================
-- Deep Insight Analyzer — Schema DDL
-- PostgreSQL 16+ / pgvector 확장 필요
--
-- 설계 원칙:
--   - Collector가 관리하는 수집 테이블(deep_insight.collected_doc 등)은 읽기 전용
--   - Analyzer 전용 테이블만 deep_insight 스키마에 생성
--   - issues 테이블이 분석 단위 — user_input + QueryBuilder 결과물 보관
--   - analysis_results가 감성 분류 결과 (doc_id / comment_id FK 참조)
--   - pipeline_runs가 파이프라인 실행 이력
--
-- Collector 테이블 참조 (읽기 전용, 여기서 DDL 생성 안 함):
--   deep_insight.collected_doc          — 수집 문서 (doc_id BIGINT PK)
--   deep_insight.collected_doc_comment  — YouTube 댓글 (comment_id BIGINT PK)
--   deep_insight.collected_doc_embedding — 임베딩 벡터 (별도 테이블)
--   deep_insight.collection_job         — 수집 작업 단위
-- ==========================================

-- 🗑️ DROP (역순)
DROP TABLE IF EXISTS deep_insight.pipeline_runs;
DROP TABLE IF EXISTS deep_insight.analysis_results;
DROP TABLE IF EXISTS deep_insight.issues;

CREATE SCHEMA IF NOT EXISTS deep_insight;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";


-- ==========================================
-- [1] 🚨 Issues — 분석 요청 단위
--
--     user_input: 사용자가 입력한 위기 상황 설명
--     QueryBuilder 결과물도 여기에 저장:
--       search_keywords, search_queries, search_time_hint
-- ==========================================
CREATE TABLE deep_insight.issues (
    issue_id        UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    user_input      TEXT        NOT NULL,               -- 사용자 입력 위기 상황 설명
    issue_type      TEXT        NOT NULL                -- SCCT 위기 유형
                    CHECK (issue_type IN ('victim', 'accidental', 'preventable')),
    status          TEXT        NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'analyzing', 'resolved', 'archived')),

    -- QueryBuilder 결과물 보관
    search_keywords     JSONB   DEFAULT '[]',           -- 핵심 키워드 배열
    search_queries      JSONB   DEFAULT '[]',           -- 벡터 검색 자연어 쿼리 배열
    search_time_hint    TEXT,                            -- 검색 시간 범위 힌트

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    extra           JSONB       DEFAULT '{}'
);


-- ==========================================
-- [2] 📊 Analysis Results — 감성 분류 + 영향력 스코어 + 원문 톤
--
--     target_type = 'doc'      → collected_doc.doc_id 참조 (영향력 스코어)
--     target_type = 'comment'  → collected_doc_comment.comment_id 참조 (감성 분류)
--     target_type = 'doc_tone' → collected_doc.doc_id 참조 (원문 톤 분류)
--
--     같은 대상도 issue마다 다른 관점으로 분석 가능
-- ==========================================
CREATE TABLE deep_insight.analysis_results (
    id              BIGSERIAL   PRIMARY KEY,
    issue_id        UUID        NOT NULL REFERENCES deep_insight.issues (issue_id) ON DELETE CASCADE,

    target_type     TEXT        NOT NULL CHECK (target_type IN ('doc', 'comment', 'doc_tone')),
    target_id       BIGINT      NOT NULL,               -- doc_id 또는 comment_id (BIGINT)

    -- 감성/톤 분류
    -- comment: positive/negative/neutral
    -- doc_tone: hostile/critical/neutral/sympathetic/supportive
    sentiment       TEXT        CHECK (sentiment IN (
        'positive', 'negative', 'neutral',
        'hostile', 'critical', 'sympathetic', 'supportive'
    )),
    sentiment_score REAL,

    -- 댓글 특화
    is_mockery      BOOLEAN     DEFAULT FALSE,
    is_advocate     BOOLEAN     DEFAULT FALSE,

    -- 원문 톤 특화 (저격/폭로 콘텐츠 여부)
    is_attack_content BOOLEAN   DEFAULT FALSE,

    -- 문서 특화 (영향력 점수)
    influence_score SMALLINT,

    -- 메타
    model_version   TEXT,
    analyzed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (issue_id, target_type, target_id, model_version)
);


-- ==========================================
-- [3] 🔄 Pipeline Runs — 파이프라인 실행 이력 & 결과
-- ==========================================
CREATE TABLE deep_insight.pipeline_runs (
    run_id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id            UUID        NOT NULL REFERENCES deep_insight.issues (issue_id) ON DELETE CASCADE,

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
    ON deep_insight.issues (status, created_at DESC);

-- analysis_results
CREATE INDEX idx_analysis_issue_type
    ON deep_insight.analysis_results (issue_id, target_type);
CREATE INDEX idx_analysis_sentiment
    ON deep_insight.analysis_results (issue_id, target_type, sentiment)
    WHERE sentiment IS NOT NULL;
CREATE INDEX idx_analysis_target
    ON deep_insight.analysis_results (target_type, target_id);

-- pipeline_runs
CREATE INDEX idx_pipeline_runs_issue
    ON deep_insight.pipeline_runs (issue_id, started_at DESC);
CREATE INDEX idx_pipeline_runs_status
    ON deep_insight.pipeline_runs (status, started_at DESC);
