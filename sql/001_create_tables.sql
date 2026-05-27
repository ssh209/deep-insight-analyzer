-- ==========================================
-- Issue Cracker - 원본 데이터 스키마
-- PostgreSQL DDL
-- ==========================================

-- 스키마 생성
CREATE SCHEMA IF NOT EXISTS issue_cracker;

-- ==========================================
-- 플랫폼 / 콘텐츠 유형 ENUM
-- ==========================================
CREATE TYPE issue_cracker.platform_type AS ENUM (
    'youtube', 'twitter', 'community', 'news', 'instagram', 'tiktok'
);

CREATE TYPE issue_cracker.content_type AS ENUM (
    'video', 'article', 'post', 'tweet', 'reel', 'short'
);

-- ==========================================
-- 📄 Posts (원문) - 발화원
-- 유튜브 영상, 뉴스 기사, 커뮤니티 글, 트윗 등
-- ==========================================
CREATE TABLE issue_cracker.posts (
    post_id         TEXT        PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- 출처
    platform        issue_cracker.platform_type NOT NULL,
    content_type    issue_cracker.content_type  NOT NULL,
    url             TEXT,
    
    -- 콘텐츠
    title           TEXT        NOT NULL,
    body            TEXT,                       -- 본문/영상 설명 (긴 텍스트)
    
    -- 작성자
    author_id       TEXT        NOT NULL,
    author_name     TEXT,
    author_followers INTEGER   DEFAULT 0,      -- 구독자/팔로워
    
    -- 반응 지표 (수집 시점 기준 스냅샷)
    view_count      INTEGER    DEFAULT 0,
    like_count      INTEGER    DEFAULT 0,
    share_count     INTEGER    DEFAULT 0,
    comment_count   INTEGER    DEFAULT 0,
    
    -- 분석 결과 (AnalyzerAgent가 채움)
    influence_score SMALLINT,                   -- 0: 일반, 1: 중형, 2: 대형
    
    -- 메타
    crisis_id       TEXT,                       -- 어떤 위기 건에 연결되는지
    extra           JSONB      DEFAULT '{}'     -- 플랫폼별 추가 데이터
);

-- ==========================================
-- 💬 Comments (댓글) - 반응
-- 원문에 달린 댓글/답글
-- ==========================================
CREATE TABLE issue_cracker.comments (
    comment_id          TEXT        PRIMARY KEY,
    post_id             TEXT        NOT NULL REFERENCES issue_cracker.posts(post_id),
    parent_comment_id   TEXT        REFERENCES issue_cracker.comments(comment_id),  -- 대댓글
    created_at          TIMESTAMPTZ NOT NULL,
    collected_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- 콘텐츠
    body                TEXT        NOT NULL,
    
    -- 작성자
    author_id           TEXT        NOT NULL,
    author_name         TEXT,
    author_followers    INTEGER,               -- 수집 가능한 경우만
    
    -- 반응 지표
    like_count          INTEGER    DEFAULT 0,
    reply_count         INTEGER    DEFAULT 0,
    
    -- 분석 결과 (AnalyzerAgent가 채움)
    sentiment           TEXT,                   -- 'positive', 'negative', 'neutral'
    is_mockery          BOOLEAN,                -- 조롱/풍자 여부
    is_advocate         BOOLEAN,                -- 적극 옹호 여부
    sentiment_score     REAL,                   -- -1.0 ~ 1.0 연속값
    
    -- 메타
    crisis_id           TEXT,
    extra               JSONB      DEFAULT '{}'
);

-- ==========================================
-- 📊 인덱스
-- ==========================================

-- Posts: 시간 범위 + 위기 건 조회
CREATE INDEX idx_posts_crisis_time 
    ON issue_cracker.posts (crisis_id, created_at);

CREATE INDEX idx_posts_platform 
    ON issue_cracker.posts (platform, created_at);

CREATE INDEX idx_posts_author_followers 
    ON issue_cracker.posts (author_followers DESC);

-- Comments: 포스트별 + 시간 범위 조회
CREATE INDEX idx_comments_post 
    ON issue_cracker.comments (post_id, created_at);

CREATE INDEX idx_comments_crisis_time 
    ON issue_cracker.comments (crisis_id, created_at);

-- Comments: 감성 분석 결과 기반 집계
CREATE INDEX idx_comments_sentiment 
    ON issue_cracker.comments (crisis_id, sentiment, is_mockery, is_advocate)
    WHERE sentiment IS NOT NULL;

-- ==========================================
-- 📋 hourly_snapshots (시간별 집계 뷰)
-- AnalyzerAgent가 산출하여 ForecasterAgent에 전달하는 형태
-- 
-- total_mentions: 분석 실패 포함 전체 (언급량)
-- negative_ratio 등: 분석 성공 건만 기준 (감성 비율)
-- ==========================================
CREATE MATERIALIZED VIEW issue_cracker.hourly_snapshots AS
SELECT
    c.crisis_id,
    date_trunc('hour', c.created_at)                            AS hour_bucket,
    
    -- 총 언급량 (분석 실패 포함)
    COUNT(*)                                                    AS total_mentions,
    
    -- 분석 완료 건수
    COUNT(*) FILTER (WHERE c.sentiment IS NOT NULL)              AS analyzed_count,
    
    -- 감성 분포 (절대 수)
    COUNT(*) FILTER (WHERE c.sentiment = 'negative')            AS negative_mentions,
    COUNT(*) FILTER (WHERE c.is_mockery = TRUE)                 AS mockery_mentions,
    COUNT(*) FILTER (WHERE c.is_advocate = TRUE)                AS advocate_mentions,
    
    -- 비율 (분모 = 분석 성공 건수)
    ROUND(
        COUNT(*) FILTER (WHERE c.sentiment = 'negative')::NUMERIC 
        / NULLIF(COUNT(*) FILTER (WHERE c.sentiment IS NOT NULL), 0), 3
    )                                                           AS negative_ratio,
    ROUND(
        COUNT(*) FILTER (WHERE c.is_mockery = TRUE)::NUMERIC 
        / NULLIF(COUNT(*) FILTER (WHERE c.sentiment IS NOT NULL), 0), 3
    )                                                           AS mockery_index,
    ROUND(
        COUNT(*) FILTER (WHERE c.is_advocate = TRUE)::NUMERIC 
        / NULLIF(COUNT(*) FILTER (WHERE c.sentiment IS NOT NULL), 0), 3
    )                                                           AS advocate_ratio,
    
    -- 인플루언서 영향 (해당 시간대 최대 영향력)
    COALESCE(
        MAX(p.influence_score) FILTER (WHERE p.influence_score IS NOT NULL), 0
    )                                                           AS influencer_impact
    
FROM issue_cracker.comments c
JOIN issue_cracker.posts p ON c.post_id = p.post_id
GROUP BY c.crisis_id, date_trunc('hour', c.created_at)
ORDER BY c.crisis_id, hour_bucket;

CREATE UNIQUE INDEX idx_hourly_snapshots_pk 
    ON issue_cracker.hourly_snapshots (crisis_id, hour_bucket);
