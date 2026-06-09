-- ==========================================
-- 002: analysis_results 테이블 — 원문 톤 분석 지원 확장
--
-- 변경 사항:
--   1. target_type CHECK 제약에 'doc_tone' 추가
--   2. sentiment CHECK 제약에 원문 톤 5분류 추가
--   3. is_attack_content 컬럼 추가
-- ==========================================

-- 1. target_type CHECK 제약 변경
ALTER TABLE deep_insight.analysis_results
    DROP CONSTRAINT IF EXISTS analysis_results_target_type_check;
ALTER TABLE deep_insight.analysis_results
    ADD CONSTRAINT analysis_results_target_type_check
    CHECK (target_type IN ('doc', 'comment', 'doc_tone'));

-- 2. sentiment CHECK 제약 변경 (톤 5분류 추가)
ALTER TABLE deep_insight.analysis_results
    DROP CONSTRAINT IF EXISTS analysis_results_sentiment_check;
ALTER TABLE deep_insight.analysis_results
    ADD CONSTRAINT analysis_results_sentiment_check
    CHECK (sentiment IN (
        'positive', 'negative', 'neutral',
        'hostile', 'critical', 'sympathetic', 'supportive'
    ));

-- 3. is_attack_content 컬럼 추가 (이미 존재하면 무시)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'deep_insight'
          AND table_name = 'analysis_results'
          AND column_name = 'is_attack_content'
    ) THEN
        ALTER TABLE deep_insight.analysis_results
            ADD COLUMN is_attack_content BOOLEAN DEFAULT FALSE;
    END IF;
END $$;
