-- ==========================================
-- Issue Cracker - hourly_snapshots MV 수정
-- 분석 실패 댓글을 언급량에는 포함, 감성 비율에서는 제외
-- ==========================================

-- Materialized View는 ALTER 불가 → DROP & RECREATE
DROP MATERIALIZED VIEW IF EXISTS issue_cracker.hourly_snapshots;

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
    
    -- 비율 (분모 = 분석 성공 건수만)
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
