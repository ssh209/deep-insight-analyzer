"""
AnalyzerAgent — 원본 텍스트(Posts/Comments) → 시계열 피처 변환

파이프라인 최전방에서:
1. DB에서 미분석 댓글 로드 (asyncpg)
2. LLM 배치 감성 분류 (Online API: asyncio.gather, 50건×5 병렬)
3. 3회 재시도 → 최종 실패분은 sentiment=NULL 유지 (언급량에만 포함)
4. Posts 영향력 스코어링 (규칙 기반)
5. Materialized View 리프레시 → CSV 내보내기 → ForecasterAgent 입력

이원화 전략:
  - 초기 대량 분석 (수만건): Vertex AI Batch Prediction (50% 할인, 비동기)
  - 실시간 증분 분석 (수백건): Online API (즉시 결과)
"""
import os
import json
import asyncio
import pandas as pd
import numpy as np
from typing import Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.cloud import storage, aiplatform

# ==========================================
# 📊 감성 분석 Structured Output 스키마
# ==========================================
class CommentAnalysis(BaseModel):
    comment_id: str = Field(description="분석 대상 댓글 ID")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="기업/이슈에 대한 태도"
    )
    is_mockery: bool = Field(description="조롱/풍자/비꼼/밈화 여부")
    is_advocate: bool = Field(description="기업 적극 옹호/방어 여부")
    sentiment_score: float = Field(description="-1.0(극부정) ~ 1.0(극긍정)")

class BatchAnalysisResult(BaseModel):
    results: list[CommentAnalysis]


# ==========================================
# 📋 감성 분류 시스템 프롬프트
# ==========================================
SENTIMENT_SYSTEM = (
    "당신은 한국어 여론 분석 전문가입니다. "
    "기업/이슈에 대한 온라인 댓글의 감성을 정밀하게 분류합니다."
)

SENTIMENT_PROMPT_TEMPLATE = """아래 댓글들의 감성을 분류하세요.

[분류 기준]
- sentiment: 해당 기업/이슈에 대한 태도 (positive/negative/neutral)
- is_mockery: 조롱, 풍자, 비꼼, 밈화 표현 여부 (ㅋㅋ+비판, 패러디, 비아냥 등)
  → is_mockery=true이면 반드시 sentiment=negative
- is_advocate: 기업을 적극 옹호/변호하는 발언 여부
  → is_advocate=true이면 반드시 sentiment=positive
- sentiment_score: -1.0(극도로 부정) ~ 1.0(극도로 긍정) 연속값

[댓글 목록]
{comments_text}
"""


class AnalyzerAgent:
    """
    원본 댓글 → 감성 분류 → 시계열 피처 변환 에이전트.
    
    asyncpg 풀을 주입받아 DB와 비동기 통신합니다.
    LangGraph 노드로 사용 시 run()이 async로 호출됩니다.
    """
    
    BATCH_SIZE = 50         # 1회 LLM 호출당 댓글 수
    CONCURRENT = 5          # 동시 배치 수
    MAX_RETRIES = 3         # 최대 재시도 횟수
    
    def __init__(self, client, model_name: str, db_pool):
        self.client = client
        self.model_name = model_name
        self.pool = db_pool     # asyncpg.Pool
    
    # ==========================================
    # 🚀 메인 실행 (LangGraph 노드)
    # ==========================================
    async def run(self, state: dict) -> dict:
        issue_id = state["issue_id"]
        post_ids = state.get("retrieved_post_ids", [])
        print(f"\n>> [Analyzer] issue_id={issue_id} 분석 시작 (posts {len(post_ids)}건)...")
        
        # 0. issues 상태 업데이트
        await self._update_issue_status(issue_id, 'analyzing')
        
        # 1. 미분석 댓글 로드 (Retriever가 확보한 posts 기준)
        comments = await self._load_unanalyzed(issue_id, post_ids)
        
        if not comments:
            print("   [SKIP] 미분석 댓글 없음 (이미 분석 완료)")
        else:
            print(f"   [LOAD] 미분석 댓글 {len(comments)}건 로드")
            
            # 2. 배치 감성 분류 + 3회 재시도
            success, failed = await self._analyze_with_retry(comments, issue_id)
            print(f"   [DONE] 분석 성공 {success}건, 최종 실패 {failed}건")
        
        # 3. Posts 영향력 스코어링
        scored = await self._score_posts(issue_id, post_ids)
        print(f"   [SCORE] Posts 영향력 스코어링: {scored}건")
        
        # 4. 집계 → CSV 내보내기
        csv_path = await self._export_csv(issue_id)
        print(f"   [EXPORT] CSV: {csv_path}")
        
        # 5. 여론 지형도 (Sentiment Landscape) 생성
        landscape = await self._build_sentiment_landscape(issue_id, post_ids)
        print(f"   [LANDSCAPE] 여론 지형도 생성 완료")
        
        # 6. 감성 타임라인 (Sentiment Timeline) 생성
        timeline = await self._build_sentiment_timeline(issue_id)
        print(f"   [TIMELINE] 감성 타임라인 생성 완료")
        
        # 7. KOL (Key Opinion Leader) 식별
        kols = await self._identify_kols(issue_id, post_ids)
        print(f"   [KOL] {len(kols)}명 식별 완료")
        
        return {
            "input_csv_path": csv_path,
            "sentiment_landscape": landscape,
            "sentiment_timeline": timeline,
            "key_opinion_leaders": kols,
        }
    
    # ==========================================
    # Step 0. issues 상태 관리
    # ==========================================
    async def _update_issue_status(self, issue_id: str, status: str):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE issue_cracker.issues
                SET status = $2, updated_at = NOW()
                WHERE issue_id = $1
            """, issue_id, status)
    
    # ==========================================
    # Step 1. 미분석 댓글 로드
    # analysis_results에 아직 없는 댓글만 조회
    # ==========================================
    async def _load_unanalyzed(self, issue_id: str, post_ids: list[str]) -> list[dict]:
        """Retriever가 확보한 posts의 댓글 중, 이 issue에서 미분석인 것만 로드."""
        if not post_ids:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.comment_id, c.body, c.like_count, c.reply_count
                FROM issue_cracker.comments c
                LEFT JOIN issue_cracker.analysis_results ar
                    ON ar.target_type = 'comment'
                    AND ar.target_id = c.comment_id
                    AND ar.issue_id = $1
                    AND ar.model_version = $3
                WHERE c.post_id = ANY($2)
                  AND ar.id IS NULL
                ORDER BY c.created_at
            """, issue_id, post_ids, self.model_name)
        return [dict(r) for r in rows]
    
    # ==========================================
    # Step 2. LLM 배치 감성 분류 + 재시도
    # ==========================================
    async def _analyze_with_retry(self, comments: list[dict], issue_id: str) -> tuple[int, int]:
        """3회 재시도 로직. 최종 실패분은 analysis_results에 미등록 → 언급량에만 포함."""
        batches = [
            comments[i:i + self.BATCH_SIZE] 
            for i in range(0, len(comments), self.BATCH_SIZE)
        ]
        
        total_success = 0
        pending = batches
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            if not pending:
                break
            
            failed_batches = []
            
            for chunk_start in range(0, len(pending), self.CONCURRENT):
                chunk = pending[chunk_start:chunk_start + self.CONCURRENT]
                tasks = [self._call_llm_and_save(batch, issue_id) for batch in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"   [RETRY {attempt}/{self.MAX_RETRIES}] "
                              f"배치 {len(chunk[i])}건 실패: {result}")
                        failed_batches.append(chunk[i])
                    else:
                        total_success += result
            
            pending = failed_batches
            
            if pending and attempt < self.MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
        
        total_failed = sum(len(b) for b in pending)
        return total_success, total_failed
    
    async def _call_llm_and_save(self, batch: list[dict], issue_id: str) -> int:
        """1개 배치(최대 50건) LLM 호출 → analysis_results INSERT. 성공 건수 반환."""
        comments_text = "\n".join(
            f"[{c['comment_id']}] {c['body'][:200]}" for c in batch
        )
        prompt = SENTIMENT_PROMPT_TEMPLATE.format(comments_text=comments_text)
        
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SENTIMENT_SYSTEM,
                response_mime_type="application/json",
                response_schema=BatchAnalysisResult,
                temperature=0.1,
            )
        )
        
        result = json.loads(response.text)
        analyses = result["results"]
        
        # analysis_results에 INSERT
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO issue_cracker.analysis_results
                    (issue_id, target_type, target_id,
                     sentiment, sentiment_score, is_mockery, is_advocate,
                     model_version)
                VALUES ($1, 'comment', $2, $3, $4, $5, $6, $7)
                ON CONFLICT (target_type, target_id, model_version) DO NOTHING
            """, [
                (issue_id, a["comment_id"], a["sentiment"],
                 a["sentiment_score"], a["is_mockery"], a["is_advocate"],
                 self.model_name)
                for a in analyses
            ])
        
        return len(analyses)
    
    # ==========================================
    # Step 3. Posts 영향력 스코어링 (규칙 기반)
    # → analysis_results에 INSERT
    # ==========================================
    async def _score_posts(self, issue_id: str, post_ids: list[str]) -> int:
        """Retriever가 확보한 posts의 영향력 스코어링."""
        if not post_ids:
            return 0
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                INSERT INTO issue_cracker.analysis_results
                    (issue_id, target_type, target_id, influence_score, model_version)
                SELECT
                    $1,
                    'post',
                    p.post_id,
                    CASE
                        WHEN p.author_followers >= 100000 OR p.view_count >= 500000 THEN 2
                        WHEN p.author_followers >= 10000  OR p.view_count >= 50000  THEN 1
                        ELSE 0
                    END,
                    'rule-based'
                FROM issue_cracker.posts p
                LEFT JOIN issue_cracker.analysis_results ar
                    ON ar.target_type = 'post'
                    AND ar.target_id = p.post_id
                    AND ar.issue_id = $1
                    AND ar.model_version = 'rule-based'
                WHERE p.post_id = ANY($2)
                  AND ar.id IS NULL
                ON CONFLICT (issue_id, target_type, target_id, model_version) DO NOTHING
            """, issue_id, post_ids)
        return int(result.split()[-1]) if result else 0
    
    # ==========================================
    # Step 4. MV 리프레시 + CSV 내보내기
    # ==========================================
    async def _export_csv(self, issue_id: str) -> str:
        """analysis_results에서 직접 시간당 집계 → CSV 내보내기."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    date_trunc('hour', c.created_at) AS hour_bucket,
                    COUNT(*)                                              AS total_mentions,
                    COUNT(ar.id)                                          AS analyzed_count,
                    COUNT(*) FILTER (WHERE ar.sentiment = 'negative')     AS negative_mentions,
                    COUNT(*) FILTER (WHERE ar.is_mockery = TRUE)          AS mockery_mentions,
                    COUNT(*) FILTER (WHERE ar.is_advocate = TRUE)         AS advocate_mentions,
                    ROUND(
                        COUNT(*) FILTER (WHERE ar.sentiment = 'negative')::NUMERIC
                        / NULLIF(COUNT(ar.id), 0), 3
                    ) AS negative_ratio,
                    ROUND(
                        COUNT(*) FILTER (WHERE ar.is_mockery = TRUE)::NUMERIC
                        / NULLIF(COUNT(ar.id), 0), 3
                    ) AS mockery_index,
                    ROUND(
                        COUNT(*) FILTER (WHERE ar.is_advocate = TRUE)::NUMERIC
                        / NULLIF(COUNT(ar.id), 0), 3
                    ) AS advocate_ratio,
                    COALESCE(
                        MAX(par.influence_score) FILTER (WHERE par.influence_score IS NOT NULL), 0
                    ) AS influencer_impact
                FROM issue_cracker.comments c
                LEFT JOIN issue_cracker.analysis_results ar
                    ON ar.target_type = 'comment'
                    AND ar.target_id = c.comment_id
                    AND ar.issue_id = $1
                JOIN issue_cracker.posts p ON c.post_id = p.post_id
                LEFT JOIN issue_cracker.analysis_results par
                    ON par.target_type = 'post'
                    AND par.target_id = p.post_id
                    AND par.issue_id = $1
                WHERE c.post_id IN (
                    SELECT DISTINCT target_id FROM issue_cracker.analysis_results
                    WHERE issue_id = $1 AND target_type = 'post'
                    UNION
                    SELECT DISTINCT post_id FROM issue_cracker.comments
                    WHERE comment_id IN (
                        SELECT target_id FROM issue_cracker.analysis_results
                        WHERE issue_id = $1 AND target_type = 'comment'
                    )
                )
                GROUP BY date_trunc('hour', c.created_at)
                ORDER BY hour_bucket
            """, issue_id)
        
        df = pd.DataFrame([dict(r) for r in rows])
        
        # Feature Engineering: ForecasterAgent 입력 형태로 변환
        df = self._engineer_features(df)
        
        os.makedirs("data", exist_ok=True)
        csv_path = f"data/analyzed_{issue_id}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간당 집계 데이터 → ForecasterAgent 입력 CSV 형태로 변환."""
        if df.empty:
            return df
        
        # 시간 기반 피처
        df["Datetime"] = pd.to_datetime(df["hour_bucket"])
        start = df["Datetime"].min()
        df["Hours_Since_Start"] = (df["Datetime"] - start).dt.total_seconds() / 3600
        
        # 컬럼명 매핑 (ForecasterAgent 기대 형식)
        df["Raw_Total_Mentions"] = df["total_mentions"].astype(float)
        df["Negative_Ratio"] = df["negative_ratio"].astype(float).fillna(0)
        df["Mockery_Index"] = df["mockery_index"].astype(float).fillna(0)
        df["Advocate_Ratio"] = df["advocate_ratio"].astype(float).fillna(0)
        df["Influencer_Impact"] = df["influencer_impact"].astype(int)
        
        # Negative_Momentum: 부정 건수 시간당 변화량
        df["Negative_Momentum"] = df["negative_mentions"].astype(float).diff().fillna(0)
        
        # Company_Action_Type: 분석 단계에서는 0 (무대응 기본)
        df["Company_Action_Type"] = 0
        
        # NVI 산출 (Analyzer 단계의 실측 NVI)
        df["Actual_NVI"] = self._compute_nvi(df)
        
        # 필요 컬럼만 선별
        output_cols = [
            "Datetime", "Hours_Since_Start", "Company_Action_Type",
            "Influencer_Impact", "Raw_Total_Mentions",
            "Negative_Ratio", "Mockery_Index", "Advocate_Ratio",
            "Negative_Momentum", "Actual_NVI"
        ]
        return df[output_cols]
    
    def _compute_nvi(self, df: pd.DataFrame) -> pd.Series:
        """감성 지표 기반 NVI(Net Valence Index) 산출."""
        nvi = 0.5  # 초기 평형점
        nvi_values = []
        
        for _, row in df.iterrows():
            neg = float(row.get("Negative_Ratio", 0))
            mock = float(row.get("Mockery_Index", 0))
            adv = float(row.get("Advocate_Ratio", 0))
            
            # 시간당 변화량
            penalty = neg * 0.3 + mock * 0.2
            bonus = adv * 0.25
            reversion = (0.5 - nvi) * 0.01  # 평형 회귀
            
            nvi = nvi - penalty + bonus + reversion
            nvi = max(0.1, min(1.0, nvi))
            nvi_values.append(round(nvi, 3))
        
        return pd.Series(nvi_values)

    # ==========================================
    # Step 5. 여론 지형도 (Sentiment Landscape)
    # ==========================================
    async def _build_sentiment_landscape(self, issue_id: str, post_ids: list[str]) -> dict:
        """감성 분석 결과를 집계하여 여론 지형도 생성.
        
        출력:
          overview: 전체 감성 비율
          top_negative_themes: 부정 여론 핵심 주제 (LLM 클러스터링)
          top_mockery_themes: 조롱/밈 주제
        """
        async with self.pool.acquire() as conn:
            # 전체 감성 비율 집계
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) AS total_analyzed,
                    COUNT(*) FILTER (WHERE sentiment = 'negative') AS negative_count,
                    COUNT(*) FILTER (WHERE sentiment = 'positive') AS positive_count,
                    COUNT(*) FILTER (WHERE sentiment = 'neutral') AS neutral_count,
                    COUNT(*) FILTER (WHERE is_mockery = TRUE) AS mockery_count,
                    COUNT(*) FILTER (WHERE is_advocate = TRUE) AS advocate_count
                FROM issue_cracker.analysis_results
                WHERE issue_id = $1 AND target_type = 'comment'
            """, issue_id)
            
            total = stats["total_analyzed"] or 1
            overview = {
                "total_analyzed": total,
                "negative_ratio": round(stats["negative_count"] / total, 3),
                "positive_ratio": round(stats["positive_count"] / total, 3),
                "neutral_ratio": round(stats["neutral_count"] / total, 3),
                "mockery_ratio": round(stats["mockery_count"] / total, 3),
                "advocate_ratio": round(stats["advocate_count"] / total, 3),
            }
            
            # 부정 댓글 상위 N건 추출 (LLM 클러스터링용)
            neg_comments = await conn.fetch("""
                SELECT c.body, c.like_count
                FROM issue_cracker.analysis_results ar
                JOIN issue_cracker.comments c ON ar.target_id = c.comment_id
                WHERE ar.issue_id = $1
                  AND ar.target_type = 'comment'
                  AND ar.sentiment = 'negative'
                ORDER BY c.like_count DESC
                LIMIT 50
            """, issue_id)
            
            # 조롱 댓글 상위 N건
            mock_comments = await conn.fetch("""
                SELECT c.body, c.like_count
                FROM issue_cracker.analysis_results ar
                JOIN issue_cracker.comments c ON ar.target_id = c.comment_id
                WHERE ar.issue_id = $1
                  AND ar.target_type = 'comment'
                  AND ar.is_mockery = TRUE
                ORDER BY c.like_count DESC
                LIMIT 20
            """, issue_id)
        
        # LLM으로 부정 주제 클러스터링
        top_negative_themes = await self._cluster_themes(
            [dict(r) for r in neg_comments], "부정 여론"
        )
        top_mockery_themes = await self._cluster_themes(
            [dict(r) for r in mock_comments], "조롱/밈"
        )
        
        return {
            "overview": overview,
            "top_negative_themes": top_negative_themes,
            "top_mockery_themes": top_mockery_themes,
        }
    
    async def _cluster_themes(self, comments: list[dict], category: str) -> list[dict]:
        """댓글 목록을 LLM으로 클러스터링하여 주제별 요약."""
        if not comments:
            return []
        
        comments_text = "\n".join(
            f"[{i+1}] (likes:{c.get('like_count', 0)}) {c['body'][:150]}"
            for i, c in enumerate(comments[:30])
        )
        
        prompt = f"""[{category}] 댓글들을 분석하여 핵심 주제 3~5개로 클러스터링하세요.

각 클러스터에 대해 JSON 배열로 답하세요:
[{{
  "theme": "주제명 (한글, 10자 이내)",
  "estimated_ratio": 0.35,
  "representative_comment": "대표 댓글 원문 (원본 그대로)",
  "intensity": "high" | "medium" | "low"
}}]

[댓글 목록]
{comments_text}"""
        
        try:
            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                )
            )
            return json.loads(res.text)
        except Exception as e:
            print(f"   [WARN] 테마 클러스터링 실패: {e}")
            return []

    # ==========================================
    # Step 6. 감성 타임라인 (Sentiment Timeline)
    # ==========================================
    async def _build_sentiment_timeline(self, issue_id: str) -> dict:
        """시간별 감성 구성 + 이벤트 마커를 생성.
        
        출력:
          history: [각 시간 버킷의 감성 분포]
          events: [주요 이벤트 마커 (KOL 저격, 기업 대응 등)]
        """
        async with self.pool.acquire() as conn:
            # 시간별 감성 구성
            hourly = await conn.fetch("""
                SELECT
                    date_trunc('hour', c.created_at) AS hour_bucket,
                    COUNT(*) AS total_mentions,
                    ROUND(
                        COUNT(*) FILTER (WHERE ar.sentiment = 'negative')::NUMERIC
                        / NULLIF(COUNT(ar.id), 0), 3
                    ) AS negative_ratio,
                    ROUND(
                        COUNT(*) FILTER (WHERE ar.is_mockery = TRUE)::NUMERIC
                        / NULLIF(COUNT(ar.id), 0), 3
                    ) AS mockery_ratio,
                    ROUND(
                        COUNT(*) FILTER (WHERE ar.is_advocate = TRUE)::NUMERIC
                        / NULLIF(COUNT(ar.id), 0), 3
                    ) AS advocate_ratio
                FROM issue_cracker.comments c
                JOIN issue_cracker.analysis_results ar
                    ON ar.target_type = 'comment'
                    AND ar.target_id = c.comment_id
                    AND ar.issue_id = $1
                GROUP BY date_trunc('hour', c.created_at)
                ORDER BY hour_bucket
            """, issue_id)
            
            # 이벤트 마커: 인플루언서 저격 (influence_score >= 2인 posts)
            events = await conn.fetch("""
                SELECT
                    p.created_at,
                    p.author_name,
                    p.platform,
                    p.title,
                    p.view_count,
                    p.author_followers
                FROM issue_cracker.posts p
                JOIN issue_cracker.analysis_results ar
                    ON ar.target_type = 'post'
                    AND ar.target_id = p.post_id
                    AND ar.issue_id = $1
                WHERE ar.influence_score >= 2
                ORDER BY p.created_at
            """, issue_id)
        
        # 과거 데이터 조립
        if not hourly:
            first_hour = None
        else:
            first_hour = hourly[0]["hour_bucket"]
        
        history = []
        for row in hourly:
            hour_offset = 0
            if first_hour:
                hour_offset = int((row["hour_bucket"] - first_hour).total_seconds() / 3600)
            history.append({
                "hour": hour_offset,
                "timestamp": row["hour_bucket"].isoformat(),
                "total_mentions": row["total_mentions"],
                "negative_ratio": float(row["negative_ratio"] or 0),
                "mockery_ratio": float(row["mockery_ratio"] or 0),
                "advocate_ratio": float(row["advocate_ratio"] or 0),
            })
        
        # 이벤트 마커 조립
        event_markers = []
        for evt in events:
            hour_offset = 0
            if first_hour:
                hour_offset = int((evt["created_at"] - first_hour).total_seconds() / 3600)
            event_markers.append({
                "hour": hour_offset,
                "timestamp": evt["created_at"].isoformat(),
                "type": "influencer_hit",
                "label": f"{evt['author_name']} ({evt['platform']}) - {evt['title'][:40]}",
                "view_count": evt["view_count"],
                "followers": evt["author_followers"],
            })
        
        return {
            "history": history,
            "events": event_markers,
        }

    # ==========================================
    # Step 7. KOL (Key Opinion Leader) 식별
    # ==========================================
    async def _identify_kols(self, issue_id: str, post_ids: list[str]) -> list[dict]:
        """영향력 높은 작성자(KOL)를 식별하고 LLM으로 성향/대응 권고를 생성.
        
        식별 기준:
          - influence_score >= 1 (followers >= 10K 또는 views >= 50K)
          - 조회수 순 정렬, 상위 10명
        
        출력:
          [{"author_name", "platform", "followers", "stance", 
            "influence_score", "key_content", "view_count", 
            "estimated_reach", "risk_assessment"}, ...]
        """
        if not post_ids:
            return []
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    p.author_name,
                    p.platform::TEXT,
                    p.author_followers,
                    p.view_count,
                    p.share_count,
                    p.title,
                    p.body,
                    p.url,
                    ar.influence_score
                FROM issue_cracker.posts p
                JOIN issue_cracker.analysis_results ar
                    ON ar.target_type = 'post'
                    AND ar.target_id = p.post_id
                    AND ar.issue_id = $1
                WHERE p.post_id = ANY($2)
                  AND ar.influence_score >= 1
                ORDER BY p.view_count DESC
                LIMIT 10
            """, issue_id, post_ids)
        
        if not rows:
            return []
        
        # LLM으로 각 KOL의 성향(stance)과 대응 권고(risk_assessment) 생성
        kol_texts = []
        for i, row in enumerate(rows):
            body_snippet = (row["body"] or "")[:200]
            kol_texts.append(
                f"[{i+1}] author={row['author_name']}, platform={row['platform']}, "
                f"followers={row['author_followers']:,}, views={row['view_count']:,}, "
                f"title=\"{row['title'][:80]}\"\n    내용: {body_snippet}"
            )
        
        prompt = f"""아래 온라인 콘텐츠 작성자들의 이슈에 대한 성향(stance)과 리스크 평가를 분석하세요.

각 작성자에 대해 JSON 배열로 답하세요:
[{{
  "index": 1,
  "stance": "hostile" | "neutral" | "supportive",
  "risk_assessment": "해당 작성자의 향후 행동 예측 및 대응 권고 (1~2문장)"
}}]

stance 기준:
- hostile: 기업에 대해 적대적/비판적
- neutral: 중립적/사실 전달
- supportive: 기업 옹호/우호적

[작성자 목록]
{chr(10).join(kol_texts)}"""
        
        try:
            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                )
            )
            stance_results = json.loads(res.text)
        except Exception as e:
            print(f"   [WARN] KOL 성향 분석 실패: {e}")
            stance_results = []
        
        # 성향 결과를 KOL 데이터에 병합
        stance_map = {s["index"]: s for s in stance_results if isinstance(s, dict)}
        
        kols = []
        for i, row in enumerate(rows):
            stance_info = stance_map.get(i + 1, {})
            estimated_reach = row["view_count"] + (row["share_count"] or 0) * 50
            
            kols.append({
                "author_name": row["author_name"],
                "platform": row["platform"],
                "followers": row["author_followers"],
                "view_count": row["view_count"],
                "influence_score": row["influence_score"],
                "key_content": row["title"][:100],
                "url": row["url"],
                "estimated_reach": estimated_reach,
                "stance": stance_info.get("stance", "unknown"),
                "risk_assessment": stance_info.get("risk_assessment", ""),
            })
        
        return kols

# ==========================================
# 🔄 Vertex AI Batch Prediction (초기 대량 분석용)
# ==========================================
class BatchAnalyzer:
    """
    Vertex AI Batch Prediction을 사용한 대량 분석.
    GCS에 JSONL 업로드 → Batch Job 제출 → Polling → 결과 반영.
    
    실시간 파이프라인이 아닌 초기 대량 투입 시 사용합니다.
    """
    
    def __init__(self, client, model_name: str, db_pool,
                 gcs_bucket: str, project_id: str, location: str = "us-central1"):
        self.client = client
        self.model_name = model_name
        self.pool = db_pool
        self.gcs_bucket = gcs_bucket
        self.project_id = project_id
        self.location = location
    
    async def run_batch(self, issue_id: str, poll_interval: int = 60):
        """
        1. DB에서 미분석 댓글 로드
        2. JSONL 생성 → GCS 업로드
        3. Batch Job 제출
        4. Polling으로 완료 대기
        5. 결과 다운로드 → DB 반영
        """

        
        comments = await self._load_unanalyzed(issue_id)
        if not comments:
            print("   [BATCH] No unanalyzed comments")
            return
        
        print(f"   [BATCH] {len(comments)}건 Batch Prediction 시작...")
        
        # 1. JSONL 생성 (50건씩 배치)
        jsonl_lines = self._build_jsonl(comments)
        
        # 2. GCS 업로드
        gcs_input = f"gs://{self.gcs_bucket}/batch_input/{issue_id}.jsonl"
        gcs_output = f"gs://{self.gcs_bucket}/batch_output/{issue_id}/"
        
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.gcs_bucket)
        blob = bucket.blob(f"batch_input/{issue_id}.jsonl")
        blob.upload_from_string("\n".join(jsonl_lines), content_type="application/jsonl")
        
        # 3. Batch Job 제출
        aiplatform.init(project=self.project_id, location=self.location)
        
        batch_job = aiplatform.BatchPredictionJob.create(
            model_name=f"publishers/google/models/{self.model_name}",
            gcs_source=gcs_input,
            gcs_destination_prefix=gcs_output,
            sync=False,
        )
        
        print(f"   [BATCH] Job submitted: {batch_job.resource_name}")
        
        # 4. Polling
        while batch_job.state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            await asyncio.sleep(poll_interval)
            batch_job.refresh()
            print(f"   [BATCH] Status: {batch_job.state}")
        
        if batch_job.state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Batch job failed: {batch_job.state}")
        
        # 5. 결과 반영
        await self._apply_batch_results(gcs_output, issue_id)
        print(f"   [BATCH] {len(comments)}건 분석 완료")
    
    def _build_jsonl(self, comments: list[dict]) -> list[str]:
        """댓글들을 50건씩 묶어 Batch JSONL 형식으로 변환."""
        lines = []
        for i in range(0, len(comments), 50):
            batch = comments[i:i + 50]
            comments_text = "\n".join(
                f"[{c['comment_id']}] {c['body'][:200]}" for c in batch
            )
            prompt = SENTIMENT_PROMPT_TEMPLATE.format(comments_text=comments_text)
            
            request = {
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "systemInstruction": {"parts": [{"text": SENTIMENT_SYSTEM}]},
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": BatchAnalysisResult.model_json_schema(),
                        "temperature": 0.1,
                    }
                }
            }
            lines.append(json.dumps(request, ensure_ascii=False))
        return lines
    
    async def _load_unanalyzed(self, issue_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.comment_id, c.body, c.like_count, c.reply_count
                FROM issue_cracker.comments c
                LEFT JOIN issue_cracker.analysis_results ar
                    ON ar.target_type = 'comment'
                    AND ar.target_id = c.comment_id
                    AND ar.model_version = $2
                WHERE c.issue_id = $1
                  AND ar.id IS NULL
                ORDER BY c.created_at
            """, issue_id, self.model_name)
        return [dict(r) for r in rows]
    
    async def _apply_batch_results(self, gcs_output: str, issue_id: str):
        """GCS 결과 JSONL 파싱 → analysis_results INSERT."""

        
        storage_client = storage.Client(project=self.project_id)
        bucket_name = gcs_output.split("/")[2]
        prefix = "/".join(gcs_output.split("/")[3:])
        
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        all_analyses = []
        for blob in blobs:
            if not blob.name.endswith(".jsonl"):
                continue
            content = blob.download_as_text()
            for line in content.strip().split("\n"):
                result = json.loads(line)
                response_text = result.get("response", {}).get("candidates", [{}])[0] \
                    .get("content", {}).get("parts", [{}])[0].get("text", "{}")
                parsed = json.loads(response_text)
                all_analyses.extend(parsed.get("results", []))
        
        if all_analyses:
            async with self.pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO issue_cracker.analysis_results
                        (issue_id, target_type, target_id,
                         sentiment, sentiment_score, is_mockery, is_advocate,
                         model_version)
                    VALUES ($1, 'comment', $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (target_type, target_id, model_version) DO NOTHING
                """, [
                    (issue_id, a["comment_id"], a["sentiment"],
                     a["sentiment_score"], a["is_mockery"], a["is_advocate"],
                     self.model_name)
                    for a in all_analyses
                ])
