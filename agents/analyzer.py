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
        crisis_id = state["crisis_id"]
        print(f"\n>> [Analyzer] crisis_id={crisis_id} 분석 시작...")
        
        # 0. crises 상태 업데이트
        await self._update_crisis_status(crisis_id, 'analyzing')
        
        # 1. 미분석 댓글 로드
        comments = await self._load_unanalyzed(crisis_id)
        
        if not comments:
            print("   [SKIP] 미분석 댓글 없음 (이미 분석 완료)")
        else:
            print(f"   [LOAD] 미분석 댓글 {len(comments)}건 로드")
            
            # 2. 배치 감성 분류 + 3회 재시도
            success, failed = await self._analyze_with_retry(comments, crisis_id)
            print(f"   [DONE] 분석 성공 {success}건, 최종 실패 {failed}건")
        
        # 3. Posts 영향력 스코어링
        scored = await self._score_posts(crisis_id)
        print(f"   [SCORE] Posts 영향력 스코어링: {scored}건")
        
        # 4. MV 리프레시 + CSV 내보내기
        csv_path = await self._export_csv(crisis_id)
        print(f"   [EXPORT] CSV: {csv_path}")
        
        # 5. crises에 csv 경로 저장
        await self._finalize_crisis(crisis_id, csv_path)
        
        return {
            "input_csv_path": csv_path,
        }
    
    # ==========================================
    # Step 0. crises 상태 관리
    # ==========================================
    async def _update_crisis_status(self, crisis_id: str, status: str):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE issue_cracker.crises
                SET status = $2, updated_at = NOW()
                WHERE crisis_id = $1
            """, crisis_id, status)
    
    async def _finalize_crisis(self, crisis_id: str, csv_path: str):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE issue_cracker.crises
                SET input_csv_path = $2, updated_at = NOW()
                WHERE crisis_id = $1
            """, crisis_id, csv_path)
    
    # ==========================================
    # Step 1. 미분석 댓글 로드
    # analysis_results에 아직 없는 댓글만 조회
    # ==========================================
    async def _load_unanalyzed(self, crisis_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.comment_id, c.body, c.like_count, c.reply_count
                FROM issue_cracker.comments c
                LEFT JOIN issue_cracker.analysis_results ar
                    ON ar.target_type = 'comment'
                    AND ar.target_id = c.comment_id
                    AND ar.model_version = $2
                WHERE c.crisis_id = $1
                  AND ar.id IS NULL
                ORDER BY c.created_at
            """, crisis_id, self.model_name)
        return [dict(r) for r in rows]
    
    # ==========================================
    # Step 2. LLM 배치 감성 분류 + 재시도
    # ==========================================
    async def _analyze_with_retry(self, comments: list[dict], crisis_id: str) -> tuple[int, int]:
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
                tasks = [self._call_llm_and_save(batch, crisis_id) for batch in chunk]
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
    
    async def _call_llm_and_save(self, batch: list[dict], crisis_id: str) -> int:
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
                    (crisis_id, target_type, target_id,
                     sentiment, sentiment_score, is_mockery, is_advocate,
                     model_version)
                VALUES ($1, 'comment', $2, $3, $4, $5, $6, $7)
                ON CONFLICT (target_type, target_id, model_version) DO NOTHING
            """, [
                (crisis_id, a["comment_id"], a["sentiment"],
                 a["sentiment_score"], a["is_mockery"], a["is_advocate"],
                 self.model_name)
                for a in analyses
            ])
        
        return len(analyses)
    
    # ==========================================
    # Step 3. Posts 영향력 스코어링 (규칙 기반)
    # → analysis_results에 INSERT
    # ==========================================
    async def _score_posts(self, crisis_id: str) -> int:
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                INSERT INTO issue_cracker.analysis_results
                    (crisis_id, target_type, target_id, influence_score, model_version)
                SELECT
                    p.crisis_id,
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
                    AND ar.model_version = 'rule-based'
                WHERE p.crisis_id = $1
                  AND ar.id IS NULL
                ON CONFLICT (target_type, target_id, model_version) DO NOTHING
            """, crisis_id)
        return int(result.split()[-1]) if result else 0
    
    # ==========================================
    # Step 4. MV 리프레시 + CSV 내보내기
    # ==========================================
    async def _export_csv(self, crisis_id: str) -> str:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "REFRESH MATERIALIZED VIEW CONCURRENTLY issue_cracker.hourly_snapshots"
            )
            
            rows = await conn.fetch("""
                SELECT hour_bucket, total_mentions, analyzed_count,
                       negative_mentions, mockery_mentions, advocate_mentions,
                       negative_ratio, mockery_index, advocate_ratio,
                       influencer_impact
                FROM issue_cracker.hourly_snapshots
                WHERE crisis_id = $1
                ORDER BY hour_bucket
            """, crisis_id)
        
        df = pd.DataFrame([dict(r) for r in rows])
        
        # Feature Engineering: ForecasterAgent 입력 형태로 변환
        df = self._engineer_features(df)
        
        os.makedirs("data", exist_ok=True)
        csv_path = f"data/analyzed_{crisis_id}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """hourly_snapshots → ForecasterAgent 입력 CSV 형태로 변환."""
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
    
    async def run_batch(self, crisis_id: str, poll_interval: int = 60):
        """
        1. DB에서 미분석 댓글 로드
        2. JSONL 생성 → GCS 업로드
        3. Batch Job 제출
        4. Polling으로 완료 대기
        5. 결과 다운로드 → DB 반영
        """
        from google.cloud import storage, aiplatform
        
        comments = await self._load_unanalyzed(crisis_id)
        if not comments:
            print("   [BATCH] No unanalyzed comments")
            return
        
        print(f"   [BATCH] {len(comments)}건 Batch Prediction 시작...")
        
        # 1. JSONL 생성 (50건씩 배치)
        jsonl_lines = self._build_jsonl(comments)
        
        # 2. GCS 업로드
        gcs_input = f"gs://{self.gcs_bucket}/batch_input/{crisis_id}.jsonl"
        gcs_output = f"gs://{self.gcs_bucket}/batch_output/{crisis_id}/"
        
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.gcs_bucket)
        blob = bucket.blob(f"batch_input/{crisis_id}.jsonl")
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
        await self._apply_batch_results(gcs_output, crisis_id)
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
    
    async def _load_unanalyzed(self, crisis_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.comment_id, c.body, c.like_count, c.reply_count
                FROM issue_cracker.comments c
                LEFT JOIN issue_cracker.analysis_results ar
                    ON ar.target_type = 'comment'
                    AND ar.target_id = c.comment_id
                    AND ar.model_version = $2
                WHERE c.crisis_id = $1
                  AND ar.id IS NULL
                ORDER BY c.created_at
            """, crisis_id, self.model_name)
        return [dict(r) for r in rows]
    
    async def _apply_batch_results(self, gcs_output: str, crisis_id: str):
        """GCS 결과 JSONL 파싱 → analysis_results INSERT."""
        from google.cloud import storage
        
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
                        (crisis_id, target_type, target_id,
                         sentiment, sentiment_score, is_mockery, is_advocate,
                         model_version)
                    VALUES ($1, 'comment', $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (target_type, target_id, model_version) DO NOTHING
                """, [
                    (crisis_id, a["comment_id"], a["sentiment"],
                     a["sentiment_score"], a["is_mockery"], a["is_advocate"],
                     self.model_name)
                    for a in all_analyses
                ])
