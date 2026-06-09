"""
AnalyzerAgent — 원본 텍스트(Posts/Comments) → 감성 분류 → 시계열 피처 변환

설계 원칙: DB SELECT 없음. Retriever가 state에 담은 메모리 데이터만 소비.

파이프라인 흐름:
1. state["retrieved_comments"]에서 댓글 로드 (메모리)
2. LLM 배치 감성 분류 (Online API: asyncio.gather, 50건×5 병렬)
3. 3회 재시도 → 최종 실패분은 sentiment=NULL 유지 (언급량에만 포함)
4. Posts 영향력 스코어링 (규칙 기반, 메모리)
5. 분석 결과 집계 → CSV 내보내기 → ForecasterAgent 입력
6. 여론 지형도 / 감성 타임라인 / KOL 식별 (메모리 + LLM)

이원화 전략:
  - 초기 대량 분석 (수만건): Vertex AI Batch Prediction (50% 할인, 비동기)
  - 실시간 증분 분석 (수백건): Online API (즉시 결과)
"""
import os
import json
import uuid
import asyncio
import pandas as pd
import numpy as np
from typing import Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
# google.cloud.storage, aiplatform → 배치 메서드에서 lazy import (시작 시간 최적화)
from config import OUTPUT_DIR

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


class DocAnalysis(BaseModel):
    doc_id: int = Field(description="분석 대상 문서 ID")
    tone: Literal["hostile", "critical", "neutral", "sympathetic", "supportive"] = Field(
        description="원문의 기업/이슈에 대한 논조"
    )
    is_attack_content: bool = Field(description="저격 영상, 폭로 게시글 등 직접 공격형 콘텐츠 여부")
    tone_score: float = Field(description="-1.0(극부정) ~ 1.0(극긍정)")

class DocBatchAnalysisResult(BaseModel):
    results: list[DocAnalysis]


class ThemeCluster(BaseModel):
    theme: str = Field(description="주제명 (한글, 10자 이내)")
    estimated_ratio: float = Field(description="해당 주제의 추정 비율 (0~1)")
    representative_comment: str = Field(description="대표 댓글 원문 (원본 그대로)")
    intensity: Literal["high", "medium", "low"] = Field(
        description="해당 주제의 여론 강도"
    )

class ThemeClusterList(BaseModel):
    clusters: list[ThemeCluster]


class KOLStanceAnalysis(BaseModel):
    index: int = Field(description="작성자 번호 (1부터 시작)")
    stance: Literal["hostile", "neutral", "supportive"] = Field(
        description="기업/이슈에 대한 성향"
    )
    risk_assessment: str = Field(
        description="해당 작성자의 향후 행동 예측 및 대응 권고 (1~2문장)"
    )

class KOLStanceList(BaseModel):
    results: list[KOLStanceAnalysis]


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

# ==========================================
# 📋 원문 톤 분류 시스템 프롬프트
# ==========================================
DOC_TONE_SYSTEM = (
    "당신은 한국어 미디어 논조 분석 전문가입니다. "
    "뉴스 기사, 블로그, 커뮤니티 게시글, SNS 포스트의 기업/이슈에 대한 논조를 분류합니다."
)

DOC_TONE_PROMPT_TEMPLATE = """아래 원문들의 논조를 분류하세요.

[분류 기준]
- tone: 기업/이슈에 대한 원문의 논조
  hostile    = 적대적 공격 (유튜버 저격, 커뮤니티 마녀사냥, 악의적 폭로)
  critical   = 비판적이지만 사실 기반 (언론 부정 보도, 문제 제기)
  neutral    = 중립 보도/정보 전달 (단순 사실 전달, 무관한 언급)
  sympathetic = 동정적/이해 표현 (피해자 시각, 상황 이해)
  supportive = 적극 옹호/방어 (기업 방어, 긍정적 재평가)
- is_attack_content: 저격 영상, 폭로 게시글 등 직접 공격형 콘텐츠 여부
- tone_score: -1.0(극도로 적대적) ~ 1.0(극도로 옹호적) 연속값

[원문 목록]
{docs_text}
"""


class AnalyzerAgent:
    """
    원본 댓글 → 감성 분류 → 시계열 피처 변환 에이전트.
    
    DB 의존성 없음. Retriever가 state에 담은 메모리 데이터만 소비.
    LangGraph 노드로 사용 시 run()이 async로 호출됩니다.
    
    듀얼 모델 전략:
      - model_name (Flash): 대량 감성 분류 배치 → 비용 최적화
      - deep_model_name (Pro): 테마 클러스터링, KOL 성향 분석 → 품질 우선
    """
    
    BATCH_SIZE = 50         # 1회 LLM 호출당 댓글 수
    CONCURRENT = 5          # 동시 배치 수
    MAX_RETRIES = 3         # 최대 재시도 횟수
    
    def __init__(self, client, model_name: str, db_pool=None,
                 deep_model_name: str = None):
        self.client = client
        self.model_name = model_name            # Flash: 감성 분류 배치
        self.deep_model_name = (                # Pro: 테마/KOL 심층 분석
            deep_model_name or model_name
        )
        self.pool = db_pool     # asyncpg.Pool (INSERT 전용, SELECT는 Retriever에서)
    
    # ==========================================
    # 🚀 메인 실행 (LangGraph 노드)
    # ==========================================
    async def run(self, state: dict) -> dict:
        issue_id = state.get("issue_id", "unknown")
        docs = state.get("retrieved_docs", [])
        comments = state.get("retrieved_comments", [])
        
        print(f"\n>> [Analyzer] issue_id={issue_id} 분석 시작 "
              f"(docs {len(docs)}건, comments {len(comments)}건)...")
        
        # 1. LLM 배치 감성 분류 + 3회 재시도 (댓글)
        if not comments:
            print("   [SKIP] 분석 대상 댓글 없음")
            analysis_results = []
        else:
            analysis_results, failed = await self._analyze_with_retry(comments)
            print(f"   [DONE] 댓글 감성 분석 성공 {len(analysis_results)}건, 최종 실패 {failed}건")
        
        # 1b. 원문 톤 분류 (Doc Tone Analysis)
        if not docs:
            print("   [SKIP] 분석 대상 원문 없음")
            doc_analysis_results = []
        else:
            doc_analysis_results, doc_failed = await self._analyze_docs(docs)
            print(f"   [DONE] 원문 톤 분석 성공 {len(doc_analysis_results)}건, 최종 실패 {doc_failed}건")
        
        # 2. 분석 결과 DB 저장 (analysis_results INSERT)
        if analysis_results and self.pool:
            saved = await self._save_analysis_results(analysis_results, issue_id)
            print(f"   [DB] 감성 분석 결과 {saved}건 INSERT")
        
        # 2b. 원문 톤 분석 결과 DB 저장
        if doc_analysis_results and self.pool:
            saved = await self._save_doc_tone_results(doc_analysis_results, issue_id)
            print(f"   [DB] 원문 톤 분석 결과 {saved}건 INSERT")
        
        # 3. Docs 영향력 스코어링 (규칙 기반, 메모리)
        doc_scores = self._score_docs(docs)
        print(f"   [SCORE] Docs 영향력 스코어링: {len(doc_scores)}건")
        
        # 4. Doc 스코어 DB 저장
        if doc_scores and self.pool:
            saved = await self._save_doc_scores(doc_scores, issue_id)
            print(f"   [DB] Doc 영향력 스코어 {saved}건 INSERT")
        
        # 5. 집계 → CSV 내보내기 (원문 톤 포함)
        csv_path = self._export_csv(
            comments, analysis_results, doc_scores, issue_id,
            docs=docs, doc_analysis_results=doc_analysis_results,
        )
        print(f"   [EXPORT] CSV: {csv_path}")
        
        # 6. 여론 지형도 (Sentiment Landscape) 생성
        landscape = await self._build_sentiment_landscape(
            comments, analysis_results
        )
        print(f"   [LANDSCAPE] 여론 지형도 생성 완료")
        
        # 7. 감성 타임라인 (Sentiment Timeline) 생성
        timeline = self._build_sentiment_timeline(
            comments, analysis_results, docs, doc_scores
        )
        print(f"   [TIMELINE] 감성 타임라인 생성 완료")
        
        # 8. KOL (Key Opinion Leader) 식별
        kols = await self._identify_kols(docs, doc_scores)
        print(f"   [KOL] {len(kols)}명 식별 완료")
        
        return {
            "input_csv_path": csv_path,
            "sentiment_landscape": landscape,
            "sentiment_timeline": timeline,
            "key_opinion_leaders": kols,
        }
    
    # ==========================================
    # Step 1. LLM 배치 감성 분류 + 재시도
    # ==========================================
    async def _analyze_with_retry(
        self, comments: list[dict]
    ) -> tuple[list[dict], int]:
        """3회 재시도 로직. 반환: (분석 결과 리스트, 실패 건수)."""
        batches = [
            comments[i:i + self.BATCH_SIZE] 
            for i in range(0, len(comments), self.BATCH_SIZE)
        ]
        
        all_results = []
        pending = batches
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            if not pending:
                break
            
            failed_batches = []
            
            for chunk_start in range(0, len(pending), self.CONCURRENT):
                chunk = pending[chunk_start:chunk_start + self.CONCURRENT]
                tasks = [self._call_llm(batch) for batch in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"   [RETRY {attempt}/{self.MAX_RETRIES}] "
                              f"배치 {len(chunk[i])}건 실패: {result}")
                        failed_batches.append(chunk[i])
                    else:
                        all_results.extend(result)
            
            pending = failed_batches
            
            # exponential back-off
            if pending and attempt < self.MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
        
        total_failed = sum(len(b) for b in pending)
        return all_results, total_failed
    
    async def _call_llm(self, batch: list[dict]) -> list[dict]:
        """1개 배치(최대 50건) LLM 호출. 분석 결과 리스트 반환."""
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
        return result["results"]
    
    # ==========================================
    # Step 1b. 분석 결과 DB 저장 (INSERT only)
    # ==========================================
    async def _save_analysis_results(self, analysis_results: list[dict], issue_id: str) -> int:
        """LLM 감성 분석 결과를 analysis_results 테이블에 INSERT."""
        uid = uuid.UUID(issue_id)
        rows = [
            (uid, int(a["comment_id"]), a["sentiment"],
             a["sentiment_score"], a["is_mockery"], a["is_advocate"],
             self.model_name)
            for a in analysis_results
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO deep_insight.analysis_results
                    (issue_id, target_type, target_id,
                     sentiment, sentiment_score, is_mockery, is_advocate,
                     model_version)
                VALUES ($1, 'comment', $2, $3, $4, $5, $6, $7)
                ON CONFLICT (issue_id, target_type, target_id, model_version) DO NOTHING
            """, rows)
        return len(rows)
    
    # ==========================================
    # Step 2. Posts 영향력 스코어링 (규칙 기반, 메모리)
    # ==========================================
    def _score_docs(self, docs: list[dict]) -> dict[int, int]:
        """docs 리스트 → {doc_id: influence_score} 매핑.
        
        influence_score:
          2 = 메가 인플루언서 (followers >= 100K or views >= 500K)
          1 = 인플루언서 (followers >= 10K or views >= 50K)
          0 = 일반
        """
        scores = {}
        for d in docs:
            followers = d.get("author_followers", 0) or 0
            views = d.get("view_count", 0) or 0
            if followers >= 100000 or views >= 500000:
                scores[d["doc_id"]] = 2
            elif followers >= 10000 or views >= 50000:
                scores[d["doc_id"]] = 1
            else:
                scores[d["doc_id"]] = 0
        return scores

    # ==========================================
    # Step 2b. Post 영향력 스코어 DB 저장 (INSERT only)
    # ==========================================
    async def _save_doc_scores(self, doc_scores: dict[int, int], issue_id: str) -> int:
        """규칙 기반 영향력 스코어를 analysis_results 테이블에 INSERT."""
        uid = uuid.UUID(issue_id)
        rows = [
            (uid, doc_id, score, 'rule-based')
            for doc_id, score in doc_scores.items()
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO deep_insight.analysis_results
                    (issue_id, target_type, target_id, influence_score, model_version)
                VALUES ($1, 'doc', $2, $3, $4)
                ON CONFLICT (issue_id, target_type, target_id, model_version) DO NOTHING
            """, rows)
        return len(rows)
    
    # ==========================================
    # Step 1c. 원문 톤 분류 (Doc Tone Analysis)
    # ==========================================
    DOC_BATCH_SIZE = 10  # 원문은 길어서 배치 크기 축소

    async def _analyze_docs(self, docs: list[dict]) -> tuple[list[dict], int]:
        """원문 톤 분류. title + snippet으로 논조를 판별.
        
        Returns: (분석 결과 리스트, 실패 건수)
        """
        batches = [
            docs[i:i + self.DOC_BATCH_SIZE]
            for i in range(0, len(docs), self.DOC_BATCH_SIZE)
        ]

        all_results = []
        pending = batches

        for attempt in range(1, self.MAX_RETRIES + 1):
            if not pending:
                break

            failed_batches = []

            for chunk_start in range(0, len(pending), self.CONCURRENT):
                chunk = pending[chunk_start:chunk_start + self.CONCURRENT]
                tasks = [self._call_doc_tone_llm(batch) for batch in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"   [DOC-RETRY {attempt}/{self.MAX_RETRIES}] "
                              f"배치 {len(chunk[i])}건 실패: {result}")
                        failed_batches.append(chunk[i])
                    else:
                        all_results.extend(result)

            pending = failed_batches
            if pending and attempt < self.MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)

        total_failed = sum(len(b) for b in pending)
        return all_results, total_failed

    async def _call_doc_tone_llm(self, batch: list[dict]) -> list[dict]:
        """원문 배치 LLM 호출. title + snippet (최대 200자)로 톤 분류."""
        docs_text = "\n".join(
            f"[{d['doc_id']}] [{d.get('channel', '')}] {d.get('title', '')} — {(d.get('snippet', '') or '')[:200]}"
            for d in batch
        )
        prompt = DOC_TONE_PROMPT_TEMPLATE.format(docs_text=docs_text)

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=DOC_TONE_SYSTEM,
                response_mime_type="application/json",
                response_schema=DocBatchAnalysisResult,
                temperature=0.1,
            )
        )

        result = json.loads(response.text)
        return result["results"]

    async def _save_doc_tone_results(self, doc_analysis_results: list[dict], issue_id: str) -> int:
        """원문 톤 분석 결과를 analysis_results 테이블에 INSERT."""
        uid = uuid.UUID(issue_id)
        rows = [
            (uid, int(a["doc_id"]), a["tone"],
             a["tone_score"], a["is_attack_content"],
             self.model_name)
            for a in doc_analysis_results
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO deep_insight.analysis_results
                    (issue_id, target_type, target_id,
                     sentiment, sentiment_score, is_attack_content,
                     model_version)
                VALUES ($1, 'doc_tone', $2, $3, $4, $5, $6)
                ON CONFLICT (issue_id, target_type, target_id, model_version) DO NOTHING
            """, rows)
        return len(rows)

    # ==========================================
    # Step 3. 분석 결과 집계 → CSV 내보내기
    # ==========================================
    def _export_csv(
        self,
        comments: list[dict],
        analysis_results: list[dict],
        doc_scores: dict[int, int],
        issue_id: str,
        docs: list[dict] = None,
        doc_analysis_results: list[dict] = None,
    ) -> str:
        """메모리 데이터로 시간당 집계 → ForecasterAgent 입력 CSV 생성.
        
        댓글 감성 + 원문 톤을 모두 집계하여 확장 피처 CSV를 생성합니다.
        """
        docs = docs or []
        doc_analysis_results = doc_analysis_results or []
        
        # 댓글+원문 모두 없으면 빈 csv 파일 생성
        if not comments and not docs:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            csv_path = os.path.join(OUTPUT_DIR, f"analyzed_{issue_id}.csv")
            pd.DataFrame(columns=[
                "Datetime", "Hours_Since_Start", "Company_Action_Type",
                "Influencer_Impact", "Raw_Total_Mentions",
                "Negative_Ratio", "Mockery_Index", "Advocate_Ratio",
                "Negative_Momentum",
                "Doc_Hostile_Ratio", "Doc_Supportive_Ratio", "Narrative_Pressure",
                "Actual_NVI"
            ]).to_csv(csv_path, index=False)
            return csv_path
        
        # 분석 결과를 comment_id로 인덱싱
        analysis_map = {}
        for a in analysis_results:
            analysis_map[a["comment_id"]] = a
        
        # 댓글 DataFrame 생성
        df = pd.DataFrame(comments)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["hour_bucket"] = df["created_at"].dt.floor("h")
        
        # 분석 결과 병합
        df["sentiment"] = df["comment_id"].map(
            lambda cid: analysis_map.get(cid, {}).get("sentiment")
        )
        df["is_mockery"] = df["comment_id"].map(
            lambda cid: analysis_map.get(cid, {}).get("is_mockery", False)
        )
        df["is_advocate"] = df["comment_id"].map(
            lambda cid: analysis_map.get(cid, {}).get("is_advocate", False)
        )
        
        # 인플루언서 임팩트 (해당 시간에 메가 인플루언서 doc가 있는지)
        influencer_hours = set()
        for d_id, score in doc_scores.items():
            if score >= 1:
                # 해당 doc의 댓글이 속한 시간대
                doc_comments = df[df["doc_id"] == d_id]
                if not doc_comments.empty:
                    influencer_hours.update(doc_comments["hour_bucket"].unique())
        
        # 시간별 감성 분석 결과 취합
        analyzed = df[df["sentiment"].notna()]
        
        hourly_total = df.groupby("hour_bucket").agg(
            total_mentions=("comment_id", "count"),
        ).reset_index()
        
        if not analyzed.empty:
            hourly_analyzed = analyzed.groupby("hour_bucket").agg(
                analyzed_count=("comment_id", "count"),
                negative_count=("sentiment", lambda x: (x == "negative").sum()),
                mockery_count=("is_mockery", "sum"),
                advocate_count=("is_advocate", "sum"),
            ).reset_index()
        else:
            hourly_analyzed = pd.DataFrame(columns=[
                "hour_bucket", "analyzed_count", "negative_count",
                "mockery_count", "advocate_count"
            ])
        
        hourly = hourly_total.merge(hourly_analyzed, on="hour_bucket", how="left")
        hourly = hourly.fillna(0)
        hourly = hourly.sort_values("hour_bucket").reset_index(drop=True)
        
        # Feature Engineering
        hourly["Datetime"] = hourly["hour_bucket"]
        start = hourly["Datetime"].min()
        hourly["Hours_Since_Start"] = (
            (hourly["Datetime"] - start).dt.total_seconds() / 3600
        )
        
        hourly["Raw_Total_Mentions"] = hourly["total_mentions"].astype(float)
        
        analyzed_cnt = hourly["analyzed_count"].astype(float).replace(0, np.nan)
        hourly["Negative_Ratio"] = (
            hourly["negative_count"].astype(float) / analyzed_cnt
        ).fillna(0).round(3)
        hourly["Mockery_Index"] = (
            hourly["mockery_count"].astype(float) / analyzed_cnt
        ).fillna(0).round(3)
        hourly["Advocate_Ratio"] = (
            hourly["advocate_count"].astype(float) / analyzed_cnt
        ).fillna(0).round(3)
        
        hourly["Negative_Momentum"] = (
            hourly["negative_count"].astype(float).diff().fillna(0)
        )
        
        hourly["Influencer_Impact"] = hourly["hour_bucket"].apply(
            lambda h: 1 if h in influencer_hours else 0
        )
        
        hourly["Company_Action_Type"] = 0

        # --- 원문 톤 집계 ---
        doc_tone_map = {}
        for da in doc_analysis_results:
            doc_tone_map[da["doc_id"]] = da
        
        if docs:
            doc_df = pd.DataFrame(docs)
            doc_df["published_at"] = pd.to_datetime(
                doc_df["published_at"], utc=True, errors="coerce"
            )
            doc_df["hour_bucket"] = doc_df["published_at"].dt.floor("h")
            
            # 톤 매핑
            doc_df["tone"] = doc_df["doc_id"].map(
                lambda did: doc_tone_map.get(did, {}).get("tone")
            )
            
            tone_analyzed = doc_df[doc_df["tone"].notna()]
            
            if not tone_analyzed.empty:
                hourly_doc = tone_analyzed.groupby("hour_bucket").agg(
                    doc_total=("doc_id", "count"),
                    hostile_count=("tone", lambda x: ((x == "hostile") | (x == "critical")).sum()),
                    supportive_count=("tone", lambda x: ((x == "sympathetic") | (x == "supportive")).sum()),
                    hostile_only=("tone", lambda x: (x == "hostile").sum()),
                    critical_only=("tone", lambda x: (x == "critical").sum()),
                    sympathetic_only=("tone", lambda x: (x == "sympathetic").sum()),
                    supportive_only=("tone", lambda x: (x == "supportive").sum()),
                ).reset_index()
                
                hourly = hourly.merge(hourly_doc, on="hour_bucket", how="left")
            else:
                for col in ["doc_total", "hostile_count", "supportive_count",
                            "hostile_only", "critical_only", "sympathetic_only", "supportive_only"]:
                    hourly[col] = 0
        else:
            for col in ["doc_total", "hostile_count", "supportive_count",
                        "hostile_only", "critical_only", "sympathetic_only", "supportive_only"]:
                hourly[col] = 0
        
        hourly = hourly.fillna(0)
        doc_cnt = hourly["doc_total"].astype(float).replace(0, np.nan)
        hourly["Doc_Hostile_Ratio"] = (
            hourly["hostile_count"].astype(float) / doc_cnt
        ).fillna(0).round(3)
        hourly["Doc_Supportive_Ratio"] = (
            hourly["supportive_count"].astype(float) / doc_cnt
        ).fillna(0).round(3)
        
        # Narrative Pressure = hostile×0.12 + critical×0.05 - sympathetic×0.08 - supportive×0.10
        hostile_r = (hourly["hostile_only"].astype(float) / doc_cnt).fillna(0)
        critical_r = (hourly["critical_only"].astype(float) / doc_cnt).fillna(0)
        sympathetic_r = (hourly["sympathetic_only"].astype(float) / doc_cnt).fillna(0)
        supportive_r = (hourly["supportive_only"].astype(float) / doc_cnt).fillna(0)
        hourly["Narrative_Pressure"] = (
            hostile_r * 0.12 + critical_r * 0.05
            - sympathetic_r * 0.08 - supportive_r * 0.10
        ).round(4)

        hourly["Actual_NVI"] = self._compute_nvi(hourly)
        
        output_cols = [
            "Datetime", "Hours_Since_Start", "Company_Action_Type",
            "Influencer_Impact", "Raw_Total_Mentions",
            "Negative_Ratio", "Mockery_Index", "Advocate_Ratio",
            "Negative_Momentum",
            "Doc_Hostile_Ratio", "Doc_Supportive_Ratio", "Narrative_Pressure",
            "Actual_NVI"
        ]
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, f"analyzed_{issue_id}.csv")
        hourly[output_cols].to_csv(csv_path, index=False)
        return csv_path
    
    def _compute_nvi(self, df: pd.DataFrame) -> pd.Series:
        """감성 지표 + 원문 톤 기반 NVI(Net Valence Index) 산출."""
        nvi = 0.5  # 초기 평형점
        nvi_values = []
        
        for _, row in df.iterrows():
            neg = float(row.get("Negative_Ratio", 0))
            mock = float(row.get("Mockery_Index", 0))
            adv = float(row.get("Advocate_Ratio", 0))
            doc_hostile = float(row.get("Doc_Hostile_Ratio", 0))
            doc_supportive = float(row.get("Doc_Supportive_Ratio", 0))
            
            # 시간당 변화량 — 댓글 감성
            penalty = neg * 0.3 + mock * 0.2
            bonus = adv * 0.25
            
            # 시간당 변화량 — 원문 톤 (Narrative Pressure)
            narrative_penalty = doc_hostile * 0.04
            narrative_bonus = doc_supportive * 0.03
            
            reversion = (0.5 - nvi) * 0.01  # 평형 회귀
            
            nvi = nvi - penalty + bonus - narrative_penalty + narrative_bonus + reversion
            nvi = max(0.1, min(1.0, nvi))
            nvi_values.append(round(nvi, 3))
        
        return pd.Series(nvi_values)

    # ==========================================
    # Step 4. 여론 지형도 (Sentiment Landscape)
    # ==========================================
    async def _build_sentiment_landscape(
        self,
        comments: list[dict],
        analysis_results: list[dict],
    ) -> dict:
        """감성 분석 결과를 집계하여 여론 지형도 생성.
        
        출력:
          overview: 전체 감성 비율
          top_negative_themes: 부정 여론 핵심 주제 (LLM 클러스터링)
          top_mockery_themes: 조롱/밈 주제
        """
        # 분석 결과를 집계
        total = len(analysis_results) or 1
        negative_count = sum(1 for a in analysis_results if a.get("sentiment") == "negative")
        positive_count = sum(1 for a in analysis_results if a.get("sentiment") == "positive")
        neutral_count = sum(1 for a in analysis_results if a.get("sentiment") == "neutral")
        mockery_count = sum(1 for a in analysis_results if a.get("is_mockery"))
        advocate_count = sum(1 for a in analysis_results if a.get("is_advocate"))
        
        overview = {
            "total_analyzed": total,
            "negative_ratio": round(negative_count / total, 3),
            "positive_ratio": round(positive_count / total, 3),
            "neutral_ratio": round(neutral_count / total, 3),
            "mockery_ratio": round(mockery_count / total, 3),
            "advocate_ratio": round(advocate_count / total, 3),
        }
        
        # 댓글 원문을 comment_id로 인덱싱
        comment_map = {c["comment_id"]: c for c in comments}
        
        # 부정 댓글 (좋아요 순)
        neg_comment_dicts = []
        for a in analysis_results:
            if a.get("sentiment") == "negative":
                c = comment_map.get(a["comment_id"], {})
                neg_comment_dicts.append({
                    "body": c.get("body", ""),
                    "like_count": c.get("like_count", 0),
                })
        neg_comment_dicts.sort(key=lambda x: x["like_count"], reverse=True)
        
        # 조롱 댓글 (좋아요 순)
        mock_comment_dicts = []
        for a in analysis_results:
            if a.get("is_mockery"):
                c = comment_map.get(a["comment_id"], {})
                mock_comment_dicts.append({
                    "body": c.get("body", ""),
                    "like_count": c.get("like_count", 0),
                })
        mock_comment_dicts.sort(key=lambda x: x["like_count"], reverse=True)
        
        # LLM으로 부정 주제 클러스터링
        top_negative_themes = await self._cluster_themes(
            neg_comment_dicts[:50], "부정 여론"
        )
        top_mockery_themes = await self._cluster_themes(
            mock_comment_dicts[:20], "조롱/밈"
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

[댓글 목록]
{comments_text}"""
        
        try:
            res = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.deep_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ThemeClusterList,
                    temperature=0.2,
                )
            )
            parsed = json.loads(res.text)
            return parsed.get("clusters", [])
        except Exception as e:
            print(f"   [WARN] 테마 클러스터링 실패: {e}")
            return []

    # ==========================================
    # Step 5. 감성 타임라인 (Sentiment Timeline)
    # ==========================================
    def _build_sentiment_timeline(
        self,
        comments: list[dict],
        analysis_results: list[dict],
        docs: list[dict],
        doc_scores: dict[int, int],
    ) -> dict:
        """시간별 감성 구성 + 이벤트 마커를 생성.
        
        출력:
          history: [각 시간 버킷의 감성 분포]
          events: [주요 이벤트 마커 (KOL 저격, 기업 대응 등)]
        """
        if not comments or not analysis_results:
            return {"history": [], "events": []}
        
        # 분석 결과 인덱싱
        analysis_map = {a["comment_id"]: a for a in analysis_results}
        
        # 댓글 DataFrame
        df = pd.DataFrame(comments)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["hour_bucket"] = df["created_at"].dt.floor("h")
        
        # 감성 병합
        df["sentiment"] = df["comment_id"].map(
            lambda cid: analysis_map.get(cid, {}).get("sentiment")
        )
        df["is_mockery"] = df["comment_id"].map(
            lambda cid: analysis_map.get(cid, {}).get("is_mockery", False)
        )
        df["is_advocate"] = df["comment_id"].map(
            lambda cid: analysis_map.get(cid, {}).get("is_advocate", False)
        )
        
        analyzed = df[df["sentiment"].notna()]
        
        if analyzed.empty:
            return {"history": [], "events": []}
        
        # 시간별 감성 구성
        hourly = analyzed.groupby("hour_bucket").agg(
            total_mentions=("comment_id", "count"),
            negative_count=("sentiment", lambda x: (x == "negative").sum()),
            mockery_count=("is_mockery", "sum"),
            advocate_count=("is_advocate", "sum"),
        ).reset_index().sort_values("hour_bucket")
        
        first_hour = hourly["hour_bucket"].min()
        
        history = []
        for _, row in hourly.iterrows():
            total = row["total_mentions"] or 1
            hour_offset = int(
                (row["hour_bucket"] - first_hour).total_seconds() / 3600
            )
            history.append({
                "hour": hour_offset,
                "timestamp": row["hour_bucket"].isoformat(),
                "total_mentions": int(row["total_mentions"]),
                "negative_ratio": round(row["negative_count"] / total, 3),
                "mockery_ratio": round(row["mockery_count"] / total, 3),
                "advocate_ratio": round(row["advocate_count"] / total, 3),
            })
        
        # 이벤트 마커: 인플루언서 저격 (influence_score >= 2인 docs)
        event_markers = []
        for d in docs:
            score = doc_scores.get(d["doc_id"], 0)
            if score >= 2:
                d_time = pd.Timestamp(d.get("published_at") or d.get("created_at"))
                hour_offset = int(
                    (d_time - first_hour).total_seconds() / 3600
                )
                event_markers.append({
                    "hour": hour_offset,
                    "timestamp": d.get("published_at") or d.get("created_at"),
                    "type": "influencer_hit",
                    "label": (
                        f"{d.get('author_name', '?')} ({d.get('channel', '?')}) "
                        f"- {d.get('title', '')[:40]}"
                    ),
                    "view_count": d.get("view_count", 0),
                    "followers": d.get("author_followers", 0),
                })
        
        return {
            "history": history,
            "events": event_markers,
        }

    # ==========================================
    # Step 6. KOL (Key Opinion Leader) 식별
    # ==========================================
    async def _identify_kols(
        self,
        docs: list[dict],
        doc_scores: dict[int, int],
    ) -> list[dict]:
        """영향력 높은 작성자(KOL)를 식별하고 LLM으로 성향/대응 권고를 생성.
        
        식별 기준:
          - influence_score >= 1 (followers >= 10K 또는 views >= 50K)
          - 조회수 순 정렬, 상위 10명
        
        출력:
          [{"author_name", "platform", "followers", "stance", 
            "influence_score", "key_content", "view_count", 
            "estimated_reach", "risk_assessment"}, ...]
        """
        # influence_score >= 1인 docs 필터링 + 조회수 순 정렬
        kol_docs = [
            d for d in docs
            if doc_scores.get(d["doc_id"], 0) >= 1
        ]
        kol_docs.sort(key=lambda d: d.get("view_count", 0), reverse=True)
        kol_docs = kol_docs[:10]
        
        if not kol_docs:
            return []
        
        # LLM으로 각 KOL의 성향(stance)과 대응 권고(risk_assessment) 생성
        kol_texts = []
        for i, d in enumerate(kol_docs):
            body_snippet = (d.get("body") or d.get("snippet") or "")[:200]
            kol_texts.append(
                f"[{i+1}] author={d.get('author_name', '?')}, "
                f"channel={d.get('channel', '?')}, "
                f"followers={d.get('author_followers', 0):,}, "
                f"views={d.get('view_count', 0):,}, "
                f"title=\"{d.get('title', '')[:80]}\"\n"
                f"    내용: {body_snippet}"
            )
        
        prompt = f"""아래 온라인 콘텐츠 작성자들의 이슈에 대한 성향(stance)과 리스크 평가를 분석하세요.

stance 기준:
- hostile: 기업에 대해 적대적/비판적
- neutral: 중립적/사실 전달
- supportive: 기업 옹호/우호적

[작성자 목록]
{chr(10).join(kol_texts)}"""
        
        try:
            res = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.deep_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=KOLStanceList,
                    temperature=0.2,
                )
            )
            parsed = json.loads(res.text)
            stance_results = parsed.get("results", [])
        except Exception as e:
            print(f"   [WARN] KOL 성향 분석 실패: {e}")
            stance_results = []
        
        # 성향 결과를 KOL 데이터에 병합
        stance_map = {s["index"]: s for s in stance_results if isinstance(s, dict)}
        
        kols = []
        for i, d in enumerate(kol_docs):
            stance_info = stance_map.get(i + 1, {})
            estimated_reach = (
                d.get("view_count", 0) + (d.get("share_count", 0) or 0) * 50
            )
            
            kols.append({
                "author_name": d.get("author_name", "?"),
                "channel": d.get("channel", "?"),
                "followers": d.get("author_followers", 0),
                "view_count": d.get("view_count", 0),
                "influence_score": doc_scores.get(d["doc_id"], 0),
                "key_content": d.get("title", "")[:100],
                "url": d.get("url", ""),
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
    주의: 이 클래스는 DB 의존성이 있음 (오프라인 배치 처리 전용).
    """
    
    def __init__(self, client, model_name: str, db_pool,
                 gcs_bucket: str, project_id: str, location: str = "us-central1"):
        self.client = client
        self.model_name = model_name
        self.pool = db_pool
        self.gcs_bucket = gcs_bucket
        self.project_id = project_id
        self.location = location
    
    async def run_batch(self, issue_id: str, post_ids: list[str] = None, poll_interval: int = 60):
        """
        1. DB에서 미분석 댓글 로드 (Retriever가 확보한 post_ids 기준)
        2. JSONL 생성 → GCS 업로드
        3. Batch Job 제출
        4. Polling으로 완료 대기
        5. 결과 다운로드 → DB 반영
        """
        post_ids = post_ids or []
        
        comments = await self._load_unanalyzed(issue_id, post_ids)
        if not comments:
            print("   [BATCH] No unanalyzed comments")
            return
        
        print(f"   [BATCH] {len(comments)}건 Batch Prediction 시작...")
        
        # 1. JSONL 생성
        lines = self._build_jsonl(comments)
        
        # 2. GCS 업로드 (lazy import — 배치 실행 시에만 로딩)
        from google.cloud import storage
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.gcs_bucket)
        input_path = f"batch_input/{issue_id}/input.jsonl"
        blob = bucket.blob(input_path)
        blob.upload_from_string("\n".join(lines), content_type="application/jsonl")
        
        gcs_input = f"gs://{self.gcs_bucket}/{input_path}"
        gcs_output = f"gs://{self.gcs_bucket}/batch_output/{issue_id}/"
        
        # 3. Batch Job 제출
        from google.cloud import aiplatform
        aiplatform.init(project=self.project_id, location=self.location)
        
        job = aiplatform.BatchPredictionJob.create(
            job_display_name=f"issue-cracker-{issue_id}",
            model_name=f"publishers/google/models/{self.model_name}",
            gcs_source=gcs_input,
            gcs_destination_prefix=gcs_output,
            sync=False,
        )
        
        print(f"   [BATCH] Job 제출 완료: {job.resource_name}")
        
        # 4. Polling
        while job.state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                                 "JOB_STATE_CANCELLED"):
            await asyncio.sleep(poll_interval)
            job = aiplatform.BatchPredictionJob(job.resource_name)
            print(f"   [BATCH] 상태: {job.state}")
        
        if job.state != "JOB_STATE_SUCCEEDED":
            print(f"   [BATCH] ❌ 실패: {job.state}")
            return
        
        # 5. 결과 반영
        await self._apply_batch_results(gcs_output, issue_id)
        print("   [BATCH] ✅ 완료, 결과 DB 반영")
    
    def _build_jsonl(self, comments: list[dict]) -> list[str]:
        """배치 입력 JSONL 생성."""
        batches = [
            comments[i:i + 50]
            for i in range(0, len(comments), 50)
        ]
        
        lines = []
        for batch in batches:
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
    
    async def _load_unanalyzed(self, issue_id: str, doc_ids: list[int]) -> list[dict]:
        """Retriever가 확보한 docs의 댓글 중, 이 issue에서 미분석인 것만 로드."""
        if not doc_ids:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.comment_id, c.content AS body
                FROM deep_insight.collected_doc_comment c
                LEFT JOIN deep_insight.analysis_results ar
                    ON ar.target_type = 'comment'
                    AND ar.target_id = c.comment_id
                    AND ar.issue_id = $1
                    AND ar.model_version = $3
                WHERE c.doc_id = ANY($2)
                  AND ar.id IS NULL
                ORDER BY c.published_at
            """, issue_id, doc_ids, self.model_name)
        return [dict(r) for r in rows]
    
    async def _apply_batch_results(self, gcs_output: str, issue_id: str):
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
                    INSERT INTO deep_insight.analysis_results
                        (issue_id, target_type, target_id,
                         sentiment, sentiment_score, is_mockery, is_advocate,
                         model_version)
                    VALUES ($1, 'comment', $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (issue_id, target_type, target_id, model_version) DO NOTHING
                """, [
                    (issue_id, int(a["comment_id"]), a["sentiment"],
                     a["sentiment_score"], a["is_mockery"], a["is_advocate"],
                     self.model_name)
                    for a in all_analyses
                ])
