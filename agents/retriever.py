"""
RetrieverAgent — pgVector 기반 관련 데이터 검색

QueryBuilderAgent가 생성한 임베딩 벡터를 사용하여:
1. posts 테이블에서 코사인 유사도 검색
2. comments 테이블에서 코사인 유사도 검색
3. 검색된 데이터를 issue의 분석 대상으로 확정
4. 시계열 피처 CSV 생성 → ForecasterAgent 입력

의존: asyncpg + pgvector
"""
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class RetrieverAgent:
    """pgVector 기반 관련 posts/comments 검색 에이전트.

    QueryBuilderAgent의 search_embeddings를 받아
    DB에서 유사도 검색 → 분석 대상 데이터 확보.
    """

    SIMILARITY_THRESHOLD = 0.5  # 최소 유사도 (코사인, 0~1)

    def __init__(self, db_pool):
        self.pool = db_pool   # asyncpg.Pool

    async def run(self, state: dict) -> dict:
        embeddings = state["search_embeddings"]
        queries = state["search_queries"]
        issue_id = state.get("issue_id", "unknown")

        print(f"\n>> [Retriever] pgVector 검색 시작 (쿼리 {len(queries)}개)...")

        # 1. Posts 검색
        posts = await self._search_posts(embeddings)
        print(f"   [POSTS] {len(posts)}건 검색됨")

        # 2. 검색된 Posts의 Comments 검색
        post_ids = [p["post_id"] for p in posts]
        comments = await self._search_comments(embeddings, post_ids)
        print(f"   [COMMENTS] {len(comments)}건 검색됨")

        # 3. 시계열 피처 변환 → CSV
        csv_path = self._build_timeseries_csv(posts, comments, issue_id)
        print(f"   [CSV] {csv_path}")

        # 4. 검색된 원문 요약 (crisis_context 보강)
        context_summary = self._build_context_summary(posts)

        return {
            "retrieved_post_ids": post_ids,
            "retrieved_comment_count": len(comments),
            "input_csv_path": csv_path,
            "crisis_context": state["crisis_context"] + "\n\n[검색된 주요 게시글]\n" + context_summary,
        }

    # ==========================================
    # Step 1. Posts pgVector 검색
    # ==========================================
    async def _search_posts(self, embeddings: list[list[float]]) -> list[dict]:
        """여러 쿼리 벡터로 posts를 검색, 유사도 임계값 이상 전부 반환."""
        all_posts = {}

        async with self.pool.acquire() as conn:
            for emb in embeddings:
                vec_str = "[" + ",".join(str(v) for v in emb) + "]"
                rows = await conn.fetch("""
                    SELECT
                        post_id, platform, content_type, title, body,
                        author_name, author_followers, view_count,
                        like_count, comment_count, created_at,
                        1 - (embedding <=> $1::vector) AS similarity
                    FROM issue_cracker.posts
                    WHERE embedding IS NOT NULL
                      AND 1 - (embedding <=> $1::vector) >= $2
                    ORDER BY embedding <=> $1::vector
                """, vec_str, self.SIMILARITY_THRESHOLD)

                for r in rows:
                    pid = r["post_id"]
                    sim = float(r["similarity"])
                    if pid not in all_posts or sim > all_posts[pid]["similarity"]:
                        all_posts[pid] = dict(r)

        # 유사도 순 정렬
        return sorted(all_posts.values(), key=lambda x: x["similarity"], reverse=True)

    # ==========================================
    # Step 2. Comments pgVector 검색
    # ==========================================
    async def _search_comments(
        self, embeddings: list[list[float]], post_ids: list[str]
    ) -> list[dict]:
        """검색된 posts의 댓글 + 벡터 유사도 검색 결과 합산 (제한 없음)."""
        all_comments = {}

        async with self.pool.acquire() as conn:
            # 2-1. 검색된 posts에 달린 댓글 전체
            if post_ids:
                rows = await conn.fetch("""
                    SELECT comment_id, post_id, body, author_id,
                           like_count, reply_count, created_at
                    FROM issue_cracker.comments
                    WHERE post_id = ANY($1)
                    ORDER BY created_at
                """, post_ids)
                for r in rows:
                    all_comments[r["comment_id"]] = dict(r)

            # 2-2. 임베딩 벡터 유사도 검색 (posts 외 독립 댓글도 포착)
            for emb in embeddings:
                vec_str = "[" + ",".join(str(v) for v in emb) + "]"
                rows = await conn.fetch("""
                    SELECT comment_id, post_id, body, author_id,
                           like_count, reply_count, created_at,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM issue_cracker.comments
                    WHERE embedding IS NOT NULL
                      AND 1 - (embedding <=> $1::vector) >= $2
                    ORDER BY embedding <=> $1::vector
                """, vec_str, self.SIMILARITY_THRESHOLD)

                for r in rows:
                    cid = r["comment_id"]
                    if cid not in all_comments:
                        all_comments[cid] = dict(r)

        return list(all_comments.values())

    # ==========================================
    # Step 3. 시계열 피처 CSV 생성
    # ==========================================
    def _build_timeseries_csv(
        self, posts: list[dict], comments: list[dict], issue_id: str
    ) -> str:
        """검색된 comments를 시간당 집계 → ForecasterAgent 입력 형식 CSV."""
        if not comments:
            # 댓글 없으면 빈 CSV
            os.makedirs("data", exist_ok=True)
            csv_path = f"data/retrieved_{issue_id}.csv"
            pd.DataFrame(columns=[
                "Datetime", "Hours_Since_Start", "Company_Action_Type",
                "Influencer_Impact", "Raw_Total_Mentions",
                "Negative_Ratio", "Mockery_Index", "Advocate_Ratio",
                "Negative_Momentum", "Actual_NVI"
            ]).to_csv(csv_path, index=False)
            return csv_path

        # 시간당 집계
        df = pd.DataFrame(comments)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["hour_bucket"] = df["created_at"].dt.floor("h")

        hourly = df.groupby("hour_bucket").agg(
            total_mentions=("comment_id", "count"),
            total_likes=("like_count", "sum"),
        ).reset_index()

        hourly = hourly.sort_values("hour_bucket").reset_index(drop=True)

        # 시간 피처
        hourly["Datetime"] = hourly["hour_bucket"]
        start = hourly["Datetime"].min()
        hourly["Hours_Since_Start"] = (
            (hourly["Datetime"] - start).dt.total_seconds() / 3600
        )

        # 인플루언서 임팩트 (posts 기반)
        influencer_hours = set()
        for p in posts:
            if p.get("author_followers", 0) >= 10000 or p.get("view_count", 0) >= 50000:
                h = pd.Timestamp(p["created_at"]).floor("h")
                influencer_hours.add(h)

        hourly["Influencer_Impact"] = hourly["hour_bucket"].apply(
            lambda h: 1 if h in influencer_hours else 0
        )

        # 감성 지표 placeholder (AnalyzerAgent가 채울 영역)
        hourly["Raw_Total_Mentions"] = hourly["total_mentions"].astype(float)
        hourly["Negative_Ratio"] = 0.5     # placeholder
        hourly["Mockery_Index"] = 0.1       # placeholder
        hourly["Advocate_Ratio"] = 0.05     # placeholder
        hourly["Negative_Momentum"] = 0.0
        hourly["Company_Action_Type"] = 0
        hourly["Actual_NVI"] = 0.5          # placeholder

        output_cols = [
            "Datetime", "Hours_Since_Start", "Company_Action_Type",
            "Influencer_Impact", "Raw_Total_Mentions",
            "Negative_Ratio", "Mockery_Index", "Advocate_Ratio",
            "Negative_Momentum", "Actual_NVI",
        ]

        os.makedirs("data", exist_ok=True)
        csv_path = f"data/retrieved_{issue_id}.csv"
        hourly[output_cols].to_csv(csv_path, index=False)
        return csv_path

    # ==========================================
    # Step 4. 검색 결과 요약
    # ==========================================
    def _build_context_summary(self, posts: list[dict], max_posts: int = 5) -> str:
        """검색된 상위 posts를 텍스트 요약으로 변환."""
        lines = []
        for i, p in enumerate(posts[:max_posts]):
            sim = p.get("similarity", 0)
            lines.append(
                f"{i+1}. [{p.get('platform', '?')}] {p.get('title', '제목없음')} "
                f"(조회 {p.get('view_count', 0):,} / 유사도 {sim:.3f})"
            )
        return "\n".join(lines) if lines else "(검색 결과 없음)"
