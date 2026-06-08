"""
RetrieverAgent — pgVector 기반 관련 Posts 검색 + 댓글 로드

QueryBuilderAgent가 생성한 임베딩 벡터를 사용하여:
1. posts 테이블에서 코사인 유사도 검색
2. 검색된 posts의 댓글을 전부 로드 (메모리)
3. 검색된 원문 요약 (crisis_context 보강)

의존: asyncpg + pgvector
설계 원칙: DB SELECT는 이 에이전트에서만 수행.
           다른 에이전트는 state에 담긴 메모리 데이터를 소비.
           DataFrame 가공 및 CSV 생성은 Analyzer가 담당.
"""


class RetrieverAgent:
    """pgVector 기반 관련 posts 검색 + 댓글 로드 에이전트.

    QueryBuilderAgent의 search_embeddings를 받아
    DB에서 유사도 검색 → 분석 대상 데이터를 메모리에 적재.
    
    원칙: DB SELECT는 이 에이전트에서만. 결과를 state에 담아
          후속 에이전트(Analyzer 등)가 메모리에서 소비.
    """

    SIMILARITY_THRESHOLD = 0.5  # 최소 유사도 (코사인, 0~1)

    def __init__(self, db_pool):
        self.pool = db_pool   # asyncpg.Pool

    async def run(self, state: dict) -> dict:
        embeddings = state["search_embeddings"]
        queries = state["search_queries"]
        issue_id = state.get("issue_id", "unknown")

        print(f"\n>> [Retriever] pgVector 검색 시작 (쿼리 {len(queries)}개)...")

        # 1. Posts pgVector 검색
        posts = await self._search_posts(embeddings)
        print(f"   [POSTS] {len(posts)}건 검색됨")

        # 2. 검색된 Posts의 모든 댓글 로드 (벡터 검색 없이 단순 SELECT)
        post_ids = [p["post_id"] for p in posts]
        comments = await self._fetch_comments(post_ids)
        print(f"   [COMMENTS] {len(comments)}건 로드됨")

        # 3. 검색된 원문 요약 (crisis_context 보강)
        context_summary = self._build_context_summary(posts)

        return {
            "retrieved_post_ids": post_ids,
            "retrieved_posts": posts,
            "retrieved_comments": comments,
            "retrieved_comment_count": len(comments),
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
                        author_name, author_id, author_followers, view_count,
                        like_count, comment_count, share_count, url, created_at,
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
                        row_dict = dict(r)
                        # datetime → isoformat 변환 (JSON 직렬화 호환)
                        if row_dict.get("created_at"):
                            row_dict["created_at"] = row_dict["created_at"].isoformat()
                        all_posts[pid] = row_dict

        # 유사도 순 정렬
        return sorted(all_posts.values(), key=lambda x: x["similarity"], reverse=True)

    # ==========================================
    # Step 2. 검색된 Posts의 모든 댓글 로드
    # ==========================================
    async def _fetch_comments(self, post_ids: list[str]) -> list[dict]:
        """검색된 posts에 달린 모든 댓글을 로드. 벡터 검색 없이 단순 JOIN."""
        if not post_ids:
            return []

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT comment_id, post_id, body, author_id,
                       like_count, reply_count, created_at
                FROM issue_cracker.comments
                WHERE post_id = ANY($1)
                ORDER BY created_at
            """, post_ids)

        result = []
        for r in rows:
            row_dict = dict(r)
            if row_dict.get("created_at"):
                row_dict["created_at"] = row_dict["created_at"].isoformat()
            result.append(row_dict)
        return result

    # ==========================================
    # Step 3. 검색 결과 요약
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
