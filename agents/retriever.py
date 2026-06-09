"""
RetrieverAgent — pgVector 기반 관련 문서 검색 + 댓글 로드

QueryBuilderAgent가 생성한 임베딩 벡터를 사용하여:
1. collected_doc + collected_doc_embedding에서 코사인 유사도 검색
2. 검색된 문서의 댓글을 전부 로드 (collected_doc_comment)
3. 검색된 원문 요약 (crisis_context 보강)

Collector DB 테이블 참조 (읽기 전용):
  - deep_insight.collected_doc          — 수집 문서 (doc_id BIGINT PK)
  - deep_insight.collected_doc_comment  — 댓글 (comment_id BIGINT PK)
  - deep_insight.collected_doc_embedding — 임베딩 벡터 (별도 테이블, JOIN)

의존: asyncpg + pgvector
설계 원칙: DB SELECT는 이 에이전트에서만 수행.
           다른 에이전트는 state에 담긴 메모리 데이터를 소비.
           DataFrame 가공 및 CSV 생성은 Analyzer가 담당.
"""


class RetrieverAgent:
    """pgVector 기반 관련 문서 검색 + 댓글 로드 에이전트.

    QueryBuilderAgent의 search_embeddings를 받아
    collected_doc_embedding에서 유사도 검색
    → 분석 대상 데이터를 메모리에 적재.

    원칙: DB SELECT는 이 에이전트에서만. 결과를 state에 담아
          후속 에이전트(Analyzer 등)가 메모리에서 소비.
    """

    SIMILARITY_THRESHOLD = 0.5  # 최소 유사도 (코사인, 0~1)

    def __init__(self, db_pool):
        self.pool = db_pool   # asyncpg.Pool

    async def run(self, state: dict) -> dict:
        embeddings = state["search_embeddings"]
        queries = state["search_queries"]

        print(f"\n>> [Retriever] pgVector 검색 시작 (쿼리 {len(queries)}개)...")

        # 1. collected_doc pgVector 검색 (JOIN collected_doc_embedding)
        docs = await self._search_docs(embeddings)
        print(f"   [DOCS] {len(docs)}건 검색됨")

        # 2. 검색된 문서의 모든 댓글 로드 (collected_doc_comment)
        doc_ids = [d["doc_id"] for d in docs]
        comments = await self._fetch_comments(doc_ids)
        print(f"   [COMMENTS] {len(comments)}건 로드됨")

        # 3. 검색된 원문 요약 (crisis_context 보강)
        context_summary = self._build_context_summary(docs)

        return {
            "retrieved_doc_ids": doc_ids,
            "retrieved_docs": docs,
            "retrieved_comments": comments,
            "retrieved_comment_count": len(comments),
            "crisis_context": state["crisis_context"] + "\n\n[검색된 주요 게시글]\n" + context_summary,
        }

    # ==========================================
    # Step 1. collected_doc + collected_doc_embedding pgVector 검색
    # ==========================================
    async def _search_docs(self, embeddings: list[list[float]]) -> list[dict]:
        """여러 쿼리 벡터로 collected_doc를 검색, 유사도 임계값 이상 전부 반환.

        collected_doc_embedding 테이블과 JOIN하여 벡터 검색 수행.
        raw JSONB에서 view_count, like_count 등 메타데이터 추출.
        """
        all_docs = {}

        async with self.pool.acquire() as conn:
            for emb in embeddings:
                vec_str = "[" + ",".join(str(v) for v in emb) + "]"
                rows = await conn.fetch("""
                    SELECT
                        d.doc_id,
                        d.channel,
                        d.source,
                        d.media,
                        d.title,
                        d.body,
                        d.snippet,
                        d.author,
                        d.url,
                        d.published_at,
                        d.impact_score,
                        -- raw JSONB에서 메타데이터 추출
                        COALESCE((d.raw->>'view_count')::int, 0)    AS view_count,
                        COALESCE((d.raw->>'like_count')::int, 0)    AS like_count,
                        COALESCE((d.raw->>'reply_count')::int, 0)   AS comment_count,
                        COALESCE((d.raw->>'retweet_count')::int, 0) AS share_count,
                        COALESCE((d.raw->>'follower_count')::int,
                                 (d.raw->>'member_count')::int, 0)  AS author_followers,
                        d.raw->>'channel_name'                      AS channel_name,
                        d.raw->>'account_name'                      AS account_name,
                        1 - (e.embedding <=> $1::vector) AS similarity
                    FROM deep_insight.collected_doc d
                    JOIN deep_insight.collected_doc_embedding e ON d.doc_id = e.doc_id
                    WHERE e.embedding IS NOT NULL
                      AND e.model = 'multilingual-e5-small'
                      AND e.text_type = 'title_body'
                      AND 1 - (e.embedding <=> $1::vector) >= $2
                    ORDER BY e.embedding <=> $1::vector
                """, vec_str, self.SIMILARITY_THRESHOLD)

                for r in rows:
                    did = r["doc_id"]
                    sim = float(r["similarity"])
                    if did not in all_docs or sim > all_docs[did]["similarity"]:
                        row_dict = dict(r)
                        # datetime → isoformat 변환 (JSON 직렬화 호환)
                        if row_dict.get("published_at"):
                            row_dict["published_at"] = row_dict["published_at"].isoformat()
                        # author_name 필드 통일 (author → author_name)
                        row_dict["author_name"] = (
                            row_dict.pop("channel_name", None)
                            or row_dict.pop("account_name", None)
                            or row_dict.get("author", "")
                        )
                        all_docs[did] = row_dict

        # 유사도 순 정렬
        return sorted(all_docs.values(), key=lambda x: x["similarity"], reverse=True)

    # ==========================================
    # Step 2. 검색된 문서의 모든 댓글 로드 (collected_doc_comment)
    # ==========================================
    async def _fetch_comments(self, doc_ids: list[int]) -> list[dict]:
        """검색된 문서에 달린 모든 댓글을 로드. 벡터 검색 없이 단순 SELECT."""
        if not doc_ids:
            return []

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT comment_id, doc_id, content AS body, author,
                       published_at AS created_at
                FROM deep_insight.collected_doc_comment
                WHERE doc_id = ANY($1)
                ORDER BY published_at
            """, doc_ids)

        result = []
        for r in rows:
            row_dict = dict(r)
            # comment_id를 문자열로 변환 (LLM 분석 호환)
            row_dict["comment_id"] = str(row_dict["comment_id"])
            if row_dict.get("created_at"):
                row_dict["created_at"] = row_dict["created_at"].isoformat()
            # author_name 필드 통일
            row_dict["author_name"] = row_dict.pop("author", "")
            result.append(row_dict)
        return result

    # ==========================================
    # Step 3. 검색 결과 요약
    # ==========================================
    def _build_context_summary(self, docs: list[dict], max_docs: int = 5) -> str:
        """검색된 상위 문서를 텍스트 요약으로 변환."""
        lines = []
        for i, d in enumerate(docs[:max_docs]):
            sim = d.get("similarity", 0)
            lines.append(
                f"{i+1}. [{d.get('channel', '?')}] {d.get('title', '제목없음')} "
                f"(조회 {d.get('view_count', 0):,} / 유사도 {sim:.3f})"
            )
        return "\n".join(lines) if lines else "(검색 결과 없음)"
