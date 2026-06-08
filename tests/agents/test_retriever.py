import pytest
import pytest_asyncio
import asyncio
import sys
import asyncpg
import os
from unittest.mock import MagicMock

from agents.retriever import RetrieverAgent

# 테스트용 로컬 DB URL
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vibe_x")

@pytest_asyncio.fixture
async def real_db_pool():
    """실제 PostgreSQL DB 커넥션 풀을 반환하는 Fixture (asyncpg)"""
    try:
        # ssl=False 옵션을 주어 윈도우 환경 로컬 연결 시 발생하는 [Errno 42] 에러를 우회합니다.
        pool = await asyncpg.create_pool(DATABASE_URL, ssl=False)
        yield pool
        await pool.close()
    except Exception as e:
        pytest.skip(f"DB 연결 실패로 Retriever 테스트를 건너뜁니다: {e}")

@pytest.mark.asyncio
async def test_retriever_agent_real_run(real_db_pool):
    # 1. Agent 초기화 (실제 DB Pool 주입)
    agent = RetrieverAgent(db_pool=real_db_pool)
    
    # 2. 실제 임베딩 모델 로드 (HuggingFace)
    print("\n[Real Test] 임베딩 모델 로드 중...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    # 3. 실제 쿼리를 임베딩하여 벡터 생성 (앞서 DB에 넣은 내용과 연관된 키워드)
    query_text = "스타벅스 탱크데이 논란 불매"
    real_vector = embeddings_model.embed_query(f"query: {query_text}")
    
    # 유사도 임계값을 정상 범위로 복구
    agent.SIMILARITY_THRESHOLD = 0.5
    
    initial_state = {
        "crisis_context": "스타벅스 탱크데이 이벤트가 5/18 민주화 운동을 비하했다는 논란을 일으키고 있어. 대응 방안을 마련해야 해.",
        "search_queries": [query_text],
        "search_embeddings": [real_vector],
        "issue_id": "test_issue_123"
    }
    
    # 3. Agent 실행
    print("\n[Real Test] RetrieverAgent DB 검색 시작...")
    result = await agent.run(initial_state)
    
    # 결과 로깅
    print("\n" + "="*50)
    print("🔥 [Real Test] RetrieverAgent 실행 결과 🔥")
    print("="*50)
    print(f"✅ 검색된 게시글 수: {len(result.get('retrieved_posts', []))} 개")
    print(f"✅ 로드된 댓글 수  : {result.get('retrieved_comment_count', 0)} 개")
    print("-" * 50)
    print("📝 [생성된 crisis_context 요약문]")
    print(result.get("crisis_context"))
    print("=" * 50)
    
    # 4. 결과 검증
    assert "retrieved_posts" in result
    assert "retrieved_comments" in result
    assert "retrieved_comment_count" in result
    assert "crisis_context" in result
