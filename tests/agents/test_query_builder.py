import pytest
import json
import os
import pytest
from google import genai
from langchain_community.embeddings import HuggingFaceEmbeddings

from agents.query_builder import QueryBuilderAgent
from config import GCP_PROJECT_ID, GCP_LOCATION, GEMINI_MODEL, EMBEDDING_MODEL

@pytest.fixture
def real_client():
    # Vertex AI 기반의 실제 Gemini API 클라이언트 초기화
    return genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)

@pytest.fixture
def real_embeddings():
    # 실제 로컬 임베딩 모델 로드 (최초 실행 시 모델 가중치 다운로드가 발생할 수 있음)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def test_query_builder_agent_real_run(real_client, real_embeddings):
    # 1. Agent 초기화 (실제 객체 주입)
    agent = QueryBuilderAgent(
        client=real_client, 
        model_name=GEMINI_MODEL, 
        embeddings=real_embeddings
    )
    
    # 2. 입력 상태(State) 정의
    initial_state = {
        "crisis_context": "스타벅스 탱크데이 이벤트가 5/18 민주화 운동을 비하했다는 논란을 일으키고 있어. 대응 방안을 마련해야 해."
    }
    
    # 3. Agent 실행 (실제 구글 API 및 HuggingFace 연산)
    print(f"\n[Real Test] QueryBuilderAgent 실제 구동 시작 (수 초 소요될 수 있음)...")
    print(f"-> 입력 주제: {initial_state['crisis_context']}")
    result = agent.run(initial_state)
    
    # 결과 로깅
    print("\n" + "="*50)
    print("🔥 [Real Test] QueryBuilderAgent 실행 결과 🔥")
    print("="*50)
    print(f"🔑 추출된 키워드: {result.get('search_keywords')}")
    print(f"⏳ 제안된 시간 범위: {result.get('search_time_hint')}")
    print(f"🔍 생성된 쿼리 수: {len(result.get('search_queries', []))} 개")
    print("-" * 50)
    print("📝 [실제 생성된 검색 쿼리 목록 (DB 벡터 검색용)]")
    for i, q in enumerate(result.get('search_queries', [])):
        print(f"  {i+1}. {q}")
        print(f"     -> (실제 임베딩 모델에 들어간 텍스트: 'query: {q}')")
    print("=" * 50)
    
    # 4. 결과 State 검증
    assert "search_keywords" in result
    assert isinstance(result["search_keywords"], list)
    assert len(result["search_keywords"]) > 0
    
    assert "search_queries" in result
    assert isinstance(result["search_queries"], list)
    assert len(result["search_queries"]) > 0
    
    assert "search_embeddings" in result
    assert isinstance(result["search_embeddings"], list)
    assert len(result["search_embeddings"]) == len(result["search_queries"])
    
    # 임베딩 벡터 차원 검증 (E5 small의 경우 384 차원)
    assert len(result["search_embeddings"][0]) == 384
    
    assert "search_time_hint" in result
