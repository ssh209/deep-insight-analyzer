import pytest
import asyncio
from langgraph.graph import StateGraph, START, END
import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가하여 모듈 임포트 에러 해결
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from google import genai
from config import GCP_PROJECT_ID, GCP_LOCATION, GEMINI_MODEL, EMBEDDING_MODEL
from db import create_db_pool

from state import PipelineState
from agents.query_builder import QueryBuilderAgent
from agents.retriever import RetrieverAgent
from agents.analyzer import AnalyzerAgent

@pytest.mark.asyncio
async def test_e2e_analyzer_pipeline():
    """
    QueryBuilder -> Retriever -> Analyzer 파이프라인 E2E 테스트.
    실제 DB와 Gemini API를 연동하여 통합 동작을 검증합니다.
    """
    # 1. 인프라 단독 초기화 (engine.py 의존성 제거)
    client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)
    
    # 임베딩 모델 로드 (Langchain HuggingFace 폴백 지원)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    db_pool = await create_db_pool()
    
    if not db_pool:
        pytest.skip("DB 연결이 없어 E2E 테스트를 수행할 수 없습니다.")
        
    try:
        # 2. 에이전트 인스턴스 생성
        query_builder = QueryBuilderAgent(client, GEMINI_MODEL, embeddings)
        retriever = RetrieverAgent(db_pool)
        analyzer = AnalyzerAgent(client, GEMINI_MODEL, db_pool)
        
        # 3. 테스트용 커스텀 그래프 구성 (Analyzer까지만)
        workflow = StateGraph(PipelineState)
        workflow.add_node("query_builder", query_builder.run)
        workflow.add_node("retriever", retriever.run)
        workflow.add_node("analyzer", analyzer.run)
        
        workflow.add_edge(START, "query_builder")
        workflow.add_edge("query_builder", "retriever")
        workflow.add_edge("retriever", "analyzer")
        workflow.add_edge("analyzer", END)
        
        app = workflow.compile()
        
        # 4. 초기 상태 설정
        initial_state = {
            "issue_id": "test_e2e_starbucks_001",
            "crisis_context": "스타벅스 탱크데이 이벤트가 5/18 민주화 운동을 비하했다는 논란을 일으키고 있어. 불매 운동 조짐까지 보이고 있으니 시급히 대응 방안을 마련해야 해.",
            "search_queries": [],
            "search_embeddings": [],
            "retrieved_posts": [],
            "retrieved_comments": [],
            "analysis_result": {},
            "forecaster_model": "lightgbm"
        }
        
        # 5. 그래프 실행
        print("\n[E2E Test] QueryBuilder -> Retriever -> Analyzer 파이프라인 실행 시작...")
        final_state = await app.ainvoke(initial_state)
        
        # 6. 결과 검증
        # QueryBuilder 결과 확인
        assert len(final_state["search_queries"]) > 0, "검색 쿼리가 생성되지 않았습니다."
        assert len(final_state["search_embeddings"]) > 0, "검색 임베딩이 생성되지 않았습니다."
        
        # Retriever 결과 확인 (seed_starbucks_data.py에서 넣은 데이터가 있어야 함)
        assert len(final_state["retrieved_posts"]) > 0, "DB에서 조회된 게시글이 없습니다."
        assert len(final_state["retrieved_comments"]) > 0, "DB에서 조회된 댓글이 없습니다."
        
        # Analyzer 결과 확인
        analysis = final_state.get("analysis_result", {})
        assert analysis, "Analyzer의 분석 결과가 비어있습니다."
        assert "sentiment_distribution" in analysis, "감성 분포(sentiment_distribution) 결과가 없습니다."
        assert "key_themes" in analysis, "핵심 주제(key_themes) 결과가 없습니다."
        
        print("\n[E2E Test] ✅ 파이프라인 정상 종료!")
        print(f"- 생성된 검색어: {final_state['search_queries']}")
        print(f"- 조회된 게시물 수: {len(final_state['retrieved_posts'])}")
        print(f"- 조회된 댓글 수: {len(final_state['retrieved_comments'])}")
        print("- 감성 분석 요약:")
        print(analysis["sentiment_distribution"])
        
    finally:
        # 안전한 종료를 위해 풀 닫기
        await db_pool.close()
