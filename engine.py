import json
from google import genai
from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 중앙 설정 모듈
from config import GCP_PROJECT_ID, GCP_LOCATION, GEMINI_MODEL, EMBEDDING_MODEL

# 모듈화된 에이전트 파트 파이프라인 임포트
from state import PipelineState
from agents.query_builder import QueryBuilderAgent
from agents.retriever import RetrieverAgent
from agents.analyzer import AnalyzerAgent
from agents.forecaster import ForecasterAgent
from agents.reporter.planner import PlannerAgent
from agents.reporter.analyst import AnalystAgent
from agents.reporter.strategist import StrategistAgent
from agents.reporter.compiler import CompilerAgent
from agents.reviewer import ReviewerAgent

MODEL_NAME = GEMINI_MODEL

# 인프라 컴포넌트 초기화 함수
def init_infrastructure():
    client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    dummy_docs = [
        Document(page_content="[과거사례] 2024년 배터리 이슈 당시, 2차 입장문(적극적 사과) 발표 후 24시간 내 NVI 반등 시작. 초기 무대응 시 NVI 0.4 이하로 추락.", metadata={"type": "history"}),
        Document(page_content="[금칙어] 절대 '오해', '유감'이라는 단어를 쓰지 말 것. 책임을 전가하는 뉘앙스 금지.", metadata={"type": "guideline"})
    ]
    vector_db = Chroma.from_documents(dummy_docs, embeddings)
    return client, vector_db, embeddings

# ==========================================
# 🔄 순차 파이프라인 워크플로우 빌더
#
# DB 모드 (db_pool 있을 때):
#   START → QueryBuilder → Retriever → Analyzer
#         → Baseline Forecaster → Planner → Strategist
#         → Mitigated Forecaster → Analyst → Compiler → Reviewer → END
#
# CSV 모드 (db_pool 없을 때):
#   START → Baseline Forecaster → ... (기존과 동일)
#
# Forecaster는 state["forecaster_model"]에 따라 런타임에 모델 자동 선택.
# ==========================================
def build_graph(client, vector_db, embeddings=None, db_pool=None):
    # 1. 에이전트 인스턴스 초기화
    query_builder = QueryBuilderAgent(client, MODEL_NAME, embeddings) if db_pool and embeddings else None
    retriever = RetrieverAgent(db_pool) if db_pool else None
    analyzer = AnalyzerAgent(client, MODEL_NAME, db_pool) if db_pool else None
    # ForecasterAgent는 루트 컨트롤러 — state["forecaster_model"]로 런타임 라우팅
    baseline_forecaster = ForecasterAgent(mode="baseline")
    mitigated_forecaster = ForecasterAgent(mode="mitigated")
    analyst = AnalystAgent(client, MODEL_NAME)
    planner = PlannerAgent(client, MODEL_NAME)
    strategist = StrategistAgent(client, MODEL_NAME, vector_db)
    compiler = CompilerAgent(client, MODEL_NAME)
    reviewer = ReviewerAgent(client, MODEL_NAME, vector_db)

    workflow = StateGraph(PipelineState)

    # 2. 핸들러 노드 바인딩
    if query_builder:
        workflow.add_node("query_builder", query_builder.run)
    if retriever:
        workflow.add_node("retriever", retriever.run)
    if analyzer:
        workflow.add_node("analyzer", analyzer.run)
    workflow.add_node("baseline_forecaster", baseline_forecaster.run)
    workflow.add_node("planner", planner.run)
    workflow.add_node("strategist", strategist.run)
    workflow.add_node("mitigated_forecaster", mitigated_forecaster.run)
    workflow.add_node("analyst", analyst.run)
    workflow.add_node("compiler", compiler.run)
    workflow.add_node("reviewer", reviewer.run)

    # 3. 순차 엣지 연결
    if query_builder and retriever:
        # DB 모드: QueryBuilder → Retriever → (Analyzer) → Forecaster
        workflow.add_edge(START, "query_builder")
        workflow.add_edge("query_builder", "retriever")
        if analyzer:
            workflow.add_edge("retriever", "analyzer")
            workflow.add_edge("analyzer", "baseline_forecaster")
        else:
            workflow.add_edge("retriever", "baseline_forecaster")
    elif analyzer:
        workflow.add_edge(START, "analyzer")
        workflow.add_edge("analyzer", "baseline_forecaster")
    else:
        # CSV 모드: 바로 예측
        workflow.add_edge(START, "baseline_forecaster")

    workflow.add_edge("baseline_forecaster", "planner")
    workflow.add_edge("planner", "strategist")
    workflow.add_edge("strategist", "mitigated_forecaster")
    workflow.add_edge("mitigated_forecaster", "analyst")
    workflow.add_edge("analyst", "compiler")
    workflow.add_edge("compiler", "reviewer")

    # 4. 가이드라인 미통과 시 재기획 단계(planner)로 피드백 조건부 라우팅
    def should_continue(state: PipelineState):
        if state.get("is_approved") or state.get("loop_count", 0) >= 3:
            return END
        return "planner"

    workflow.add_conditional_edges("reviewer", should_continue)

    return workflow.compile()

if __name__ == "__main__":
    client, vector_db, embeddings = init_infrastructure()
    app = build_graph(client, vector_db, embeddings)
    print(app.get_graph().print_ascii())