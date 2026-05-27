import json
from google import genai
from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 모듈화된 에이전트 파트 파이프라인 임포트
from state import PipelineState
from agents.analyzer import AnalyzerAgent
from agents.forecaster import ForecasterAgent
from agents.reporter.planner import PlannerAgent
from agents.reporter.analyst import AnalystAgent
from agents.reporter.strategist import StrategistAgent
from agents.reporter.compiler import CompilerAgent
from agents.reviewer import ReviewerAgent

MODEL_NAME = "gemini-2.5-pro"

# 인프라 컴포넌트 초기화 함수
def init_infrastructure():
    client = genai.Client(vertexai=True, project="deep-insight-496705", location="global")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    dummy_docs = [
        Document(page_content="[과거사례] 2024년 배터리 이슈 당시, 2차 입장문(적극적 사과) 발표 후 24시간 내 NVI 반등 시작. 초기 무대응 시 NVI 0.4 이하로 추락.", metadata={"type": "history"}),
        Document(page_content="[금칙어] 절대 '오해', '유감'이라는 단어를 쓰지 말 것. 책임을 전가하는 뉘앙스 금지.", metadata={"type": "guideline"})
    ]
    vector_db = Chroma.from_documents(dummy_docs, embeddings)
    return client, vector_db

# ==========================================
# 🔄 고도화된 순차적 파이프라인 워크플로우 빌더
# 무대응(Baseline) → 전략 수립 → 전략 적용(Mitigated) → Gap 분석
# ==========================================
def build_graph(client, vector_db, db_pool=None):
    # 1. 에이전트 인스턴스 초기화
    analyzer = AnalyzerAgent(client, MODEL_NAME, db_pool) if db_pool else None
    baseline_forecaster = ForecasterAgent(mode="baseline")
    planner = PlannerAgent(client, MODEL_NAME)
    strategist = StrategistAgent(client, MODEL_NAME, vector_db)
    mitigated_forecaster = ForecasterAgent(mode="mitigated")
    analyst = AnalystAgent(client, MODEL_NAME)
    compiler = CompilerAgent(client, MODEL_NAME)
    reviewer = ReviewerAgent(client, MODEL_NAME, vector_db)

    workflow = StateGraph(PipelineState)

    # 2. 핸들러 노드 바인딩
    if analyzer:
        workflow.add_node("analyzer", analyzer.run)
    workflow.add_node("baseline_forecaster", baseline_forecaster.run)
    workflow.add_node("planner", planner.run)
    workflow.add_node("strategist", strategist.run)
    workflow.add_node("mitigated_forecaster", mitigated_forecaster.run)
    workflow.add_node("analyst", analyst.run)
    workflow.add_node("compiler", compiler.run)
    workflow.add_node("reviewer", reviewer.run)

    # 3. 순차 인과관계 엣지 연결
    if analyzer:
        workflow.add_edge(START, "analyzer")                      # 원본 분석
        workflow.add_edge("analyzer", "baseline_forecaster")      # 예측
    else:
        workflow.add_edge(START, "baseline_forecaster")            # DB 없으면 바로 예측
    workflow.add_edge("baseline_forecaster", "planner")   # 기획 지시
    workflow.add_edge("planner", "strategist")            # 전략 수립 + 타임라인 도출
    workflow.add_edge("strategist", "mitigated_forecaster")  # 전략 적용 재예측
    workflow.add_edge("mitigated_forecaster", "analyst")  # Gap 비교 분석
    workflow.add_edge("analyst", "compiler")              # 보고서 취합
    workflow.add_edge("compiler", "reviewer")             # CCO 검토

    # 4. 가이드라인 미통과 시 재기획 단계(planner)로 피드백 조건부 라우팅
    def should_continue(state: PipelineState):
        if state.get("is_approved") or state.get("loop_count", 0) >= 3:
            return END
        return "planner"

    workflow.add_conditional_edges("reviewer", should_continue)

    return workflow.compile()

if __name__ == "__main__":
    client, vector_db = init_infrastructure()
    app = build_graph(client, vector_db)
    print(app.get_graph().print_ascii())