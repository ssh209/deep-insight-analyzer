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

# 오리지널 병렬 워크플로우 맵 빌더
def build_graph(client, vector_db):
    analyzer = AnalyzerAgent(client, MODEL_NAME)
    forecaster = ForecasterAgent()
    planner = PlannerAgent(client, MODEL_NAME)
    analyst = AnalystAgent(client, MODEL_NAME)
    strategist = StrategistAgent(client, MODEL_NAME, vector_db)
    compiler = CompilerAgent(client, MODEL_NAME)
    reviewer = ReviewerAgent(client, MODEL_NAME, vector_db)

    workflow = StateGraph(PipelineState)

    # 핸들러 노드 바인딩
    workflow.add_node("analyzer", analyzer.run)
    workflow.add_node("forecaster", forecaster.run)
    workflow.add_node("planner", planner.run)
    workflow.add_node("analyst", analyst.run)
    workflow.add_node("strategist", strategist.run)
    workflow.add_node("compiler", compiler.run)
    workflow.add_node("reviewer", reviewer.run)

    # 순차 및 병렬(Fan-out/Fan-in) 구조 구현
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "forecaster")
    workflow.add_edge("forecaster", "planner")
    
    # 기획서 생성 후 분석가와 전략가 비동기 병렬 분기
    workflow.add_edge("planner", "analyst")
    workflow.add_edge("planner", "strategist")
    
    # 두 작업 완료 후 취합처로 병합
    workflow.add_edge("analyst", "compiler")
    workflow.add_edge("strategist", "compiler")
    
    workflow.add_edge("compiler", "reviewer")

    # 가이드라인 미통과 시 재기획 단계(planner)로 피드백 조건부 라우팅
    def should_continue(state: PipelineState):
        if state["is_approved"] or state["loop_count"] >= 3:
            return END
        return "planner"

    workflow.add_conditional_edges("reviewer", should_continue)

    return workflow.compile()

if __name__ == "__main__":
    client, vector_db = init_infrastructure()
    app = build_graph(client, vector_db)
    print(app.get_graph().print_ascii())