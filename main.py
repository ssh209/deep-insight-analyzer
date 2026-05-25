import json
from google import genai
from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 모듈화된 파트들 임포트
from state import PipelineState
from agents.analyzer import AnalyzerAgent
from agents.forecaster import ForecasterAgent
from agents.reporter.planner import PlannerAgent
from agents.reporter.analyst import AnalystAgent
from agents.reporter.strategist import StrategistAgent
from agents.reporter.compiler import CompilerAgent
from agents.reviewer import ReviewerAgent

# 1. 코어 클라이언트 인프라 셋업
client = genai.Client(vertexai=True, project="deep-insight-496705", location="global")
MODEL_NAME = "gemini-2.5-pro"

# 2. RAG 벡터 스토어 컴포넌트 빌드
print("⏳ 인프라 레이어: 지식베이스 벡터 DB 인스턴스 초기화 중...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
dummy_docs = [
    Document(page_content="[과거사례] 2024년 배터리 이슈 당시, 2차 입장문(적극적 사과) 발표 후 24시간 내 NVI 반등 시작. 초기 무대응 시 NVI 0.4 이하로 추락.", metadata={"type": "history"}),
    Document(page_content="[금칙어] 절대 '오해', '유감'이라는 단어를 쓰지 말 것. 책임을 전가하는 뉘앙스 금지.", metadata={"type": "guideline"})
]
vector_db = Chroma.from_documents(dummy_docs, embeddings)

# 3. 개별 에이전트 모듈 인스턴스 초기화 및 조립
analyzer = AnalyzerAgent(client, MODEL_NAME)
forecaster = ForecasterAgent()
planner = PlannerAgent(client, MODEL_NAME)
analyst = AnalystAgent(client, MODEL_NAME)
strategist = StrategistAgent(client, MODEL_NAME, vector_db)
compiler = CompilerAgent(client, MODEL_NAME)
reviewer = ReviewerAgent(client, MODEL_NAME, vector_db)

# 4. 루프 제어 조건 함수 정의
def should_continue(state: PipelineState):
    if state["is_approved"] or state["loop_count"] >= 3:
        return END
    return "generator"

# 5. LangGraph 토폴로지 구성
workflow = StateGraph(PipelineState)

# 개별 에이전트의 실행 핸들러 함수를 노드로 바인딩
workflow.add_node("analyzer", analyzer.run)
workflow.add_node("forecaster", forecaster.run)
workflow.add_node("planner", planner.run)
workflow.add_node("analyst", analyst.run)
workflow.add_node("strategist", strategist.run)
workflow.add_node("compiler", compiler.run)
workflow.add_node("reviewer", reviewer.run)

workflow.add_edge(START, "analyzer")
workflow.add_edge("analyzer", "forecaster")
workflow.add_edge("forecaster", "planner")
workflow.add_edge("planner", "analyst")
workflow.add_edge("planner", "strategist")
workflow.add_edge("analyst", "compiler")
workflow.add_edge("strategist", "compiler")
workflow.add_edge("compiler", "reviewer")

def should_continue(state: PipelineState):
    if state["is_approved"] or state["loop_count"] >= 3:
        return END
    return "planner"

workflow.add_conditional_edges("reviewer", should_continue)

app = workflow.compile()

# 6. 메인 통합 테스트 엔트리포인트
if __name__ == "__main__":
    target_csv = "data/pr_crisis_dataset.csv"
    
    # 1회차 루프에서 CCO 반려를 확실히 유도하도록 '유감스럽다' 키워드 가상 계획에 주입
    input_metadata = """
    [현재 상황] 초기 대응 지연 및 무대응 기간 유튜버 2연타 타격으로 여론 최악 직면.
    [향후 대응 계획]
    - 4시간 뒤: '사실무근이며 깊은 유감이다'라는 1차 해명문 배포 (action_type: 1)
    - 24시간 뒤: 전면 리콜 공표 및 대표이사 명의의 2차 대고객 사과문 발표 (action_type: 2)
    """
    
    initial_state = {
        "input_csv_path": target_csv,
        "crisis_context": input_metadata,
        "timeline_events": [], 
        "nvi_forecast": [], 
        "draft_report": "", 
        "review_feedback": "", 
        "is_approved": False, 
        "loop_count": 0
    }
    
    print("\n" + "="*70)
    print("🚀 [Modularized Engine] 워크플로우 추론 가동")
    print("="*70)
    
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*70)
    print("🎉 [파이프라인 터미널 아웃풋 - 최종 정제된 대시보드용 JSON 데이터]")
    print("="*70)
    
    try:
        parsed_report = json.loads(final_state["draft_report"])
        print(json.dumps(parsed_report, indent=4, ensure_ascii=False))
    except json.JSONDecodeError:
        print(final_state["draft_report"])