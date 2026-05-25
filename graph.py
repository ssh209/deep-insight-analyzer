from langgraph.graph import StateGraph, END
from common.state import AgentState
from agent.analyzer import (
    analyze_context_node,
    forecasting_node
)
from agent.writer import (
    generate_report_node,
    review_report_node,
    check_approval
)

# ==========================================
# 4. LangGraph 파이프라인 조립
# ==========================================
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("Analyzer", analyze_context_node)
workflow.add_node("Forecaster", forecasting_node)
workflow.add_node("Generator", generate_report_node)
workflow.add_node("Reviewer", review_report_node)

# 엣지(흐름) 연결
workflow.set_entry_point("Analyzer")
workflow.add_edge("Analyzer", "Forecaster")
workflow.add_edge("Forecaster", "Generator")
workflow.add_edge("Generator", "Reviewer")

# 핵심: Reviewer의 결과에 따른 조건부 분기 (순환 루프)
workflow.add_conditional_edges(
    "Reviewer",
    check_approval,
    {
        "approved": END,             # 승인되면 그래프 종료
        "rejected": "Generator"      # 반려되면 다시 Generator로 돌아감 (Loop)
    }
)

# 그래프 컴파일 (실행 가능한 객체로 변환)
app = workflow.compile()
