from typing import TypedDict, Optional, List

# ==========================================
# 1. 그래프 상태(State) 정의
# 에이전트들이 서로 주고받는 '공유 메모리' 역할입니다.
# ==========================================
class AgentState(TypedDict):
    crisis_context: str            # 초기 입력된 뉴스 기사나 상황 텍스트
    extracted_params: dict         # 1번 에이전트가 뽑아낸 JSON 파라미터 (MOIRAI용)
    nvi_forecast: List[float]      # 2번 에이전트(MOIRAI)가 생성한 여론 예측 궤적
    draft_report: str              # 2-1 에이전트가 작성한 리포트 초안
    review_feedback: Optional[str] # 2-2 에이전트가 반려 시 작성한 피드백 (수정 지시사항)
    is_approved: bool              # 최종 승인 여부
