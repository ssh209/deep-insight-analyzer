from typing import List, TypedDict
from pydantic import BaseModel, Field

# ==========================================
# 📊 [Agent 1 -> Agent 2] 타임라인 시나리오 스키마
# ==========================================
class CrisisEvent(BaseModel):
    hour_offset: int = Field(description="이벤트 발생 예상 시점 (현재 시간 기준 +N 시간 뒤)")
    action_type: int = Field(description="해당 시점의 기업 대응 상태 (0: 무대응, 1: 1차 입장문, 2: 2차 사과문)")
    influencer_hit: int = Field(description="인플루언서 타격 여부 (0: 없음, 1: 발생)")

class CrisisTimeline(BaseModel):
    events: List[CrisisEvent] = Field(description="향후 계획된 이벤트 타임라인 배열")
    prediction_hours: int = Field(description="시뮬레이션 필요 시간 (기본 72시간)")

# ==========================================
# 📝 [Agent 3-1] 최종 아웃풋 데이터 서빙용 구조화 JSON 스키마
# ==========================================
class ActionItem(BaseModel):
    timeframe: str = Field(description="실행 시점 (예: 즉시, 12시간 내, 24시간 내)")
    action: str = Field(description="구체적인 실행 계획 및 내용")

class CrisisReport(BaseModel):
    executive_summary: str = Field(description="현재 상황 및 향후 NVI 트렌드에 대한 핵심 요약 (2줄 이내)")
    alert_level: str = Field(description="위험 등급 (RED, ORANGE, YELLOW 중 택일)")
    expected_nvi_bottom: float = Field(description="향후 72시간 내 예상되는 NVI 최저점")
    legal_and_pr_risk: str = Field(description="예상되는 대중의 반발 또는 법적 리스크")
    immediate_action_items: List[ActionItem] = Field(description="시간대별 구체적인 대응 액션 플랜")

# ==========================================
# 🕵️ [Agent 3-2] CCO 검토 결과 스키마
# ==========================================
class ReviewResult(BaseModel):
    is_approved: bool = Field(description="가이드라인을 완벽히 준수했으면 true, 아니면 false")
    feedback: str = Field(description="반려 시 구체적인 지시사항, 승인 시 '없음'")

# ==========================================
# 🏗️ LangGraph 공통 컨텍스트 데이터 버스
# ==========================================
class PipelineState(TypedDict):
    input_csv_path: str
    crisis_context: str
    timeline_events: list
    nvi_forecast: list
    planner_instructions: str   # 플래너의 작업 지시서
    analyst_draft: str          # 데이터 분석가의 분석 결과
    strategist_draft: str       # 전략가의 액션 플랜 결과
    draft_report: str           # 최종 취합된 JSON 리포트
    review_feedback: str
    is_approved: bool
    loop_count: int