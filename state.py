from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field

# ==========================================
# 📊 [Strategist → Mitigated Forecaster] 대응 타임라인 스키마
# ==========================================
class CrisisEvent(BaseModel):
    hour_offset: int = Field(description="대응 액션 실행 예상 시점 (현재 시간 기준 +N 시간 뒤)")
    action_type: int = Field(description="해당 시점의 기업 대응 상태 (0: 무대응, 1: 1차 입장문, 2: 2차 사과문)")
    influencer_hit: int = Field(description="인플루언서 타격 여부 (0: 없음, 1: 발생)")

class CrisisTimeline(BaseModel):
    events: List[CrisisEvent] = Field(description="향후 계획된 대응 이벤트 타임라인 배열")
    prediction_hours: int = Field(description="시뮬레이션 필요 시간 (기본 72시간)")

# ==========================================
# 📝 [Compiler] 최종 아웃풋 데이터 서빙용 구조화 JSON 스키마
# ==========================================
class ActionItem(BaseModel):
    timeframe: str = Field(description="실행 시점 (예: 즉시, 12시간 내, 24시간 내)")
    action: str = Field(description="구체적인 실행 계획 및 내용")

class RiskItem(BaseModel):
    risk: str = Field(description="리스크명 (예: 소비자 집단 소송, 밈 고착화)")
    probability: str = Field(description="발생 확률 (low / medium / high)")
    impact: str = Field(description="영향도 (low / medium / high / critical)")
    category: str = Field(description="리스크 범주 (legal / reputation / competitive / operational)")
    mitigation: str = Field(description="완화 전략")

class CrisisReport(BaseModel):
    executive_summary: str = Field(description="현재 상황 및 향후 NVI 트렌드에 대한 핵심 요약 (2줄 이내)")
    alert_level: str = Field(description="위험 등급 (RED, ORANGE, YELLOW 중 택일)")
    baseline_nvi_bottom: float = Field(description="무대응 시 향후 72시간 내 예상 NVI 최저점")
    mitigated_nvi_bottom: float = Field(description="전략 적용 시 향후 72시간 내 예상 NVI 최저점")
    defense_effect: float = Field(description="전략 적용을 통한 NVI 방어 효과 (mitigated - baseline)")
    legal_and_pr_risk: str = Field(description="예상되는 대중의 반발 또는 법적 리스크")
    immediate_action_items: List[ActionItem] = Field(description="시간대별 구체적인 대응 액션 플랜")
    risk_matrix: Optional[List[dict]] = Field(default=None, description="리스크 매트릭스 (확률 × 영향도 정량화)")
    key_opinion_leaders: Optional[List[dict]] = Field(default=None, description="주요 오피니언 리더 (KOL) 식별 및 성향 분류")
    draft_statements: Optional[List[dict]] = Field(default=None, description="대응문 초안 (입장문/사과문 등)")
    benchmark_cases: Optional[List[dict]] = Field(default=None, description="유사사례 벤치마킹")
    sentiment_landscape: Optional[dict] = Field(default=None, description="여론 지형도 (감성 구성 + 부정 주제 클러스터링)")
    sentiment_timeline: Optional[dict] = Field(default=None, description="감성 타임라인 (과거 실측 + 미래 예측 + 이벤트 마커)")

# ==========================================
# 🕵️ [Reviewer] CCO 검토 결과 스키마
# ==========================================
class ReviewResult(BaseModel):
    is_approved: bool = Field(description="가이드라인을 완벽히 준수했으면 true, 아니면 false")
    feedback: str = Field(description="반려 시 구체적인 지시사항, 승인 시 '없음'")

# ==========================================
# 🏗️ LangGraph 공통 컨텍스트 데이터 버스
# ==========================================
class PipelineState(TypedDict):
    issue_id: str                       # 이슈 식별자 (DB 조회용)
    train_csv_path: str                 # 학습 데이터 경로 (720h 전체 생애주기)
    input_csv_path: str                 # 실전 입력 데이터 경로 (analyzer가 생성 또는 수동 지정)
    crisis_context: str
    crisis_type: str                    # SCCT 위기 유형 ("victim" | "accidental" | "preventable")
    forecaster_model: str               # 예측 모델 ("lightgbm" | "tft" | "moirai" | "arima")
    
    # 🔍 QueryBuilder 산출물
    search_keywords: list               # LLM이 추출한 핵심 키워드
    search_queries: list                # 벡터 검색용 자연어 쿼리 목록
    search_embeddings: list             # 검색 쿼리의 임베딩 벡터 (384차원)
    search_time_hint: str               # 검색 시간 범위 힌트
    
    # 📊 여론 지형도 + 감성 타임라인 + KOL (P0/P1 리포트 요소)
    sentiment_landscape: dict            # 여론 지형도 (감성 구성 + 부정 주제 클러스터)
    sentiment_timeline: dict             # 감성 타임라인 (시간별 감성 구성 + 이벤트 마커)
    key_opinion_leaders: list            # KOL 식별 결과 (author, platform, stance, influence)
    
    # 📊 Retriever 산출물 (메모리 전달 — DB SELECT는 Retriever에서만)
    retrieved_post_ids: list            # pgVector 검색으로 확보된 post_id 목록
    retrieved_posts: list               # 검색된 posts 전체 데이터 (dict 리스트)
    retrieved_comments: list            # 검색된 posts의 모든 댓글 (dict 리스트)
    retrieved_comment_count: int        # 검색된 댓글 수
    
    # 🎯 예측 결과의 이원화 (무대응 vs 전략 적용)
    # - LightGBM: list[float] (point forecast만)
    # - TFT: dict {"point": list, "lower": list, "upper": list} (신뢰구간 포함)
    actual_nvi_history: list            # 과거 실제 NVI 기록 (차트 렌더링용)
    nvi_baseline_forecast: list | dict  # 무대응(Do Nothing) 시 NVI 예측
    nvi_mitigated_forecast: list | dict # 전략 적용(Mitigated) 시 NVI 예측
    
    # 🎯 전략가 산출물
    strategist_timeline: list           # 전략가가 도출한 구조화된 대응 타임라인 (Action Type 배열)
    strategist_draft: str               # 전략가의 서술형 리포트
    risk_matrix: list                   # 리스크 매트릭스 (확률 × 영향도)
    draft_statements: list              # 대응문 초안 (입장문/사과문)
    benchmark_cases: list               # 유사사례 벤치마킹
    
    planner_instructions: str           # 플래너의 작업 지시서
    analyst_draft: str                  # 무대응 vs 전략적용 Gap 분석 리포트
    
    draft_report: str                   # 최종 취합된 JSON 리포트
    review_feedback: str
    is_approved: bool
    loop_count: int