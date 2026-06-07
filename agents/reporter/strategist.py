import json
from typing import List, Optional
from google.genai import types
from pydantic import BaseModel, Field
from state import CrisisTimeline, RiskItem

# ==========================================
# Structured Output 스키마
# ==========================================
class RiskMatrixOutput(BaseModel):
    risks: List[RiskItem] = Field(description="식별된 리스크 목록 (4~8개)")

class BenchmarkCase(BaseModel):
    case_name: str = Field(description="사례명 (예: YY자동차 리콜 은폐 (2023))")
    crisis_type: str = Field(description="SCCT 위기 유형 (victim/accidental/preventable)")
    similarity_score: float = Field(description="현재 위기와의 유사도 (0.0~1.0)")
    initial_response_hours: Optional[int] = Field(default=None, description="첫 대응까지 걸린 시간")
    nvi_bottom: Optional[float] = Field(default=None, description="NVI 최저점")
    recovery_days: Optional[int] = Field(default=None, description="NVI 0.5 회복까지 일수")
    lesson: str = Field(description="핵심 교훈 (1~2문장)")

class BenchmarkOutput(BaseModel):
    cases: List[BenchmarkCase] = Field(description="유사 위기 사례 목록 (2~4개)")

class DraftStatement(BaseModel):
    type: str = Field(description="문서 유형 (예: 1차 입장문, 2차 사과문, SNS 대응문)")
    target_audience: str = Field(description="대상 (예: 언론, SNS, 내부)")
    tone: str = Field(description="어조 (예: 진정성 있는 사과 + 구체적 조치)")
    draft: str = Field(description="실제 발표 가능한 공식 입장문 전문")

class DraftStatementsOutput(BaseModel):
    statements: List[DraftStatement] = Field(description="대응문 초안 목록 (1~3건)")


class StrategistAgent:
    """PR 전략가 에이전트.
    
    RAG 기반 과거 성공 사례를 참고하여:
    1. 구조화된 대응 타임라인(JSON) — Mitigated Forecaster 입력
    2. 서술형 전략 리포트 — Compiler 입력
    3. 리스크 매트릭스 — 확률 × 영향도 정량화
    4. 유사사례 벤치마킹 — RAG 기반 과거 사례 비교
    5. 대응문 초안 — 실제 발표 가능한 입장문/사과문
    을 5중 출력합니다.
    """
    def __init__(self, client, model_name, vector_db):
        self.client = client
        self.model_name = model_name
        self.vector_db = vector_db

    def run(self, state):
        print("   -> 🛡️ [Strategist] 전략 수립 중 (타임라인 + 리포트 + 리스크 + 벤치마크 + 대응문)...")
        rag_results = self.vector_db.similarity_search("과거 위기 사례 대응 결과", k=3)
        rag_ctx = "\n".join([doc.page_content for doc in rag_results])
        
        # NVI 수치 추출 (list 또는 dict 대응)
        baseline = state['nvi_baseline_forecast']
        baseline_points = baseline["point"] if isinstance(baseline, dict) else baseline
        baseline_min = min(baseline_points)
        crisis_type = state.get('crisis_type', 'accidental')
        
        # ============================================================
        # 1단계: 구조화된 타임라인 추출
        # ============================================================
        timeline_prompt = f"""
        [현재 위기 상황]: {state['crisis_context']}
        [TF 총괄의 지시]: {state['planner_instructions']}
        [무대응 시 NVI 최저점]: {baseline_min:.3f}
        [과거 성공 사례]: {rag_ctx}
        
        당신은 PR 전략가입니다. 위 상황과 과거 사례를 참고하여, 
        향후 1주일(168시간) 이내에 실행할 수 있는 시간대별 대응 액션 타임라인을 구조화된 JSON으로 출력하세요.
        모든 이벤트의 hour_offset은 반드시 0~168 범위여야 합니다.
        각 이벤트는 hour_offset(몇 시간 뒤), action_type(0: 무대응, 1: 1차 입장문, 2: 2차 사과문), 
        influencer_hit(0/1)로 구성합니다.
        """
        
        timeline_res = self.client.models.generate_content(
            model=self.model_name,
            contents=timeline_prompt,
            config=types.GenerateContentConfig(
                system_instruction="당신은 위기관리 PR 전략가입니다. 상황을 분석하여 최적의 대응 타임라인을 설계하세요.",
                response_mime_type="application/json",
                response_schema=CrisisTimeline,
                temperature=0.2
            )
        )
        timeline = json.loads(timeline_res.text)
        print(f"   ✅ [1/5] 대응 타임라인 {len(timeline['events'])}개 이벤트")
        
        # ============================================================
        # 2단계: 서술형 전략 리포트
        # ============================================================
        draft_prompt = f"""
        [현재 위기 상황]: {state['crisis_context']}
        [TF 총괄의 지시]: {state['planner_instructions']}
        [무대응 시 NVI 최저점]: {baseline_min:.3f}
        [과거 성공 사례]: {rag_ctx}
        [수립된 대응 타임라인]: {json.dumps(timeline['events'], ensure_ascii=False)}
        
        당신은 PR 전략가입니다. 위 대응 타임라인의 근거와 기대 효과를 설명하고,
        시간대별 구체적인 위기 대응 액션 아이템과 예상되는 리스크를 서술형으로 작성하세요.
        """
        draft_res = self.client.models.generate_content(model=self.model_name, contents=draft_prompt)
        print(f"   ✅ [2/5] 전략 리포트 작성")
        
        # ============================================================
        # 3단계: 리스크 매트릭스
        # ============================================================
        landscape = state.get('sentiment_landscape', {})
        landscape_summary = ""
        if landscape:
            overview = landscape.get("overview", {})
            themes = landscape.get("top_negative_themes", [])
            landscape_summary = f"""
        [여론 지형도]
        - 부정 비율: {overview.get('negative_ratio', 0):.1%}, 조롱 비율: {overview.get('mockery_ratio', 0):.1%}
        - 부정 핵심 주제: {json.dumps(themes[:3], ensure_ascii=False)[:200]}"""
        
        risk_prompt = f"""
        [현재 위기 상황]: {state['crisis_context']}
        [위기 유형 (SCCT)]: {crisis_type}
        [무대응 시 NVI 최저점]: {baseline_min:.3f}
        [수립된 대응 타임라인]: {json.dumps(timeline['events'], ensure_ascii=False)}
        [과거 성공 사례]: {rag_ctx}{landscape_summary}
        
        당신은 위기관리 리스크 분석 전문가입니다. 
        현재 위기 상황에서 발생 가능한 리스크를 4~8개 식별하고,
        각 리스크에 대해 확률(probability), 영향도(impact), 범주(category), 완화 전략(mitigation)을 평가하세요.
        
        범주: legal(법적), reputation(평판), competitive(경쟁), operational(운영)
        확률: low / medium / high
        영향도: low / medium / high / critical
        """
        
        risk_res = self.client.models.generate_content(
            model=self.model_name,
            contents=risk_prompt,
            config=types.GenerateContentConfig(
                system_instruction="당신은 위기관리 리스크 분석 전문가입니다. 구조화된 리스크 평가를 제공하세요.",
                response_mime_type="application/json",
                response_schema=RiskMatrixOutput,
                temperature=0.2
            )
        )
        risk_matrix = json.loads(risk_res.text)
        risks = risk_matrix.get("risks", [])
        print(f"   ✅ [3/5] 리스크 매트릭스 {len(risks)}개 리스크")
        
        # ============================================================
        # 4단계: 유사사례 벤치마킹 (RAG 확장)
        # ============================================================
        benchmark_prompt = f"""
        [현재 위기 상황]: {state['crisis_context']}
        [위기 유형 (SCCT)]: {crisis_type}
        [과거 사례 DB]: {rag_ctx}
        
        당신은 위기관리 사례 분석 전문가입니다.
        현재 위기와 유사한 과거 사례를 2~4개 식별하세요.
        실제 발생했던 기업 위기 사례를 기반으로 하되, 
        정확한 데이터가 없으면 합리적으로 추정하세요.
        
        각 사례에 대해:
        - case_name: 사례명 (기업명 + 이슈 + 연도)
        - crisis_type: SCCT 위기 유형
        - similarity_score: 현재 위기와의 유사도 (0.0~1.0)
        - initial_response_hours: 첫 대응까지 걸린 시간
        - nvi_bottom: NVI 최저점 (추정)
        - recovery_days: 회복까지 일수
        - lesson: 핵심 교훈 (1~2문장)
        """
        
        benchmark_res = self.client.models.generate_content(
            model=self.model_name,
            contents=benchmark_prompt,
            config=types.GenerateContentConfig(
                system_instruction="당신은 위기관리 사례 분석 전문가입니다. 유사 사례를 비교 분석하세요.",
                response_mime_type="application/json",
                response_schema=BenchmarkOutput,
                temperature=0.3
            )
        )
        benchmark_data = json.loads(benchmark_res.text)
        benchmarks = benchmark_data.get("cases", [])
        print(f"   ✅ [4/5] 유사사례 벤치마킹 {len(benchmarks)}건")
        
        # ============================================================
        # 5단계: 대응문 초안 생성
        # ============================================================
        statement_prompt = f"""
        [현재 위기 상황]: {state['crisis_context']}
        [위기 유형 (SCCT)]: {crisis_type}
        [수립된 대응 타임라인]: {json.dumps(timeline['events'], ensure_ascii=False)}
        [전략 리포트]: {draft_res.text[:500]}
        {landscape_summary}
        
        당신은 위기관리 커뮤니케이션 전문가입니다.
        현재 상황에 맞는 공식 대응문 초안을 작성하세요.
        
        다음 규칙을 반드시 준수하세요:
        - '오해', '유감'이라는 단어를 절대 사용하지 마세요
        - 책임을 전가하는 뉘앙스를 금지합니다
        - 구체적인 조치 사항을 포함하세요
        - 진정성 있는 어조를 사용하세요
        
        1~2건의 대응문을 작성하세요:
        - 1차 입장문 (즉시 발표용, 언론+SNS 대상)
        - 필요 시 2차 사과문 (24~48시간 후, 조사 결과 포함)
        """
        
        statement_res = self.client.models.generate_content(
            model=self.model_name,
            contents=statement_prompt,
            config=types.GenerateContentConfig(
                system_instruction="당신은 위기관리 커뮤니케이션 전문가입니다. 실제 발표 가능한 수준의 공식 대응문을 작성하세요.",
                response_mime_type="application/json",
                response_schema=DraftStatementsOutput,
                temperature=0.3
            )
        )
        statement_data = json.loads(statement_res.text)
        statements = statement_data.get("statements", [])
        print(f"   ✅ [5/5] 대응문 초안 {len(statements)}건")
        
        return {
            "strategist_timeline": timeline["events"],
            "strategist_draft": draft_res.text,
            "risk_matrix": [r.model_dump() if hasattr(r, 'model_dump') else r for r in risks],
            "benchmark_cases": [b.model_dump() if hasattr(b, 'model_dump') else b for b in benchmarks],
            "draft_statements": [s.model_dump() if hasattr(s, 'model_dump') else s for s in statements],
        }