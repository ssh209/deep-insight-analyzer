import json
from google.genai import types
from state import CrisisTimeline

class StrategistAgent:
    """PR 전략가 에이전트.
    
    RAG 기반 과거 성공 사례를 참고하여:
    1. 구조화된 대응 타임라인(JSON) — Mitigated Forecaster의 입력으로 사용
    2. 서술형 전략 리포트 — Compiler의 입력으로 사용
    을 이중 출력합니다.
    """
    def __init__(self, client, model_name, vector_db):
        self.client = client
        self.model_name = model_name
        self.vector_db = vector_db

    def run(self, state):
        print("   -> 🛡️ [Strategist] RAG 기반 대응 타임라인 + 전략 리포트 수립 중...")
        rag_ctx = self.vector_db.similarity_search("과거 사례 반등", k=1)[0].page_content
        
        # 1단계: 구조화된 타임라인 추출 (Structured Output)
        timeline_prompt = f"""
        [현재 위기 상황]: {state['crisis_context']}
        [TF 총괄의 지시]: {state['planner_instructions']}
        [무대응 시 NVI 최저점]: {min(state['nvi_baseline_forecast']):.3f}
        [과거 성공 사례]: {rag_ctx}
        
        당신은 PR 전략가입니다. 위 상황과 과거 사례를 참고하여, 
        NVI 하락을 최대한 방어할 수 있는 시간대별 대응 액션 타임라인을 구조화된 JSON으로 출력하세요.
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
        print(f"   ✅ 대응 타임라인 {len(timeline['events'])}개 이벤트 도출 완료!")
        
        # 2단계: 서술형 전략 리포트 작성
        draft_prompt = f"""
        [현재 위기 상황]: {state['crisis_context']}
        [TF 총괄의 지시]: {state['planner_instructions']}
        [무대응 시 NVI 최저점]: {min(state['nvi_baseline_forecast']):.3f}
        [과거 성공 사례]: {rag_ctx}
        [수립된 대응 타임라인]: {json.dumps(timeline['events'], ensure_ascii=False)}
        
        당신은 PR 전략가입니다. 위 대응 타임라인의 근거와 기대 효과를 설명하고,
        시간대별 구체적인 위기 대응 액션 아이템과 예상되는 리스크를 서술형으로 작성하세요.
        """
        draft_res = self.client.models.generate_content(model=self.model_name, contents=draft_prompt)
        
        return {
            "strategist_timeline": timeline["events"],
            "strategist_draft": draft_res.text
        }