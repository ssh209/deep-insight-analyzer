import json
from google.genai import types
from state import CrisisReport

class CompilerAgent:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def run(self, state):
        print("▶️ [Compiler] 에이전트 결과물 취합 및 JSON 포맷팅 중...")
        
        baseline_min = min(state['nvi_baseline_forecast'])
        mitigated_min = min(state['nvi_mitigated_forecast'])
        defense = mitigated_min - baseline_min
        
        prompt = f"""
        [데이터 분석가의 Gap 분석]: {state['analyst_draft']}
        [PR 전략가의 플랜]: {state['strategist_draft']}
        
        [핵심 수치]
        - 무대응(Baseline) 시 NVI 최저점: {baseline_min:.3f}
        - 전략 적용(Mitigated) 시 NVI 최저점: {mitigated_min:.3f}
        - 방어 효과(defense_effect): {defense:.3f}
        
        당신은 최종 보고서 취합자입니다. 두 전문가의 초안을 완벽하게 융합하여, 
        위 핵심 수치를 정확히 반영한 JSON 스키마 규격에 맞춰 출력하세요.
        baseline_nvi_bottom, mitigated_nvi_bottom, defense_effect 값은 반드시 위 수치를 사용하세요.
        """
        res = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CrisisReport,
                temperature=0.1
            )
        )
        return {"draft_report": res.text, "loop_count": state.get("loop_count", 0) + 1}