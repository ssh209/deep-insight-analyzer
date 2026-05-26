class PlannerAgent:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def run(self, state):
        print(f"\n▶️ [Planner] 작업 기획 및 업무 할당 중... (루프: {state.get('loop_count', 0)+1}회차)")
        
        baseline_min = min(state['nvi_baseline_forecast'])
        
        prompt = f"""
        [현재 위기 상황]: {state['crisis_context']}
        [무대응 시 NVI 최저점]: {baseline_min:.3f} (1.0 만점 기준)
        
        당신은 위기관리 TF 총괄입니다. 무대응 시 NVI가 {baseline_min:.3f}까지 추락할 것으로 예측됩니다.
        산하의 'PR 전략가'에게 이 최저점을 최대한 방어할 수 있는 시간대별 대응 액션 플랜 수립을 지시하세요.
        또한 '데이터 분석가'에게는 무대응 시나리오와 전략 적용 시나리오 간의 정량적 갭(Gap) 분석을 준비하라고 지시하세요.
        명확한 2~3줄짜리 업무 지시서를 작성하세요.
        """
        if state.get("review_feedback"):
            prompt += f"\n[🚨 CCO 반려 사유]: {state['review_feedback']} (이 내용을 반영하도록 지시할 것)"
            
        res = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return {"planner_instructions": res.text}