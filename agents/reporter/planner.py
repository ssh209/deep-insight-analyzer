class PlannerAgent:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def run(self, state):
        print(f"\n▶️ [Planner] 작업 기획 및 업무 할당 중... (루프: {state.get('loop_count', 0)+1}회차)")
        prompt = f"""
        [상황]: {state['crisis_context']}
        [NVI 예측 최저점]: {min(state['nvi_forecast'])}
        
        당신은 위기관리 TF 총괄입니다. 산하의 '데이터 분석가'와 'PR 전략가'가 각각 어떤 부분을 중점적으로 
        작성해야 할지 명확한 2~3줄짜리 업무 지시서를 작성하세요.
        """
        if state.get("review_feedback"):
            prompt += f"\n[🚨 CCO 반려 사유]: {state['review_feedback']} (이 내용을 반영하도록 지시할 것)"
            
        res = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return {"planner_instructions": res.text}