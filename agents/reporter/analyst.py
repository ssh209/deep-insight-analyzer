class AnalystAgent:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def run(self, state):
        print("   -> 📊 [Analyst] NVI 시뮬레이션 수치 분석 중 (병렬 실행)")
        prompt = f"""
        [현재 상황]: {state['crisis_context']}
        [TF 총괄의 지시]: {state['planner_instructions']}
        [NVI 예측 데이터]: {state['nvi_forecast']}
        
        당신은 데이터 분석가입니다. 위 지시에 맞춰 주어진 NVI 데이터의 하락 폭과 
        위험도(Alert Level)를 정량적으로 분석한 초안을 작성하세요.
        """
        res = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return {"analyst_draft": res.text}