class AnalystAgent:
    """데이터 분석가 에이전트.
    
    무대응(Baseline) vs 전략 적용(Mitigated) NVI 예측 결과를 비교하여
    방어 효과(Gap)를 정량적으로 분석합니다.
    """
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def run(self, state):
        print("   -> 📊 [Analyst] 무대응 vs 전략 적용 Gap 분석 중...")
        
        baseline_min = min(state['nvi_baseline_forecast'])
        mitigated_min = min(state['nvi_mitigated_forecast'])
        defense = mitigated_min - baseline_min
        
        prompt = f"""
        [TF 총괄의 지시]: {state['planner_instructions']}
        
        [무대응(Do Nothing) 시 NVI 72시간 예측]: {state['nvi_baseline_forecast']}
        [전략 적용(Mitigated) 시 NVI 72시간 예측]: {state['nvi_mitigated_forecast']}
        
        [핵심 수치 요약]
        - 무대응 시 최저점: {baseline_min:.3f}
        - 전략 적용 시 최저점: {mitigated_min:.3f}
        - 방어 효과: +{defense:.3f} 포인트
        
        당신은 데이터 분석가입니다. 위 두 시나리오의 NVI 추이를 비교 분석하여:
        1. 무대응 시 최악의 구간과 위험도(Alert Level: RED/ORANGE/YELLOW)
        2. 전략 적용 시 방어 효과의 크기와 ROI 평가
        3. NVI 반등 시점 비교 (무대응 vs 전략 적용)
        4. 경영진에게 전달할 핵심 인사이트
        를 정량적으로 분석한 초안을 작성하세요.
        """
        res = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return {"analyst_draft": res.text}