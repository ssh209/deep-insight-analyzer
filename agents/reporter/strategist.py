class StrategistAgent:
    def __init__(self, client, model_name, vector_db):
        self.client = client
        self.model_name = model_name
        self.vector_db = vector_db

    def run(self, state):
        print("   -> 🛡️ [Strategist] 과거 RAG 기반 액션 플랜 수립 중 (병렬 실행)")
        rag_ctx = self.vector_db.similarity_search("과거 사례 반등", k=1)[0].page_content
        prompt = f"""
        [현재 상황]: {state['crisis_context']}
        [TF 총괄의 지시]: {state['planner_instructions']}
        [과거 성공 사례]: {rag_ctx}
        
        당신은 PR 전략가입니다. 상황과 과거 사례를 참고하여, 시간대별 구체적인 
        위기 대응 액션 아이템과 예상되는 리스크를 작성하세요.
        """
        res = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return {"strategist_draft": res.text}