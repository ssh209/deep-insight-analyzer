import json
from google import genai
from google.genai import types
from state import PipelineState, ReviewResult

class ReviewerAgent:
    def __init__(self, client: genai.Client, model_name: str, vector_db):
        self.client = client
        self.model_name = model_name
        self.vector_db = vector_db

    def run(self, state: PipelineState) -> dict:
        print("\n▶️ [Agent 3-2] 사내 리스크 관리 규정 검토 중 (CCO 레드팀)...")
        rag_guide = self.vector_db.similarity_search("금칙어", k=1)[0].page_content
        
        prompt = f"""
        [사내 PR 준수 가이드라인]:
        {rag_guide}

        [검토 가상 대상 리포트 객체 (JSON)]:
        {state['draft_report']}

        리포트 내부 텍스트 속 'executive_summary', 'legal_and_pr_risk', 'action' 데이터 필드에 
        '유감', '오해', '착오' 등 사내 금칙어가 매칭되어 면피성 기조가 감지되는지 필터링하십시오.
        """
        
        res = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewResult,
                temperature=0.1
            )
        )
        result = json.loads(res.text)
        
        if result["is_approved"]: 
            print("   ✅ 검토 결과: 규정 적합 판정 (최종 승인)")
        else: 
            print(f"   ❌ 검토 결과: 규정 위반 적발 (반려 사유: {result['feedback']})")
            
        return {"is_approved": result["is_approved"], "review_feedback": result["feedback"]}