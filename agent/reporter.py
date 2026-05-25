import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ==========================================
# 1. 리뷰어용 Pydantic 스키마 정의
# ==========================================
class ReviewResult(BaseModel):
    is_approved: bool = Field(description="가이드라인을 완벽히 준수했으면 true, 아니면 false")
    feedback: str = Field(description="반려 시 구체적인 수정 지시사항. 승인 시 '없음'")

# ==========================================
# 2. Client 설정 (모든 최신 모델을 쓸 수 있는 global 리전)
# ==========================================
client = genai.Client(vertexai=True, project="deep-insight-496705", location="global")

# ==========================================
# 3. RAG 벡터 DB (Chroma) 임시 세팅
# ==========================================
print("⏳ 로컬 RAG(벡터 DB) 초기화 중...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

dummy_docs = [
    Document(page_content="[과거사례] 48시간 내 CEO 사과시 여론 회복됨.", metadata={"type": "history"}),
    Document(page_content="[금칙어] 절대 '유감'이라는 단어를 쓰지 말 것. 책임을 회피하는 느낌을 줌.", metadata={"type": "guideline"})
]
vector_db = Chroma.from_documents(dummy_docs, embeddings)

# ==========================================
# 📝 4. Report Generation Agent (2-1)
# ==========================================
def generate_report_node(state: dict) -> dict:
    print("\n▶️ [Agent 2-1] 리포트 초안 작성 중...")
    
    context = state.get("crisis_context", "")
    forecast = state.get("nvi_forecast", [])
    feedback = state.get("review_feedback", "")
    
    # RAG 검색: 과거 사례만 가져옴
    rag_context = vector_db.similarity_search("과거 사례", k=1)[0].page_content
    
    prompt = f"[위기 상황]: {context}\n[NVI 예측]: {forecast[:5]}\n[과거 사례]: {rag_context}\n위 내용으로 경영진 보고용 초안을 작성하세요."
    
    # 반려당해서 피드백이 있는 경우 프롬프트에 추가
    if feedback:
        print(f"   ⚠️ 반려 사유 반영 중: {feedback}")
        prompt += f"\n[이전 반려 사유 반영 필수!]: {feedback}"
        
    response = client.models.generate_content(
        model='gemini-3.1-pro-preview', 
        contents=prompt
    )
    
    print("   ✅ 초안 작성 완료!")
    print(response.text)
    return {"draft_report": response.text}

# ==========================================
# 🕵️ 5. Report Review Agent (2-2)
# ==========================================
def review_report_node(state: dict) -> dict:
    print("\n▶️ [Agent 2-2] PR 가이드라인 검토 중...")
    
    draft = state.get("draft_report", "")
    
    # RAG 검색: 사내 가이드라인/금칙어만 가져옴
    rag_guidelines = vector_db.similarity_search("금칙어", k=1)[0].page_content
    
    prompt = f"[PR 가이드라인]: {rag_guidelines}\n[리포트 초안]: {draft}\n위 초안이 가이드라인을 어겼는지 깐깐하게 검토하세요."
    
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="당신은 깐깐한 홍보 책임자입니다.",
            response_mime_type="application/json",
            response_schema=ReviewResult, # Pydantic 모델을 여기에 주입
            temperature=0.1
        )
    )
    
    result = json.loads(response.text)
    
    if result["is_approved"]:
        print("   ✅ 검토 결과: 최종 승인")
    else:
        print(f"   ❌ 검토 결과: 반려 (사유: {result['feedback']})")
        
    return {"is_approved": result["is_approved"], "review_feedback": result["feedback"]}

# ==========================================
# 🚀 6. 테스트 실행부 (순환 루프 제어)
# ==========================================
if __name__ == "__main__":
    current_state = {
        "crisis_context": "[속보] A전자 배터리 화재 영상 확산 중. 회사 측은 '일부 오해일 뿐이며 유감'이라는 1차 입장문 배포.",
        "nvi_forecast": [0.599, 0.550, 0.400, 0.350, 0.300], 
        "review_feedback": ""
    }
    
    print("\n🚀 PR 위기 대응 리포트 자동 생성 파이프라인 시작")
    print("=" * 60)

    max_loops = 3
    for i in range(max_loops):
        print(f"\n🔄 --- [루프 {i+1}회차] ---")
        
        # 1. 초안 작성
        gen_result = generate_report_node(current_state)
        current_state.update(gen_result)
        
        # 2. 초안 검토
        rev_result = review_report_node(current_state)
        current_state.update(rev_result)
        
        # 3. 승인 시 루프 탈출
        if current_state["is_approved"]:
            print("\n🎉 최종 리포트가 CCO 에이전트의 승인을 받았습니다!")
            print("=" * 60)
            print(current_state["draft_report"])
            print("=" * 60)
            break
        else:
            print("   -> 📝 CCO의 피드백을 반영하여 초안을 다시 작성합니다...")
            
    if not current_state.get("is_approved"):
        print("\n⚠️ 최대 루프 횟수를 초과했습니다. 경영진의 직접 검토가 필요합니다.")