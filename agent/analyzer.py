import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Pydantic으로 출력 스키마 정의 (가독성 폭발!)
class CrisisParams(BaseModel):
    action_type: int = Field(description="현재 기업 대응 상태 (0: 무대응, 1: 원론적 방어, 2: 공식 사과)")
    impact_weight: float = Field(description="여론 악화 가급력 (0.5 ~ 2.0). 메이저 언론 보도시 1.5 이상")
    duration_hours: int = Field(description="이슈 지속 예상 시간 (24, 48, 72, 120 중 택일)")
    reasoning: str = Field(description="수치 산정의 논리적 근거 1줄 요약")

# 2. 최신 Client 초기화 (vertexai=True 옵션으로 클라우드 망 탑승)
client = genai.Client(
    vertexai=True, 
    project="deep-insight-496705", # 본인 프로젝트 ID로 변경
    location="global"
)

system_instruction = """
당신은 최고 권위의 기업 PR 위기 평가 전문가입니다.
입력된 뉴스나 상황 보고를 읽고, 시뮬레이션 모델에 주입할 파라미터를 정확하게 추출하세요.
"""

def analyze_context_node(state: dict) -> dict:
    print("\n▶️ [Agent 1] 상황 분석 및 파라미터 추출 (최신 SDK) 가동...")
    crisis_context = state.get("crisis_context", "")
    
    prompt = f"다음 상황을 분석하여 파라미터를 추출하세요:\n\n{crisis_context}"
    
    # 3. generate_content 호출 (Pydantic 스키마 직접 주입)
    response = client.models.generate_content(
        model='gemini-3.1-pro-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=CrisisParams, # 여기서 마법이 일어납니다
            temperature=0.1
        )
    )
    
    # JSON 문자열을 딕셔너리로 변환 (응답이 완벽한 JSON임이 보장됨)
    import json
    extracted_params = json.loads(response.text)
    
    print(f"   ✅ 파라미터 추출 완료: {extracted_params}")
    return {"extracted_params": extracted_params}

if __name__ == "__main__":
    test_state = {"crisis_context": "[속보] A전자 배터리 화재 영상 확산 중. 회사 측은 확인 중이라며 선을 그었다."}
    analyze_context_node(test_state)