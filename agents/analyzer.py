import json
from google import genai
from google.genai import types
from state import PipelineState, CrisisTimeline

class AnalyzerAgent:
    def __init__(self, client: genai.Client, model_name: str):
        self.client = client
        self.model_name = model_name

    def run(self, state: PipelineState) -> dict:
        print("\n▶️ [Agent 1] 상황 분석 및 향후 타임라인 추출 중...")
        prompt = f"다음 위기 상황(메타 정보)을 분석하여 향후 시뮬레이션할 타임라인 배열을 추출하세요:\n{state['crisis_context']}"
        
        res = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="당신은 위기 상황 분석기입니다. 텍스트를 읽고 미래의 대응 스케줄을 구조화된 JSON 배열로 출력하세요.",
                response_mime_type="application/json",
                response_schema=CrisisTimeline,
                temperature=0.1
            )
        )
        extracted = json.loads(res.text)
        print(f"   ✅ 미래 시나리오 이벤트 {len(extracted['events'])}개 분리 완료!")
        return {"timeline_events": extracted["events"]}