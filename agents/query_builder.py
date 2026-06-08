"""
QueryBuilderAgent — 사용자 입력 → 벡터 검색 쿼리 변환

파이프라인 최전방에서:
1. Google Search Grounding으로 실시간 이슈 맥락 조사 (Pro 모델)
2. 맥락 기반 벡터 서치 최적화 검색 쿼리 추출 (Flash 모델, JSON)
3. 각 쿼리를 임베딩 벡터로 변환 (multilingual-e5-small)
   - E5 모델은 쿼리에 'query: ' prefix 필요

출력: search_queries(텍스트), search_embeddings(벡터)
→ RetrieverAgent가 이를 받아 pgVector 검색 수행
"""
import json
import logging
from typing import List
from pydantic import BaseModel, Field
from google.genai import types

logger = logging.getLogger(__name__)


# ==========================================
# 📋 LLM 출력 스키마
# ==========================================
class SearchQueries(BaseModel):
    keywords: List[str] = Field(
        description="위기 상황의 핵심 키워드 (예: 'XX전자', '배터리 발화', '은폐')"
    )
    search_queries: List[str] = Field(
        description="벡터 검색에 사용할 자연어 쿼리 (5~10개). "
                    "다양한 관점과 표현을 포함하여 리콜 범위를 넓힘."
    )
    time_range_hint: str = Field(
        description="검색 대상 시간 범위 힌트 (예: '최근 7일', '최근 30일', '전체')"
    )


SYSTEM_PROMPT = (
    "당신은 PR 위기 분석을 위한 검색 쿼리 설계 전문가입니다. "
    "사용자의 위기 상황 설명을 분석하여, 소셜 미디어 게시글과 댓글을 "
    "벡터 유사도 검색으로 찾기 위한 최적의 검색 쿼리를 설계합니다."
)

# Step 1: Google Search Grounding 프롬프트
GROUNDING_PROMPT_TEMPLATE = """다음 위기 상황에 대해 최신 뉴스와 여론 동향을 조사하세요.

[사용자 입력]
{user_input}

조사해야 할 내용:
1. 이 이슈의 최신 진행 상황 (언제 발생, 현재 단계)
2. 관련된 핵심 인물, 기업, 제품
3. 소셜 미디어에서의 주요 반응과 키워드
4. 관련 뉴스 보도의 핵심 내용
5. 여론의 흐름 (비판/옹호/조롱 등)

위 내용을 간결하게 정리하세요.
"""

# Step 2: 맥락 기반 검색 쿼리 생성 프롬프트
QUERY_PROMPT_TEMPLATE = """아래 위기 상황 설명과 실시간 조사 결과를 분석하여, 관련 게시글/댓글을 찾기 위한 검색 쿼리를 설계하세요.

[사용자 입력]
{user_input}

[실시간 이슈 조사 결과]
{search_context}

[요구사항]
1. keywords: 핵심 키워드를 추출하세요 (기업명, 이슈 키워드, 관련 인물/제품 등)
2. search_queries: 벡터 검색에 사용할 자연어 쿼리를 5~10개 생성하세요.
   - 직접적 표현: "XX전자 배터리 발화 사고"
   - 감성 표현: "XX전자 불매 운동 시작"
   - 밈/조롱 표현: "XX전자 폰 폭발 ㅋㅋ"
   - 옹호 표현: "XX전자 억울하다 오해"
   - 뉴스 표현: "XX전자 리콜 공식 입장"
   - 조사 결과에서 발견된 실제 사용 표현도 포함하세요.
3. time_range_hint: 적절한 검색 시간 범위를 제안하세요.
"""


class QueryBuilderAgent:
    """사용자 입력 → Google Search Grounding → 검색 쿼리 추출 → 임베딩 벡터 변환.

    2-step LLM 호출:
      Step 1: Pro 모델 + Google Search로 실시간 이슈 맥락 조사
      Step 2: Flash 모델로 구조화된 검색 쿼리 JSON 추출
    """

    def __init__(self, client, model_name: str, embeddings,
                 grounding_model_name: str = None):
        self.client = client
        self.model_name = model_name              # Step 2: JSON 추출 (Flash)
        self.grounding_model_name = (              # Step 1: Grounding (Pro)
            grounding_model_name or model_name
        )
        self.embeddings = embeddings               # HuggingFaceEmbeddings 인스턴스

    def run(self, state: dict) -> dict:
        user_input = state["crisis_context"]
        print(f"\n>> [QueryBuilder] 사용자 입력 분석 시작...")
        print(f"   입력: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")

        # ── Step 1: Google Search Grounding으로 실시간 맥락 조사 ──
        search_context = self._grounding_search(user_input)

        # ── Step 2: 맥락 기반 구조화된 검색 쿼리 생성 ──
        prompt = QUERY_PROMPT_TEMPLATE.format(
            user_input=user_input,
            search_context=search_context,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=SearchQueries,
                temperature=0.3,
            ),
        )

        result = json.loads(response.text)
        queries = result["search_queries"]
        keywords = result["keywords"]
        time_hint = result.get("time_range_hint", "최근 7일")

        print(f"   [OK] 키워드 {len(keywords)}개: {keywords}")
        print(f"   [OK] 검색 쿼리 {len(queries)}개 생성")
        print(f"   [OK] 시간 범위: {time_hint}")

        # 3. 검색 쿼리 → 임베딩 벡터 변환 (E5: 'query: ' prefix 필요)
        prefixed_queries = [f"query: {q}" for q in queries]
        search_embeddings = self.embeddings.embed_documents(prefixed_queries)
        print(f"   [OK] 임베딩 벡터 {len(search_embeddings)}개 생성 (dim={len(search_embeddings[0])})")
        return {
            "search_keywords": keywords,
            "search_queries": queries,
            "search_embeddings": search_embeddings,
            "search_time_hint": time_hint,
        }

    # ==========================================
    # Step 1: Google Search Grounding
    # ==========================================
    def _grounding_search(self, user_input: str) -> str:
        """Google Search Grounding으로 실시간 이슈 맥락을 조사합니다.

        실패 시 빈 문자열을 반환하여 Step 2가 사용자 입력만으로 동작합니다.
        """
        print(f"   🔍 [Step 1] Google Search Grounding 조사 중... (model: {self.grounding_model_name})")
        try:
            grounding_response = self.client.models.generate_content(
                model=self.grounding_model_name,
                contents=GROUNDING_PROMPT_TEMPLATE.format(user_input=user_input),
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2,
                ),
            )
            context = grounding_response.text
            print(f"   [OK] 실시간 맥락 조사 완료 ({len(context)}자)")
            return context
        except Exception as e:
            logger.warning(f"Google Search Grounding 실패, 기본 모드로 진행: {e}")
            print(f"   [WARN] Grounding 실패: {e} — 사용자 입력만으로 진행")
            return "(실시간 조사 결과 없음 — 사용자 입력 기반으로 쿼리를 설계하세요)"
