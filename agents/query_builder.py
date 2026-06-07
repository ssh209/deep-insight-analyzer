"""
QueryBuilderAgent — 사용자 입력 → 벡터 검색 쿼리 변환

파이프라인 최전방에서:
1. 사용자의 자연어 위기 상황 설명을 LLM으로 분석
2. 벡터 서치에 최적화된 검색 쿼리 목록 추출
3. 각 쿼리를 임베딩 벡터로 변환 (multilingual-e5-small)
   - E5 모델은 쿼리에 'query: ' prefix 필요

출력: search_queries(텍스트), search_embeddings(벡터)
→ RetrieverAgent가 이를 받아 pgVector 검색 수행
"""
import json
from typing import List
from pydantic import BaseModel, Field
from google.genai import types


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

QUERY_PROMPT_TEMPLATE = """아래 위기 상황 설명을 분석하여, 관련 게시글/댓글을 찾기 위한 검색 쿼리를 설계하세요.

[사용자 입력]
{user_input}

[요구사항]
1. keywords: 핵심 키워드를 추출하세요 (기업명, 이슈 키워드, 관련 인물/제품 등)
2. search_queries: 벡터 검색에 사용할 자연어 쿼리를 5~10개 생성하세요.
   - 직접적 표현: "XX전자 배터리 발화 사고"
   - 감성 표현: "XX전자 불매 운동 시작"
   - 밈/조롱 표현: "XX전자 폰 폭발 ㅋㅋ"
   - 옹호 표현: "XX전자 억울하다 오해"
   - 뉴스 표현: "XX전자 리콜 공식 입장"
3. time_range_hint: 적절한 검색 시간 범위를 제안하세요.
"""


class QueryBuilderAgent:
    """사용자 입력 → LLM 키워드 추출 → 임베딩 벡터 변환 에이전트."""

    def __init__(self, client, model_name: str, embeddings):
        self.client = client
        self.model_name = model_name
        self.embeddings = embeddings      # HuggingFaceEmbeddings 인스턴스

    def run(self, state: dict) -> dict:
        user_input = state["crisis_context"]
        print(f"\n>> [QueryBuilder] 사용자 입력 분석 시작...")
        print(f"   입력: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")

        # 1. LLM으로 검색 쿼리 추출
        prompt = QUERY_PROMPT_TEMPLATE.format(user_input=user_input)

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

        # 2. 검색 쿼리 → 임베딩 벡터 변환 (E5: 'query: ' prefix 필요)
        prefixed_queries = [f"query: {q}" for q in queries]
        search_embeddings = self.embeddings.embed_documents(prefixed_queries)
        print(f"   [OK] 임베딩 벡터 {len(search_embeddings)}개 생성 (dim={len(search_embeddings[0])})")
        return {
            "search_keywords": keywords,
            "search_queries": queries,
            "search_embeddings": search_embeddings,
            "search_time_hint": time_hint,
        }
