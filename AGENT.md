# Issue Cracker — 코딩 규칙 (AGENT.md)

> **이 문서는 AI 에이전트와 개발자 모두를 위한 코딩 컨벤션입니다.**
> 코드 생성/수정 시 반드시 이 규칙을 준수하세요.

---

## 1. Import 규칙

- **모든 `import`/`from ... import` 문은 파일 최상단에만 위치**시킨다.
- 함수, 메서드, 조건문 내부의 인라인 import는 금지한다.
- 순서: 표준 라이브러리 → 서드파티 → 프로젝트 내부 모듈 (각 그룹 사이에 빈 줄)

```python
# ✅ 올바른 예
import os
import json

import pandas as pd
from google import genai

from state import PipelineState
from agents.analyzer import AnalyzerAgent
```

```python
# ❌ 금지: 함수 내부 import
def run_batch(self):
    from google.cloud import storage  # ← 금지
```

---

## 2. 네이밍 컨벤션

| 대상 | 규칙 | 예시 |
|:---|:---|:---|
| 파일명 | `snake_case.py` | `query_builder.py` |
| 클래스명 | `PascalCase` | `QueryBuilderAgent` |
| 함수/메서드 | `snake_case` | `_load_unanalyzed()` |
| 상수 | `UPPER_SNAKE_CASE` | `SIMILARITY_THRESHOLD` |
| DB 컬럼 | `snake_case` | `issue_id`, `created_at` |

---

## 3. DB 스키마 규칙

- **FK(Foreign Key) 제약조건을 사용하지 않는다.** 관계는 앱 레벨에서 관리.
- 테이블 참조 시 반드시 `issue_cracker.` 스키마 접두사를 붙인다.
- DB 식별자: `crisis_id` ❌ → `issue_id` ✅
- `posts`, `comments`는 **이슈 독립** (issue_id 컬럼 없음). `analysis_results`만 `issue_id`로 연결.

---

## 4. 에이전트 구조

### 클래스 구조
- 모든 에이전트는 `run(self, state: dict) -> dict` 메서드를 가진다.
- `run()`은 LangGraph 노드 핸들러로 바인딩된다.
- 반환값은 `PipelineState`에 머지될 dict이다.

### 비동기 에이전트
- DB 접근이 필요한 에이전트(`AnalyzerAgent`, `RetrieverAgent`)는 `async def run()`을 사용한다.
- DB 풀은 생성자 주입(`__init__`의 `db_pool` 파라미터).

### 파이프라인 흐름 (DB 모드)
```
START → QueryBuilder → Retriever → Analyzer
      ↳ Analyzer 산출물: input_csv, sentiment_landscape, sentiment_timeline
      → Baseline Forecaster → Planner → Strategist
      → Mitigated Forecaster → Analyst → Compiler
      ↳ Compiler: landscape + timeline + NVI 예측을 CrisisReport에 통합
      → Reviewer → END (or ≤3회 Planner 복귀)
```

---

## 5. PipelineState 규칙

- 모든 파이프라인 공유 데이터는 `state.py`의 `PipelineState(TypedDict)`에 정의한다.
- 새 필드 추가 시 반드시 타입 어노테이션 + 주석을 단다.
- 에이전트 `run()`의 반환 dict 키는 반드시 `PipelineState`에 선언된 키여야 한다.

---

## 6. SQL 작성 규칙

- 파라미터 바인딩: `$1, $2, ...` (asyncpg 스타일)
- 키워드: 대문자 (`SELECT`, `FROM`, `WHERE`, ...)
- 들여쓰기: 4칸 (Python 코드 내 SQL 문자열)
- `ON CONFLICT ... DO NOTHING` 또는 `DO UPDATE`로 멱등성 보장

---

## 7. pgVector 사용 규칙

- 임베딩 모델: `intfloat/multilingual-e5-small` (384차원)
- 컬럼 타입: `vector(384)`
- 유사도 연산자: `<=>` (코사인 거리)
- 유사도 값 변환: `1 - (embedding <=> $1::vector) AS similarity`
- 검색 수 제한 없음. `SIMILARITY_THRESHOLD` (기본 0.5)로만 필터.
- **E5 prefix 규칙** (필수):
  - 검색 쿼리: `"query: " + query_text`
  - 문서 임베딩: `"passage: " + document_text`
  - prefix 없이 인코딩하면 성능 크게 저하됨

---

## 8. 환경변수

- 모든 환경 설정은 `config.py`에서 중앙 관리.
- 하드코딩된 프로젝트 ID, 리전, 모델명 금지.
- 로컬: `.env` 파일, 배포: ECS Task Definition 환경변수.

---

## 9. 의존성 관리 규칙

- 모든 패키지는 `requirements.txt`에 **버전 고정**하여 명시한다.
- 새 패키지 추가 시 반드시 `requirements.txt`를 동시 업데이트한다.
- 그룹별로 주석 구분: LLM, ML, TFT, RAG, UI, DB
- **선택적 의존성**(모든 환경에서 불필요한 패키지)은 그룹 주석에 조건을 명시한다.
  ```
  # --- TFT 예측 모델 (FORECASTER_MODEL=tft 시 필요) ---
  torch>=2.4.0
  ```
- `pip freeze` 출력을 그대로 붙이지 않는다. 직접 사용하는 패키지만 명시 (transitive 의존성 제외).
- 버전 범위(`>=`)는 PyTorch 등 플랫폼 의존 패키지에만 허용. 그 외는 `==`으로 고정.

## 10. 파일 구조

```
Issue_Cracker/
├── app.py                  # Streamlit 대시보드
├── engine.py               # LangGraph 워크플로우 빌더
├── state.py                # PipelineState + Pydantic 스키마
├── config.py               # 환경변수 중앙 설정
├── requirements.txt        # 의존성 목록 (버전 고정)
├── AGENT.md                # 코딩 규칙 (본 문서)
├── agents/
│   ├── query_builder.py    # 사용자 입력 → 벡터 검색 쿼리 변환
│   ├── retriever.py        # pgVector 기반 데이터 검색
│   ├── analyzer.py         # 감성 분석 + NVI 산출
│   ├── forecaster/          # NVI 예측 패키지 (모델 라우터 + 서브 에이전트)
│   │   ├── __init__.py      # ForecasterAgent export
│   │   ├── forecaster.py    # 루트 컨트롤러 (state["forecaster_model"]로 라우팅)
│   │   ├── lightgbm.py      # LightGBM + SCCT 감쇠 (기본)
│   │   ├── tft.py           # TFT (Direct multi-horizon + quantile)
│   │   ├── moirai.py        # MOIRAI 2.0 (zero-shot foundation model)
│   │   └── arima.py         # SARIMAX (통계적 벤치마크)
│   ├── reviewer.py         # CCO 가이드라인 검토
│   └── reporter/
│       ├── planner.py      # 기획 지시
│       ├── strategist.py   # RAG 기반 전략 수립
│       ├── analyst.py      # Gap 분석
│       └── compiler.py     # 최종 보고서 취합
├── scripts/
│   ├── train_model.py      # LightGBM 학습
│   ├── train_tft.py        # TFT 학습 (data/cases/*.csv → models/tft_nvi.ckpt)
│   └── generate_embeddings.py  # DB 데이터 임베딩 배치 생성
├── manual/
│   └── seed_db.py          # DB 시드 데이터 생성
├── sql/
│   └── 001_create_tables.sql
├── models/
│   ├── nvi_forecaster.pkl  # LightGBM 학습된 모델
│   └── tft_nvi.ckpt        # TFT 학습된 체크포인트
└── data/
    ├── pr_crisis_dataset.csv
    ├── input_crisis_72h.csv
    └── cases/              # TFT 학습용 위기 사례 CSV들
```
