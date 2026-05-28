# CLAUDE.md — Issue Cracker 프로젝트 컨텍스트

## 프로젝트 한줄 요약

LangGraph 기반 8-에이전트 순차 파이프라인으로, 원본 텍스트 감성 분석부터 **무대응(Baseline) vs 전략 적용(Mitigated)** NVI 예측 비교, 경영진 보고서 자동 생성까지 엔드투엔드.

## NVI (Net Valence Index) — 커스텀 지표

> ⚠️ 업계 표준(NSS, NPS)이 아닌 **본 프로젝트에서 정의한 복합 여론 건강도 지표**. 범위 0.1(최악)~1.0(정상).

**산출**: `NVI(t) = clip(NVI(t-1) - Penalty + Bonus + noise, 0.1, 1.0)`
- 감점: `Negative_Ratio×0.10` + `Mockery_Index×0.15` + `Momentum>2000이면 +0.10`
- 가점: `Advocate_Ratio×0.20` + `Action1=+0.05` + `Action2=+0.15`
- 유사 지표: NSS(Net Sentiment Score)의 다차원 가중합 변형
- 산출 코드: `manual/make_dataset.py` 95~111줄

## 아키텍처

```
START → Analyzer(asyncpg + LLM 배치 감성 분류)
  → Baseline Forecaster(LightGBM, 무대응 168h 예측)
  → Planner(기획 지시)
  → Strategist(RAG + Structured Output → 1주 이내 타임라인 JSON + 드래프트)
  → Mitigated Forecaster(LightGBM, 전략 적용 168h 재예측)
  → Analyst(무대응 vs 전략 Gap 분석)
  → Compiler(JSON 보고서 취합)
  → Reviewer(RAG 금칙어 필터링)
  → END (반려 시 Planner로 복귀, 최대 3회)
```

- **완전 순차 구조** — 데이터 인과관계 기반 (Forecaster 2회 실행)
- **db_pool=None이면** Analyzer 생략 → 기존 호환 유지

## 핵심 파일

| 파일 | 역할 |
|:---|:---|
| `engine.py` | 인프라 초기화(Gemini Client, ChromaDB, asyncpg Pool) + LangGraph 그래프 빌드 |
| `state.py` | `PipelineState`(TypedDict) 공유 상태 + Pydantic 스키마 |
| `app.py` | Streamlit 대시보드 — Graphviz + Agent I/O 로그 |
| `main.py` | CLI 테스트 엔트리포인트 |
| `agents/analyzer.py` | **감성 분석기** — asyncpg + LLM 배치(50건×5병렬) + 3회 재시도 + analysis_results INSERT + MV→CSV |
| `agents/forecaster.py` | LightGBM NVI 168h 예측 — 사전 학습 모델 로드 + SCCT 감쇠 |
| `agents/reporter/strategist.py` | RAG 기반 1주 이내 대응 타임라인(JSON) + 전략 리포트 |
| `agents/reporter/planner.py` | Baseline 최저점 참조 업무 지시서 |
| `agents/reporter/analyst.py` | Baseline vs Mitigated Gap 정량 분석 |
| `agents/reporter/compiler.py` | CrisisReport JSON 취합 (Structured Output) |
| `agents/reviewer.py` | RAG 금칙어 필터링 + ReviewResult |
| `sql/001_create_tables.sql` | 전체 DDL (crises/posts/comments/analysis_results/hourly_snapshots MV) |
| `manual/seed_db.py` | DB 시드 데이터 생성기 (72h 시나리오, 11개 포스트, ~300건 댓글) |
| `models/nvi_forecaster.pkl` | 사전 학습 LightGBM (720h 기반) |
| `data/pr_crisis_dataset.csv` | 720h 학습 데이터 (5-페이즈) |
| `data/input_crisis_72h.csv` | 72h 실전 입력 데이터 |
| `scripts/train_model.py` | LightGBM 사전 학습 & pkl 저장 |

## PipelineState (공유 상태 버스)

```python
class PipelineState(TypedDict):
    crisis_id: str                      # 위기 건 식별자 (DB 조회용)
    train_csv_path: str                 # 학습 데이터 경로 (720h)
    input_csv_path: str                 # 실전 입력 데이터 (analyzer가 생성 또는 수동 지정)
    crisis_context: str                 # 위기 상황 텍스트
    crisis_type: str                    # SCCT 위기 유형 (victim|accidental|preventable)
    actual_nvi_history: list            # 과거 실제 NVI (차트용)
    nvi_baseline_forecast: list         # 무대응 시 168h NVI 예측
    nvi_mitigated_forecast: list        # 전략 적용 시 168h NVI 예측
    strategist_timeline: list           # 1주 이내 대응 타임라인 (hour_offset 0~168)
    strategist_draft: str               # 전략가 서술형 리포트
    planner_instructions: str           # 기획 지시서
    analyst_draft: str                  # Gap 분석 리포트
    draft_report: str                   # 최종 JSON 리포트
    review_feedback: str                # 반려 피드백
    is_approved: bool                   # 승인 여부
    loop_count: int                     # 루프 횟수
```

## 기술 스택

- **LLM**: Gemini 2.5 Pro (Vertex AI, `google-genai`)
- **오케스트레이션**: LangGraph `StateGraph` (async 지원)
- **ML 예측**: LightGBM + joblib (사전 학습 pkl)
- **DB**: PostgreSQL + asyncpg (crises/posts/comments/analysis_results/hourly_snapshots)
- **배치 분석**: Vertex AI Batch Prediction (초기 대량용, 50% 할인)
- **RAG**: ChromaDB + HuggingFace `all-MiniLM-L6-v2`
- **UI**: Streamlit + Graphviz
- **스키마**: Pydantic v2
- **GCP 프로젝트**: `deep-insight-496705` (engine.py에 하드코딩)

## Dynamic Decay — SCCT 위기 유형별 파라미터화

`PipelineState.crisis_type`에 따라 `CRISIS_DECAY_PARAMS` 프리셋이 선택됨 (Coombs 2007 기반).
파라미터 정의: `agents/forecaster.py` 상단.

| 파라미터 | victim (빠른 감쇠) | accidental (보통) | preventable (느린 감쇠) |
|:---|:---:|:---:|:---:|
| `momentum_decay` | 0.45 | 0.60 | 0.75 |
| `neg_ratio_decay` | 0.98 | 0.995 | 0.999 |
| `mockery_decay` | 0.95 | 0.99 | 0.998 |
| `advocate_growth` | 1.01 | 1.002 | 1.0005 |
| `action_2_neg` | 0.90 | 0.95 | 0.98 |
| `action_2_advocate` | 1.08 | 1.05 | 1.02 |

UI: 사이드바 `st.selectbox("⚠️ 위기 유형 (SCCT)")` → `crisis_type` → ForecasterAgent에 전달

## 실행 방법

```bash
# DB 스키마 생성
psql -f sql/001_create_tables.sql

# 시드 데이터
python manual/seed_db.py --dsn postgresql://...

# 대시보드
streamlit run app.py

# CLI
python main.py
```

## DB 스키마 구조

```
crises (마스터)            posts (원문)            comments (댓글)
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ crisis_id (PK)   │───│ crisis_id (FK)   │   │ crisis_id (FK)   │
│ title            │   │ post_id (PK)     │───│ post_id (FK)     │
│ description      │   │ platform         │   │ comment_id (PK)  │
│ crisis_type      │   │ title, body      │   │ body             │
│ status           │   │ author, views    │   │ like_count       │
└──────────────────┘   └──────────────────┘   └──────────────────┘
                                  │                     │
                                  └──────────┬──────────┘
                                           │
                              analysis_results (통합 분석)
                              ┌────────────────────────┐
                              │ target_type (post|comment) │
                              │ target_id                  │
                              │ sentiment, score            │
                              │ is_mockery, is_advocate     │
                              │ influence_score             │
                              │ model_version               │
                              │ UNIQUE(type, id, version)   │
                              └────────────────────────┘
                                           │
                              hourly_snapshots (MV, LEFT JOIN)
```

- 원본 데이터(posts/comments)와 분석 결과(analysis_results)가 완전 분리
- 동일 댓글을 다른 모델로 재분석해도 `UNIQUE(target_type, target_id, model_version)`로 공존
- 분석 실패 댓글은 analysis_results에 없음 → MV의 `LEFT JOIN`으로 `total_mentions`에만 포함

## 주의사항

- RAG 벡터 DB는 하드코딩된 2개 더미 Document만 포함
- GCP 프로젝트 ID가 engine.py에 하드코딩되어 있음
- `models/nvi_forecaster.pkl` 변경 시 `scripts/train_model.py` 재실행 필요
- AnalyzerAgent는 `db_pool`이 없으면 생략됨 (기존 CSV 직접 입력 호환)
- 감성 분석 3회 재시도 후 실패분은 analysis_results에 미등록 (언급량에만 포함)
- `ON CONFLICT DO NOTHING`으로 먱등성 보장 (같은 모델로 재실행해도 중복 없음)
