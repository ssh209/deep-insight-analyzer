# CLAUDE.md — Issue Cracker 프로젝트 컨텍스트

## 프로젝트 한줄 요약

LangGraph 기반 7-에이전트 순차 파이프라인으로, PR 위기 시 **무대응(Baseline) vs 전략 적용(Mitigated)** NVI 예측을 비교 분석하여 경영진 의사결정용 보고서를 자동 생성하는 시스템.

## NVI (Net Valence Index) — 커스텀 지표

> ⚠️ 업계 표준(NSS, NPS)이 아닌 **본 프로젝트에서 정의한 복합 여론 건강도 지표**. 범위 0.1(최악)~1.0(정상).

**산출**: `NVI(t) = clip(NVI(t-1) - Penalty + Bonus + noise, 0.1, 1.0)`
- 감점: `Negative_Ratio×0.10` + `Mockery_Index×0.15` + `Momentum>2000이면 +0.10`
- 가점: `Advocate_Ratio×0.20` + `Action1=+0.05` + `Action2=+0.15`
- 유사 지표: NSS(Net Sentiment Score)의 다차원 가중합 변형
- 산출 코드: `manual/make_dataset.py` 95~111줄

## 아키텍처

```
START → Baseline Forecaster(LightGBM, 무대응 예측)
  → Planner(기획 지시)
  → Strategist(RAG + Structured Output → 타임라인 JSON + 드래프트)
  → Mitigated Forecaster(LightGBM, 전략 적용 재예측)
  → Analyst(무대응 vs 전략 Gap 분석)
  → Compiler(JSON 보고서 취합)
  → Reviewer(RAG 금칙어 필터링)
  → END (반려 시 Planner로 복귀, 최대 3회)
```

- **완전 순차 구조** — 데이터 인과관계 기반 (Forecaster 2회 실행)
- **병렬 없음** — 이전 버전의 Fan-out/Fan-in 구조에서 전환됨

## 핵심 파일

| 파일 | 역할 |
|:---|:---|
| `engine.py` | 인프라 초기화(Gemini Client, ChromaDB) + LangGraph 그래프 빌드 |
| `state.py` | `PipelineState`(TypedDict) 공유 상태 + Pydantic 스키마(CrisisReport, CrisisTimeline, ReviewResult) |
| `app.py` | Streamlit 대시보드 — Graphviz 플로우 차트 + 리스트 전이 UI + Agent I/O 사이드바 로그 |
| `main.py` | CLI 테스트 엔트리포인트 (engine.py 재사용) |
| `agents/forecaster.py` | LightGBM NVI 예측 — 사전 학습 모델 로드(`models/nvi_forecaster.pkl`) + SCCT 감쇠 |
| `agents/reporter/strategist.py` | RAG 기반 전략 수립 — `strategist_timeline`(JSON) + `strategist_draft`(텍스트) 이중 출력 |
| `agents/reporter/planner.py` | Baseline 최저점 참조 업무 지시서 작성 |
| `agents/reporter/analyst.py` | Baseline vs Mitigated NVI Gap 정량 분석 |
| `agents/reporter/compiler.py` | CrisisReport 스키마로 JSON 취합 (Structured Output) |
| `agents/reviewer.py` | RAG 금칙어 필터링 + ReviewResult 스키마 |
| `models/nvi_forecaster.pkl` | 사전 학습된 LightGBM 모델 (720h 데이터 기반) |
| `data/pr_crisis_dataset.csv` | 720시간(30일) × 14컨럼 학습 데이터 (5-페이즈 V자 곡선) |
| `data/input_crisis_72h.csv` | 72시간 실전 입력 데이터 (위기 초기~폭발기) |
| `manual/make_dataset.py` | 데이터셋 생성기 (720h 학습 + 72h 입력) |
| `scripts/train_model.py` | LightGBM 모델 사전 학습 & pkl 저장 |
| `scripts/make_graph.py` | ForecasterAgent 단독 실행 차트 (Streamlit) |

## PipelineState (공유 상태 버스)

```python
class PipelineState(TypedDict):
    train_csv_path: str                 # 학습 데이터 경로 (720h)
    input_csv_path: str                 # 실전 입력 데이터 경로 (72h)
    crisis_context: str                 # 위기 상황 텍스트
    crisis_type: str                    # SCCT 위기 유형 (victim|accidental|preventable)
    actual_nvi_history: list            # 과거 실제 NVI (차트용)
    nvi_baseline_forecast: list         # 무대응 시 720h NVI 예측
    nvi_mitigated_forecast: list        # 전략 적용 시 720h NVI 예측
    strategist_timeline: list           # 전략가 산출 대응 타임라인
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
- **오케스트레이션**: LangGraph `StateGraph`
- **ML 예측**: LightGBM Regression
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
# 대시보드
streamlit run app.py

# CLI
python main.py
```

## 주의사항

- RAG 벡터 DB는 하드코딩된 2개 더미 Document만 포함 (과거사례 1개, 금칙어 1개)
- GCP 프로젝트 ID가 engine.py에 하드코딩되어 있음
- LLM 호출에 대한 예외 처리 미구현
- `common/` 디렉토리는 레거시 — 현재 미사용 (삭제 예정)
