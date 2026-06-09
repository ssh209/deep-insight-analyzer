# 🚨 Deep Insight Analyzer — PR Crisis Simulation Dashboard

**Collector DB 기반 멀티 에이전트 파이프라인으로 무대응(Do Nothing) vs 전략 적용(Mitigated) NVI 시뮬레이션 및 위기 대응 보고서를 자동 생성하는 시스템**

> 기업 PR 위기 상황(issue)을 입력하면, 10개의 전문 AI 에이전트가 순차 파이프라인으로 협업하여 **Collector가 수집한 원본 데이터**에서 벡터 검색 → 감성 분석 → **무대응 시 여론 최저점** → **전략 적용 시 방어 효과(ROI)** 비교 분석 → 경영진 보고서 자동 생성까지 엔드투엔드로 처리합니다.

---

## 📑 목차

- [개요](#-개요)
- [NVI 지표 정의](#-nvi-지표-정의)
- [시스템 아키텍처](#-시스템-아키텍처)
- [에이전트 파이프라인](#-에이전트-파이프라인)
- [프로젝트 구조](#-프로젝트-구조)
- [기술 스택](#-기술-스택)
- [설치 및 실행](#-설치-및-실행)
- [사용 방법](#-사용-방법)
- [데이터셋](#-데이터셋)

---

## 📌 개요

기업에 PR 위기가 발생했을 때, 경영진(C-Level)이 가장 궁금해하는 것은 두 가지입니다:

1. **"가만히 있으면 어떻게 되는데?"** (Do Nothing — Baseline)
2. **"이 전략대로 하면 얼마나 방어할 수 있는데?"** (Mitigated)

이 시스템은 LangGraph 기반 **순차 인과관계 파이프라인**으로 다음을 자동화합니다:

1. **이슈 정의 → 검색 쿼리 생성** — 사용자의 `user_input`을 분석하여 벡터 검색용 임베딩 쿼리를 자동 생성
2. **Collector DB 벡터 검색** — `deep_insight.collected_doc_embedding`에서 pgVector 코사인 유사도 검색으로 관련 문서·댓글 확보
3. **원본 텍스트 감성 분석** — 검색된 문서의 댓글에 대해 LLM 배치 감성 분류 (Gemini Structured Output, 50건×5 병렬)
4. **무대응 NVI 예측** — 사전 학습된 LightGBM 모델로 향후 1주(168시간) 여론 궤적을 시뮬레이션
5. **전략 수립** — RAG 기반 과거 성공 사례를 참고하여 1주 이내 대응 타임라인을 구조화된 JSON으로 도출
6. **전략 적용 NVI 재예측** — 수립된 전략을 시계열에 주입하여 방어된 NVI 궤적을 재시뮬레이션
7. **Gap 분석 보고서** — 무대응 vs 전략 적용 간 정량적 차이(방어 효과, ROI)를 분석하여 경영진 보고서 생성
8. **자동 규정 검토** — RAG 기반 CCO 레드팀이 금칙어/가이드라인 위반을 필터링하고, 미통과 시 자동 재작성

---

## 📐 NVI 지표 정의

**NVI (Net Valence Index)**는 본 프로젝트에서 정의한 **복합 여론 건강도 지표**입니다.

> ⚠️ NVI는 업계 표준 지표(NSS, NPS 등)가 아닌, 위기 시뮬레이션을 위해 설계된 **커스텀 복합 지표**입니다.

### 개념

| 항목 | 내용 |
|:---|:---|
| **정의** | 특정 시점에서 기업에 대한 종합적 여론 건강도를 0.1~1.0으로 정규화한 수치 |
| **1.0** | 정상 상태 (위기 발생 전) |
| **0.1** | 최악 상태 (여론 완전 이탈, 하한 클리핑) |
| **유사 지표** | Net Sentiment Score (NSS)의 변형. NSS가 (긍정% - 부정%)의 단순 차이인 반면, NVI는 조롱·모멘텀 등 다차원 피처를 가중합으로 반영 |

### 산출 공식

매 시간(t) NVI는 이전 시점 값에서 **감점 요인**을 빼고 **가점 요인**을 더하여 갱신됩니다:

```
NVI(t) = clip( NVI(t-1) - Penalty(t) + Bonus(t) + noise,  0.1,  1.0 )
```

| 구분 | 요인 | 가중치 | 설명 |
|:---|:---|:---:|:---|
| **감점** | `Negative_Ratio` | ×0.10 | 전체 언급 중 부정 비율 |
| **감점** | `Mockery_Index` | ×0.15 | 전체 언급 중 조롱 비율 (밈화 강도) |
| **감점** | `Momentum Spike` | +0.10 | 부정 모멘텀 > 2,000일 때 추가 패널티 |
| **가점** | `Advocate_Ratio` | ×0.20 | 전체 언급 중 옹호 비율 |
| **가점** | `Action Type 1` | +0.05 | 1차 해명문 배포 시 |
| **가점** | `Action Type 2` | +0.15 | 전면 사과/리콜 공표 시 |

### 업계 표준 대비 포지셔닝

| 지표 | 데이터 소스 | 계산법 | 용도 |
|:---|:---|:---|:---|
| **NSS** (Net Sentiment Score) | 소셜 미디어 (비요청형) | 긍정% - 부정% | 실시간 감성 모니터링 |
| **NPS** (Net Promoter Score) | 설문 조사 (요청형) | 추천자% - 비추천자% | 장기 고객 충성도 |
| **NVI** (본 프로젝트) | 합성 시뮬레이션 | 다차원 감성 가중합 | 위기 시뮬레이션 및 전략 ROI 비교 |

---

## 🏗️ 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard (app.py)                     │
│    실시간 Graphviz + 리스트 전이 UI + Agent I/O 로그 (사이드바)      │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Engine Layer (engine.py)                        │
│    인프라 초기화 (Gemini Client, Vector DB, asyncpg Pool)          │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              LangGraph StateGraph Pipeline (순차 인과관계)          │
│                                                                    │
│  DB 모드:                                                          │
│  START → QueryBuilder → Retriever → Analyzer (감성 분류)          │
│       → Baseline Forecaster → Planner → Strategist                │
│       → Mitigated Forecaster → Analyst → Compiler → Reviewer      │
│                      │    ▲                           → END        │
│                      └────┘ (반려 시 Planner로 피드백 루프, 최대 3회) │
│                                                                    │
│  CSV 모드 (DB 없이):                                                │
│  START → Baseline Forecaster → Planner → ... → END                │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                Data Layer (PostgreSQL — deep_insight 스키마)       │
│                                                                    │
│  Collector 테이블 (읽기 전용):                                      │
│    collected_doc / collected_doc_comment / collected_doc_embedding │
│                                                                    │
│  Analyzer 전용 테이블:                                              │
│    issues / analysis_results / pipeline_runs                      │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│               Pre-trained Model (models/nvi_forecaster.pkl)       │
│    720h 학습 데이터 → LightGBM (1회 학습, 매번 로드)                 │
└──────────────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙

#### 1. Collector/Analyzer 역할 분리

- **deep-insight-collector** (별도 프로젝트): 외부 소스에서 문서/댓글을 수집하여 `deep_insight` 스키마에 저장
- **deep-insight-analyzer** (본 프로젝트): Collector가 수집한 데이터를 **읽기 전용으로 소비**하여 분석 결과를 같은 스키마에 저장
- 두 프로젝트는 **동일 DB, 동일 스키마(`deep_insight`)**를 공유하되 테이블 소유권이 분리됨

#### 2. 데이터 인과관계 기반 순차 파이프라인

기존의 Fan-out/Fan-in 병렬 구조 대신, **데이터 의존성(Data Dependency)에 따른 순차 실행**을 채택했습니다.
Forecaster가 두 번 실행(무대응 → 전략 적용)되어야 하므로, 전략가의 산출물이 2차 예측의 입력이 되는 인과관계가 필수적입니다.

---

## 🤖 에이전트 파이프라인

| 순서 | 에이전트 | 역할 | 핵심 기술 |
|:---:|:---|:---|:---|
| 0 | **QueryBuilder** (쿼리 생성기) | 사용자 `user_input`에서 검색 키워드 추출 → 벡터 검색용 임베딩 쿼리 생성 | Gemini + HuggingFace Embeddings |
| 1 | **Retriever** (벡터 검색기) | Collector DB에서 pgVector 코사인 유사도 검색 → 관련 문서+댓글 확보 | asyncpg + pgVector |
| 2 | **Analyzer** (감성 분석기) | 검색된 댓글 LLM 배치 감성 분류 + 영향력 스코어링 + 시계열 피처 CSV 생성 | Gemini Structured Output + asyncpg |
| 3 | **Baseline Forecaster** (무대응 예측) | 사전 학습 모델로 무대응(action=0) 시 향후 1주(168h) NVI 예측 | LightGBM + Dynamic Decay |
| 4 | **Planner** (TF 기획자) | 무대응 최저점을 확인하고 전략가/분석가에게 업무 지시서 작성 | Gemini 2.5 Pro |
| 5 | **Strategist** (PR 전략가) | RAG 기반 과거 사례 참조 → 1주 이내 대응 타임라인(JSON) + 전략 리포트 이중 출력 | Gemini 2.5 Pro + RAG + Structured Output |
| 6 | **Mitigated Forecaster** (전략 적용 예측) | 전략가가 도출한 액션 타임라인을 시계열에 주입하여 방어된 NVI 재예측 | LightGBM + Dynamic Decay |
| 7 | **Analyst** (데이터 분석가) | 무대응 vs 전략 적용 NVI를 비교하여 방어 효과(Gap/ROI) 정량 분석 | Gemini 2.5 Pro |
| 8 | **Compiler** (보고서 취합자) | 분석가와 전략가의 초안을 단일 규격 JSON 보고서로 융합 | Gemini 2.5 Pro + Structured Output |
| 9 | **Reviewer** (CCO 레드팀) | RAG 기반 금칙어 필터링 및 가이드라인 준수 검토. 위반 시 반려 피드백 | Gemini 2.5 Pro + RAG |

### Dynamic Decay — SCCT 위기 유형별 파라미터화

Forecaster는 미래 감성 지표에 **동적 감쇠(Dynamic Decay)**를 적용합니다.
감쇠 속도는 **SCCT 위기 유형**([Coombs, 2007](https://doi.org/10.1057/palgrave.crr.1550049))에 따라 차등 적용됩니다:

| 위기 유형 | 책임 귀인 | 감쇠 속도 | 사과 효과 | 예시 |
|:---|:---:|:---:|:---:|:---|
| **Victim** (피해자형) | 낮음 | 🟢 빠름 | 강력 | 자연재해, 루머, 외부 범행 |
| **Accidental** (사고형) | 보통 | 🟡 보통 | 보통 | 리콜, 기술적 결함, 장비 고장 |
| **Preventable** (예방가능형) | 높음 | 🔴 느림 | 미미 | 경영진 비리, 안전 위반, 의도적 은폐 |

#### 핵심 파라미터 비교

| 파라미터 | Victim | Accidental | Preventable |
|:---|:---:|:---:|:---:|
| 모멘텀 감쇠 (`momentum_decay`) | ×0.45 | ×0.60 | ×0.75 |
| 부정 비율 자연 감쇠 (`neg_ratio_decay`) | ×0.98 | ×0.995 | ×0.999 |
| 조롱 지수 자연 감쇠 (`mockery_decay`) | ×0.95 | ×0.99 | ×0.998 |
| 사과문 부정 감소 (`action_2_neg`) | ×0.90 | ×0.95 | ×0.98 |
| 사과문 옹호 증가 (`action_2_advocate`) | ×1.08 | ×1.05 | ×1.02 |
| 부정 비율 하한선 (`neg_ratio_floor`) | 0.25 | 0.30 | **0.40** |

> 💡 Preventable(비리/은폐) 유형에서는 도덕적 분노(Moral Outrage)로 인해 사과해도 부정 비율이 0.40 아래로 내려가지 않는 **Sticky Crisis** 현상을 재현합니다 ([Antonetti & Maklan, 2016](https://doi.org/10.1007/s10551-014-2487-y)).

대시보드 사이드바의 **⚠️ 위기 유형 (SCCT)** 드롭다운에서 유형을 선택하면, 해당 프리셋이 Baseline/Mitigated 양쪽 Forecaster에 동시 적용됩니다.

### 워크플로우 특징

- **이원화 시뮬레이션**: 동일 Forecaster 클래스를 `mode` 파라미터로 2회 실행 (baseline/mitigated)
- **조건부 피드백 루프**: Reviewer가 반려하면 Planner → Strategist → Mitigated Forecaster → Analyst → Compiler → Reviewer 전체 재실행 (최대 3회)
- **RAG 기반 규정 준수**: ChromaDB 벡터 스토어에 저장된 사내 가이드라인으로 일관된 품질 보장
- **듀얼 모드**: DB 연결 시 전체 파이프라인 실행, DB 없으면 CSV 모드로 Forecaster부터 시작

---

## 📁 프로젝트 구조

```
deep-insight-analyzer/
├── app.py                          # Streamlit 대시보드 UI (메인 실행)
├── main.py                         # CLI 통합 테스트 엔트리포인트
├── engine.py                       # 인프라 초기화 + LangGraph 워크플로우 빌드
├── state.py                        # PipelineState 및 Pydantic 스키마 정의
├── config.py                       # 환경변수 기반 설정 (DB, GCP, 모델명 등)
├── db.py                           # asyncpg 커넥션 풀 + 스키마 체크
│
├── agents/                         # 에이전트 모듈
│   ├── query_builder.py            # Agent 0: 쿼리 생성기 (user_input → 임베딩 벡터)
│   ├── retriever.py                # Agent 1: pgVector 벡터 검색 + 댓글 로드
│   ├── analyzer.py                 # Agent 2: 감성 분석기 (LLM 배치 + 영향력 스코어링)
│   ├── forecaster/                 # Agent 3/6: LightGBM NVI 예측 (baseline/mitigated)
│   ├── reviewer.py                 # Agent 9: CCO 레드팀 (RAG 검토)
│   └── reporter/                   # 보고서 생성 에이전트 그룹
│       ├── planner.py              # Agent 4: TF 총괄 기획자
│       ├── strategist.py           # Agent 5: PR 전략가 (RAG + Structured Output)
│       ├── analyst.py              # Agent 7: 데이터 분석가 (Gap 분석)
│       └── compiler.py             # Agent 8: 보고서 취합 및 JSON 포맷팅
│
├── models/                         # 사전 학습 모델
│   └── nvi_forecaster.pkl          # LightGBM 모델 (720h 데이터 기반)
│
├── data/                           # 데이터셋
│   ├── pr_crisis_dataset.csv       # 720h(30일) 학습 데이터 (5-페이즈 V자 곡선)
│   └── input_crisis_72h.csv        # 72h 실전 입력 데이터 (위기 초기~폭발기)
│
├── sql/                            # PostgreSQL DDL
│   └── 001_create_tables.sql       # Analyzer 전용 테이블 (deep_insight 스키마)
│
├── manual/                         # 수동 스크립트
│   ├── make_dataset.py             # 합성 위기 데이터셋 생성기 (720h + 72h)
│   └── seed_db.py                  # DB 시드 데이터 생성기
│
├── scripts/                        # 유틸리티 스크립트
│   ├── train_model.py              # LightGBM 사전 학습 & pkl 저장
│   └── make_graph.py               # ForecasterAgent 단독 실행 차트 (Streamlit)
│
├── pyproject.toml                  # Poetry 의존성 및 프로젝트 설정
├── .env.example                    # 환경변수 템플릿
├── Dockerfile                      # 컨테이너 빌드
├── CLAUDE.md
└── .gitignore
```

---

## 🛠️ 기술 스택

| 카테고리 | 기술 | 용도 |
|:---|:---|:---|
| **LLM** | Google Gemini 2.5 Pro / Flash (Vertex AI) | 감성 분류, 전략 수립, 보고서 생성 |
| **에이전트 오케스트레이션** | LangGraph | 순차 인과관계 파이프라인 + 피드백 루프 |
| **벡터 DB (RAG)** | ChromaDB + HuggingFace Embeddings | 과거 사례 검색, 가이드라인 준수 검토 |
| **벡터 검색 (Retrieval)** | pgVector + asyncpg | Collector 수집 문서 유사도 검색 |
| **임베딩 모델** | `multilingual-e5-small` (검색) / `all-MiniLM-L6-v2` (RAG) | 다국어 벡터 임베딩 |
| **ML 예측** | LightGBM + joblib | NVI 시계열 예측 |
| **대시보드 UI** | Streamlit | 실시간 파이프라인 모니터링 |
| **시각화** | Graphviz | 워크플로우 다이어그램 |
| **데이터 처리** | Pandas / NumPy | 시계열 피처 가공 |
| **스키마 검증** | Pydantic | 에이전트 I/O 구조체 검증 |
| **DB** | PostgreSQL 16+ + pgVector + asyncpg | Collector/Analyzer 공유 DB |
| **Batch 분석** | Vertex AI Batch Prediction | 대규모 댓글 감성 분류 |

---

## 🚀 설치 및 실행

### 사전 요구사항

- **Python** 3.12.x
- **PostgreSQL** 16+ (pgVector 확장 필수)
- **Google Cloud** 프로젝트 및 Vertex AI API 활성화
- **Graphviz** 시스템 설치 ([다운로드](https://graphviz.org/download/))
- **deep-insight-collector** 스키마 사전 적용 (Collector의 `db_schema/schema_collection.sql`)

### 0. Poetry 설치

> Poetry가 이미 설치되어 있다면 이 단계를 건너뛰세요.

```powershell
pip install poetry
```

```bash
poetry --version   # Poetry (version 2.x.x)
```

> ⚠️ 설치 후 `poetry` 명령이 인식되지 않으면 [공식 문서](https://python-poetry.org/docs/#installation)를 참고하여 PATH를 설정하세요.

### 1. 프로젝트 클론 및 의존성 설치

```bash
git clone <repository-url>
cd deep-insight-analyzer

# 가상환경을 프로젝트 폴더 내(.venv/)에 생성하도록 설정
poetry config virtualenvs.in-project true

# 의존성 설치 (가상환경 자동 생성)
poetry install
```

> 💡 개발 의존성(pytest 등)을 제외하려면 `poetry install --without dev`를 사용하세요.

### 2. PostgreSQL + pgVector 설정

#### 2-1. DB 및 pgVector 확장 생성

```sql
-- pgVector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- deep_insight 스키마 생성 (Collector와 공유)
CREATE SCHEMA IF NOT EXISTS deep_insight;
```

#### 2-2. Collector 스키마 적용 (선행 필수)

Collector의 DDL을 먼저 실행하여 수집 테이블을 생성합니다:

```bash
# deep-insight-collector 프로젝트의 스키마 적용
psql -f <collector-project>/db_schema/schema_collection.sql
```

이 단계에서 다음 테이블이 `deep_insight` 스키마에 생성됩니다:
- `deep_insight.collected_doc` — 수집 문서 (doc_id BIGINT PK)
- `deep_insight.collected_doc_comment` — 댓글 (comment_id BIGINT PK)
- `deep_insight.collected_doc_embedding` — 임베딩 벡터 (pgVector)
- `deep_insight.collection_job` — 수집 작업 단위

#### 2-3. Analyzer 스키마 적용

```bash
# Analyzer 전용 테이블 생성
psql -f sql/001_create_tables.sql
```

이 단계에서 다음 테이블이 추가됩니다:
- `deep_insight.issues` — 분석 대상 이슈 (user_input + QueryBuilder 결과물 보관)
- `deep_insight.analysis_results` — 감성 분석 결과 (doc/comment 공통, 모델 버전별 관리)
- `deep_insight.pipeline_runs` — 파이프라인 실행 이력

#### 2-4. 환경변수 설정

```bash
# .env.example을 복사하여 수정
cp .env.example .env
```

```env
# DATABASE_URL 필수 설정 (Neon, Supabase, 또는 로컬 PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/deep_insight

# SSL 연결 (클라우드 DB 사용 시 true)
DB_SSL=true
```

> ⚠️ `DATABASE_URL`이 설정되지 않으면 CSV 모드로 폴백합니다. 이 경우 QueryBuilder → Retriever → Analyzer 단계가 생략되고, 사전 생성된 CSV 파일로 Forecaster부터 시작합니다.

### 3. Google Cloud 인증 설정

```bash
gcloud auth application-default login
```

> ⚠️ `config.py`의 `GCP_PROJECT_ID` 환경변수를 본인의 GCP 프로젝트 ID로 설정하세요.

### 4-A. 대시보드 실행 (Streamlit UI)

```bash
poetry run streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속합니다.

### 4-B. CLI 모드 실행 (터미널)

```bash
poetry run python main.py
```

---

## 💡 사용 방법

### Streamlit 대시보드

1. 좌측 사이드바에서 **데이터셋 경로**와 **위기 메타 정보**를 입력합니다.
2. **🚀 파이프라인 가동** 버튼을 클릭합니다.
3. 메인 화면에서 워크플로우 진행 상황을 모니터링합니다:
   - **플로우 차트 (좌)**: Graphviz 다이어그램으로 실시간 노드 상태 표시
   - **파이프라인 리스트 (우)**: 아이콘 전이 방식으로 진행 상태 표시
   - ⬜ 대기 → 🔵 실행 중 → ✅ 완료 / ❌ 반려
4. 사이드바 하단의 **Agent I/O 로그**에서 각 에이전트의 입출력을 확인합니다.
5. 파이프라인 완료 후 종합 분석 결과를 대시보드에서 확인합니다:
   - **무대응 시 NVI 최저점** vs **전략 적용 시 NVI 최저점**
   - **🛡️ 방어 효과** (포인트 차이)
   - **3-라인 NVI 비교 차트** (실제/무대응/전략 적용)
   - **시간대별 액션 플랜**

### 입력 예시

```
[현재 상황] 초기 대응 지연 및 무대응 기간 유튜버 2연타 타격으로 여론 최악 직면.
```

---

## 📊 데이터셋

### 이원화 데이터 구조

| 파일 | 용도 | 규모 |
|:---|:---|:---|
| `data/pr_crisis_dataset.csv` | LightGBM **학습** 데이터 | 720h (30일), 5-페이즈 V자 곡선 |
| `data/input_crisis_72h.csv` | 실전 **입력** 데이터 | 72h, 위기 초기~폭발기 |
| `models/nvi_forecaster.pkl` | 사전 학습 모델 | 1회 학습, 매번 로드 |

### 학습 데이터 시나리오 (720h / 30일)

| 페이즈 | 시간대 | NVI 범위 | 주요 이벤트 |
|:---|:---|:---|:---|
| ① 잠복기 | H0~H48 | 1.0→0.4 | 최초 이슈, 유튜버 A 저격 |
| ② 폭발기 | H48~H120 | 0.4→0.1 | 유튜버 B 확산, 여론 최악 |
| ③ 확산/바닥 | H120~H240 | 0.1 고착 | Sticky Crisis |
| ④ 교착기 | H240~H432 | 0.1~0.3 | 1차 해명, 느린 반등 |
| ⑤ 수습기 | H432~H720 | 0.3→1.0 | 전면 사과, 완전 회복 |

### DB 기반 데이터 (PostgreSQL — `deep_insight` 스키마)

#### Collector 테이블 (읽기 전용)

| 테이블 | 설명 |
|:---|:---|
| `deep_insight.collected_doc` | 수집 문서 (유튜브 영상, 뉴스 기사, 커뮤니티 글). `raw` JSONB에 원본 메타데이터 포함 |
| `deep_insight.collected_doc_comment` | 수집 문서의 댓글. `content` 컬럼에 원문 저장 |
| `deep_insight.collected_doc_embedding` | 문서별 임베딩 벡터 (`multilingual-e5-small` 모델 기준) |
| `deep_insight.collection_job` | 수집 작업 단위 (Collector 관리) |

#### Analyzer 테이블

| 테이블 | 설명 |
|:---|:---|
| `deep_insight.issues` | 분석 대상 이슈. `user_input` + QueryBuilder 생성 쿼리(`search_keywords`, `search_queries`) 보관 |
| `deep_insight.analysis_results` | 통합 분석 결과 (doc/comment 공통). `target_type` + `target_id`로 대상 구분, 모델 버전별 관리 |
| `deep_insight.pipeline_runs` | 파이프라인 실행 이력 (상태, 실행 시간, 산출물 요약) |

#### 데이터 흐름

```
Collector 수집 → collected_doc + collected_doc_embedding
                            │
              QueryBuilder 쿼리 생성 → Retriever 벡터 검색
                                          │
                          Analyzer 감성 분석 → analysis_results INSERT
                                          │
                    Forecaster가 CSV 피처로 NVI 예측
```

데이터 세팅:

```bash
# 1. Collector DB 스키마 생성 (deep-insight-collector 프로젝트)
psql -f <collector-project>/db_schema/schema_collection.sql

# 2. Analyzer 전용 스키마 생성
psql -f sql/001_create_tables.sql

# 3. 합성 학습/입력 데이터 생성
poetry run python manual/make_dataset.py

# 4. LightGBM 모델 사전 학습
poetry run python scripts/train_model.py
```

---

## 📄 라이선스

이 프로젝트는 개인 연구 및 학습 목적으로 작성되었습니다.
