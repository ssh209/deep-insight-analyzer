# 🚨 Issue Cracker — PR Crisis Simulation Dashboard

**순차 인과관계 기반 멀티 에이전트 파이프라인으로 무대응(Do Nothing) vs 전략 적용(Mitigated) NVI 시뮬레이션 및 위기 대응 보고서를 자동 생성하는 시스템**

> 기업 PR 위기 상황을 입력하면, 7개의 전문 AI 에이전트가 순차 파이프라인으로 협업하여 **무대응 시 여론 최저점**과 **전략 적용 시 방어 효과(ROI)**를 비교 분석하고, 경영진 의사결정을 위한 보고서를 자동 생성합니다.

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

1. **무대응 NVI 예측** — 아무 대응도 하지 않았을 때 향후 72시간 여론 지수 궤적을 LightGBM으로 시뮬레이션
2. **전략 수립** — RAG 기반 과거 성공 사례를 참고하여 시간대별 대응 타임라인을 구조화된 JSON으로 도출
3. **전략 적용 NVI 재예측** — 수립된 전략을 시계열에 주입하여 방어된 NVI 궤적을 재시뮬레이션
4. **Gap 분석 보고서** — 무대응 vs 전략 적용 간 정량적 차이(방어 효과, ROI)를 분석하여 경영진 보고서 생성
5. **자동 규정 검토** — RAG 기반 CCO 레드팀이 금칙어/가이드라인 위반을 필터링하고, 미통과 시 자동 재작성

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
│      인프라 초기화 (Gemini Client, Vector DB) + 그래프 빌드          │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              LangGraph StateGraph Pipeline (순차 인과관계)          │
│                                                                    │
│  START → Baseline Forecaster → Planner → Strategist               │
│              → Mitigated Forecaster → Analyst → Compiler           │
│                  → Reviewer → END                                  │
│                      │    ▲                                        │
│                      └────┘ (반려 시 Planner로 피드백 루프, 최대 3회) │
└──────────────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙: 데이터 인과관계 기반 순차 파이프라인

기존의 Fan-out/Fan-in 병렬 구조 대신, **데이터 의존성(Data Dependency)에 따른 순차 실행**을 채택했습니다.
Forecaster가 두 번 실행(무대응 → 전략 적용)되어야 하므로, 전략가의 산출물이 2차 예측의 입력이 되는 인과관계가 필수적입니다.

---

## 🤖 에이전트 파이프라인

| 순서 | 에이전트 | 역할 | 핵심 기술 |
|:---:|:---|:---|:---|
| 1 | **Baseline Forecaster** (무대응 예측) | 과거 72시간 데이터로 학습 후, 무대응(action=0) 시 향후 72시간 NVI 예측 | LightGBM + Dynamic Decay |
| 2 | **Planner** (TF 기획자) | 무대응 최저점을 확인하고 전략가/분석가에게 업무 지시서 작성 | Gemini 2.5 Pro |
| 3 | **Strategist** (PR 전략가) | RAG 기반 과거 사례 참조 → 구조화된 대응 타임라인(JSON) + 서술형 전략 리포트 이중 출력 | Gemini 2.5 Pro + RAG + Structured Output |
| 4 | **Mitigated Forecaster** (전략 적용 예측) | 전략가가 도출한 액션 타임라인을 시계열에 주입하여 방어된 NVI 재예측 | LightGBM + Dynamic Decay |
| 5 | **Analyst** (데이터 분석가) | 무대응 vs 전략 적용 NVI를 비교하여 방어 효과(Gap/ROI) 정량 분석 | Gemini 2.5 Pro |
| 6 | **Compiler** (보고서 취합자) | 분석가와 전략가의 초안을 단일 규격 JSON 보고서로 융합 | Gemini 2.5 Pro + Structured Output |
| 7 | **Reviewer** (CCO 레드팀) | RAG 기반 금칙어 필터링 및 가이드라인 준수 검토. 위반 시 반려 피드백 | Gemini 2.5 Pro + RAG |

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

---

## 📁 프로젝트 구조

```
Issue_Cracker/
├── app.py                          # Streamlit 대시보드 UI (메인 실행)
├── main.py                         # CLI 통합 테스트 엔트리포인트
├── engine.py                       # 인프라 초기화 + LangGraph 워크플로우 빌드
├── state.py                        # PipelineState 및 Pydantic 스키마 정의
│
├── agents/                         # 에이전트 모듈
│   ├── forecaster.py               # Agent 1/4: LightGBM NVI 예측 (baseline/mitigated)
│   ├── reviewer.py                 # Agent 7: CCO 레드팀 (RAG 검토)
│   └── reporter/                   # 보고서 생성 에이전트 그룹
│       ├── planner.py              # Agent 2: TF 총괄 기획자
│       ├── strategist.py           # Agent 3: PR 전략가 (RAG + Structured Output)
│       ├── analyst.py              # Agent 5: 데이터 분석가 (Gap 분석)
│       └── compiler.py             # Agent 6: 보고서 취합 및 JSON 포맷팅
│
├── data/                           # 데이터셋
│   └── pr_crisis_dataset.csv       # PR 위기 시뮬레이션 데이터 (72h × 14 features)
│
├── manual/                         # 수동 스크립트
│   └── make_dataset.py             # 합성 위기 데이터셋 생성기
│
├── requirements.txt                # Python 의존성 목록
├── CLAUDE.md                       # AI 어시스턴트용 프로젝트 컨텍스트
└── .gitignore
```

---

## 🛠️ 기술 스택

| 카테고리 | 기술 | 버전 |
|:---|:---|:---|
| **LLM** | Google Gemini 2.5 Pro (Vertex AI) | `google-genai 2.6.0` |
| **에이전트 오케스트레이션** | LangGraph | `1.2.1` |
| **벡터 DB (RAG)** | ChromaDB + HuggingFace Embeddings | `chromadb 1.5.9` |
| **임베딩 모델** | `all-MiniLM-L6-v2` | `sentence-transformers 5.5.1` |
| **ML 예측** | LightGBM | `4.6.0` |
| **대시보드 UI** | Streamlit | `1.57.0` |
| **시각화** | Graphviz | `0.21` |
| **데이터 처리** | Pandas / NumPy | `3.0.3` / `2.4.6` |
| **스키마 검증** | Pydantic | `2.13.4` |

---

## 🚀 설치 및 실행

### 사전 요구사항

- **Python** 3.10+
- **Google Cloud** 프로젝트 및 Vertex AI API 활성화
- **Graphviz** 시스템 설치 ([다운로드](https://graphviz.org/download/))

### 1. 프로젝트 클론 및 가상환경 설정

```bash
git clone <repository-url>
cd Issue_Cracker

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. Google Cloud 인증 설정

```bash
gcloud auth application-default login
```

> ⚠️ `engine.py`의 `project` 파라미터를 본인의 GCP 프로젝트 ID로 변경하세요.

### 4-A. 대시보드 실행 (Streamlit UI)

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속합니다.

### 4-B. CLI 모드 실행 (터미널)

```bash
python main.py
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

`data/pr_crisis_dataset.csv`는 가상의 PR 위기 시나리오 데이터를 포함합니다.

### 시뮬레이션 시나리오

| 시점 | 이벤트 |
|:---|:---|
| T+0h | 🛑 최초 이슈 발생 (커뮤니티 호소글) |
| T+10h | 💥 대형 유튜버 A 저격 영상 업로드 |
| T+25h | 💣 중소형 유튜버 B 확산 영상 업로드 |
| T+30h | 📝 회사 1차 해명문 배포 |
| T+48h | 🙇 회사 전면 사과 및 리콜 공표 |

### 주요 피처 (14개 컬럼)

| 컬럼 | 설명 |
|:---|:---|
| `Datetime` | 시계열 타임스탬프 |
| `Hours_Since_Start` | 이슈 발생 이후 경과 시간 |
| `Company_Action_Type` | 기업 대응 상태 (0: 무대응, 1: 1차 입장문, 2: 2차 사과문) |
| `Influencer_Impact` | 인플루언서 타격 강도 (0/1/2) |
| `Raw_Total_Mentions` | 총 SNS 언급량 |
| `Raw_Negative_Mentions` | 부정 언급량 |
| `Raw_Mockery_Mentions` | 조롱 언급량 |
| `Raw_Advocate_Mentions` | 옹호 언급량 |
| `Negative_Ratio` | 부정 비율 (파생 지표) |
| `Mockery_Index` | 조롱 지수 (파생 지표) |
| `Advocate_Ratio` | 옹호 비율 (파생 지표) |
| `SNS_Mentions_Velocity` | 언급량 변화 속도 (파생 지표) |
| `Negative_Momentum` | 부정 모멘텀 (파생 지표) |
| `Actual_NVI` | 여론 지수 (0.1~1.0, 낮을수록 부정적) |

데이터셋을 재생성하려면:

```bash
python manual/make_dataset.py
```

---

## 📄 라이선스

이 프로젝트는 개인 연구 및 학습 목적으로 작성되었습니다.
