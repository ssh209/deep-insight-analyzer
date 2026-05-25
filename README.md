# 🚨 PR Crisis Simulation Dashboard

**분산형 멀티 에이전트 워크플로우 기반 실시간 여론(NVI) 시뮬레이션 및 위기 대응 보고서 자동 생성 시스템**

> 기업 PR 위기 상황을 입력하면, 7개의 전문 AI 에이전트가 병렬·순차 파이프라인으로 협업하여 여론 지수(NVI)를 예측하고, 시간대별 대응 전략 보고서를 자동으로 생성합니다.

---

## 📑 목차

- [개요](#-개요)
- [시스템 아키텍처](#-시스템-아키텍처)
- [에이전트 파이프라인](#-에이전트-파이프라인)
- [프로젝트 구조](#-프로젝트-구조)
- [기술 스택](#-기술-스택)
- [설치 및 실행](#-설치-및-실행)
- [사용 방법](#-사용-방법)
- [데이터셋](#-데이터셋)

---

## 📌 개요

기업에 PR 위기가 발생했을 때, **초기 대응 전략의 수립 속도**가 여론 회복의 핵심 변수입니다.  
이 시스템은 LangGraph 기반의 **멀티 에이전트 오케스트레이션**을 통해 다음을 자동화합니다:

1. **위기 상황 구조화** — 비정형 텍스트에서 대응 타임라인을 자동 추출
2. **여론 지수(NVI) 시뮬레이션** — LightGBM 모델로 향후 72시간 NVI 궤적을 동적 예측
3. **전문가급 보고서 생성** — 데이터 분석 + PR 전략을 병렬로 작성 후 단일 JSON 보고서로 취합
4. **자동 규정 검토** — RAG 기반 CCO 레드팀이 금칙어/가이드라인 위반을 필터링하고, 미통과 시 자동 재작성

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard (app.py)                    │
│         실시간 Graphviz 시각화 + 결과 대시보드 UI                      │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Engine Layer (engine.py)                        │
│        인프라 초기화 (Gemini Client, Vector DB) + 그래프 빌드           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 LangGraph StateGraph Pipeline                       │
│                                                                     │
│   START → Analyzer → Forecaster → Planner ──┬──→ Analyst    ──┐    │
│                                              └──→ Strategist ──┤    │
│                                                                ▼    │
│                    END ←── Reviewer ←────────── Compiler            │
│                             │    ▲                                   │
│                             └────┘ (반려 시 Planner로 피드백 루프)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 에이전트 파이프라인

| 순서 | 에이전트 | 역할 | 핵심 기술 |
|:---:|:---|:---|:---|
| 1 | **Analyzer** (상황 분석기) | 위기 메타 정보에서 향후 대응 타임라인 이벤트를 구조화된 JSON으로 추출 | Gemini 2.5 Pro + Structured Output |
| 2 | **Forecaster** (NVI 시뮬레이터) | 과거 데이터로 학습한 ML 모델에 시나리오 변곡점을 투영하여 72시간 NVI 예측 | LightGBM Regression |
| 3 | **Planner** (TF 기획자) | NVI 최저점과 상황을 종합해 하위 에이전트(분석가/전략가)에게 업무 지시서 작성 | Gemini 2.5 Pro |
| 4-A | **Analyst** (데이터 분석가) | NVI 하락 폭과 위험도를 정량적으로 분석한 초안 작성 | Gemini 2.5 Pro |
| 4-B | **Strategist** (PR 전략가) | RAG로 과거 성공 사례를 검색하여 시간대별 대응 액션 플랜 수립 | Gemini 2.5 Pro + RAG (ChromaDB) |
| 5 | **Compiler** (보고서 취합자) | 분석가와 전략가의 초안을 단일 규격 JSON 보고서로 융합 | Gemini 2.5 Pro + Structured Output |
| 6 | **Reviewer** (CCO 레드팀) | RAG 기반 금칙어 필터링 및 가이드라인 준수 여부 검토. 위반 시 반려 피드백 | Gemini 2.5 Pro + RAG (ChromaDB) |

### 워크플로우 특징

- **Fan-out / Fan-in 병렬 처리**: Planner 이후 Analyst와 Strategist가 **동시 병렬 실행**되어 처리 시간 단축
- **조건부 피드백 루프**: Reviewer가 반려하면 Planner 단계로 되돌아가 재작성 (최대 3회)
- **RAG 기반 규정 준수**: ChromaDB 벡터 스토어에 저장된 사내 가이드라인으로 일관된 품질 보장

---

## 📁 프로젝트 구조

```
Issue-Analayzer/
├── app.py                          # Streamlit 대시보드 UI (메인 실행)
├── main.py                         # CLI 통합 테스트 엔트리포인트
├── engine.py                       # 인프라 초기화 + LangGraph 워크플로우 빌드
├── state.py                        # 공유 상태(PipelineState) 및 스키마 정의
│
├── agents/                         # 에이전트 모듈
│   ├── analyzer.py                 # Agent 1: 상황 분석 및 타임라인 추출
│   ├── forecaster.py               # Agent 2: LightGBM 기반 NVI 예측
│   ├── reviewer.py                 # Agent 6: CCO 레드팀 (RAG 검토)
│   └── reporter/                   # 보고서 생성 에이전트 그룹
│       ├── planner.py              # Agent 3: TF 총괄 기획자
│       ├── analyst.py              # Agent 4-A: 데이터 분석가
│       ├── strategist.py           # Agent 4-B: PR 전략가 (RAG)
│       └── compiler.py             # Agent 5: 보고서 취합 및 JSON 포맷팅
│
├── common/                         # 공통 유틸리티
│   └── state.py                    # (레거시) 초기 상태 정의
│
├── data/                           # 데이터셋
│   └── pr_crisis_dataset.csv       # PR 위기 시뮬레이션 데이터
│
├── manual/                         # 수동 스크립트
│   └── make_dataset.py             # 합성 위기 데이터셋 생성기
│
├── requirements.txt                # Python 의존성 목록
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
cd Issue-Analayzer

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
3. 실시간으로 에이전트 워크플로우 진행 상황을 Graphviz 다이어그램으로 모니터링합니다.
   - 🟢 초록: 완료된 에이전트
   - 🔵 파랑: 현재 실행 중인 에이전트
   - 🔴 빨강: CCO 반려 (피드백 루프 발생)
4. 파이프라인 완료 후 종합 분석 결과를 대시보드에서 확인합니다:
   - **위기 경보 등급** (RED / ORANGE / YELLOW)
   - **예상 NVI 최저점**
   - **법무 및 PR 리스크 진단**
   - **시간대별 액션 플랜**

### 입력 예시

```
[현재 상황] 초기 대응 지연 및 무대응 기간 유튜버 2연타 타격으로 여론 최악 직면.
[향후 대응 계획]
- 4시간 뒤: '사실무근이며 깊은 유감이다'라는 1차 해명문 배포 (action_type: 1)
- 24시간 뒤: 전면 리콜 공표 및 대표이사 명의의 2차 대고객 사과문 발표 (action_type: 2)
```

---

## 📊 데이터셋

`data/pr_crisis_dataset.csv`는 가상의 PR 위기 시나리오 데이터를 포함합니다.

### 시뮬레이션 시나리오

| 시점 | 이벤트 |
|:---|:---|
| 2026-06-05 20:00 | 🛑 최초 이슈 발생 (커뮤니티 호소글) |
| 2026-06-06 10:00 | 💥 유튜버 A 저격 영상 업로드 (14시간 뒤) |
| 2026-06-06 16:00 | 💣 유튜버 B 확인사살 영상 업로드 (20시간 뒤) |

### 주요 피처

| 컬럼 | 설명 |
|:---|:---|
| `SNS_Mentions` | SNS 언급량 |
| `News_Articles` | 뉴스 기사 수 |
| `Influencer_Hit` | 인플루언서 타격 여부 (0/1) |
| `Victim_Claims` | 피해 호소 건수 |
| `Boycott_Mentions` | 불매 운동 언급량 |
| `Company_Action` | 기업 대응 상태 (0: 무대응, 1: 1차 입장문, 2: 2차 사과문) |
| `Actual_NVI` | 여론 지수 (0.1 ~ 1.0, 낮을수록 부정적) |

데이터셋을 재생성하려면:

```bash
python manual/make_dataset.py
```

---

## 📄 라이선스

이 프로젝트는 개인 연구 및 학습 목적으로 작성되었습니다.
