# LegalAdvisor 에이전트 + 법률 RAG + 파이프라인 병렬화

> **상태**: ⏸️ HOLD
> **의존성**: 법률 코퍼스 데이터 확보 후 진행
> **예상 소요**: Phase 4 (3~4h) + Phase 5 (2~3h) = 총 5~7h

---

## 개요

| Phase | 내용 | 핵심 변경 |
|:---:|:---|:---|
| **4** | LegalAdvisor 에이전트 + 법률 RAG | 에이전트 신규, 법률 전용 ChromaDB |
| **5** | 파이프라인 병렬 구조 개편 | Strategist ∥ LegalAdvisor fan-out/fan-in |

---

## Phase 4: LegalAdvisor 에이전트

### 목표

위기 상황에서 법적 리스크를 사전 평가하고, PR 전략에 대한 법적 제약 조건을 제공합니다.

| 항목 | 설명 |
|:---|:---|
| 에이전트 | `LegalAdvisorAgent` |
| RAG DB | 법률 전용 ChromaDB collection (PR용 RAG와 분리) |
| 입력 | `crisis_context`, `crisis_type`, `sentiment_landscape` |
| 출력 | `legal_assessment` (법적 리스크 + 제약 + 권고 조치 + 유사 판례) |

### 4-1. 법률 RAG 코퍼스

#### [NEW] `data/legal_corpus/`

| 파일 | 내용 |
|:---|:---|
| `consumer_protection.md` | 소비자보호법 핵심 조문 (소비자 피해 보상, 리콜 의무 등) |
| `product_liability.md` | 제조물책임법 핵심 조문 (결함 입증, 무과실 책임 등) |
| `fair_trade.md` | 공정거래법 (허위/과대 광고, 표시 위반 등) |
| `precedent_cases.md` | 유사 판례 구조화 데이터 (사건명, 쟁점, 판결요지, 시사점) |
| `internal_guidelines.md` | 기업 내부 법무 가이드라인 (대응문 작성 시 법적 주의사항) |

### 4-2. LegalAdvisorAgent

#### [NEW] `agents/legal_advisor.py`

```python
class LegalAssessment(BaseModel):
    legal_risks: List[LegalRisk]           # 법적 리스크 (확률 x 영향도 x 법적 근거)
    constraints: List[str]                 # PR 전략에 대한 법적 제약 (금칙어/표현/약속)
    recommended_actions: List[LegalAction] # 선제적 법적 조치 (리콜 신고, 보험 등)
    precedent_cases: List[PrecedentCase]   # 유사 판례 요약

class LegalRisk(BaseModel):
    risk: str                              # 리스크명
    probability: str                       # low / medium / high
    impact: str                            # low / medium / high / critical
    legal_basis: str                       # 근거 법조문
    max_penalty: str                       # 최대 제재 수준

class LegalAction(BaseModel):
    action: str                            # 조치 내용
    urgency: str                           # immediate / within_24h / within_week
    responsible: str                       # 담당 부서

class PrecedentCase(BaseModel):
    case_name: str                         # 사건명
    court: str                             # 법원
    year: int                              # 판결 연도
    issue: str                             # 쟁점
    ruling_summary: str                    # 판결요지
    implication: str                       # 현재 위기에 대한 시사점
```

**동작 흐름:**
1. 법률 RAG에서 `crisis_context` 관련 법조문/판례 3~5건 검색
2. LLM에 위기 상황 + RAG 결과 전달 -> `LegalAssessment` Structured Output
3. Reviewer가 대응문에 법적 제약 위반이 없는지 추가 검증

### 4-3. 스키마 변경

#### [MODIFY] `state.py`

```python
# PipelineState 추가
legal_assessment: dict              # LegalAdvisor 산출물

# CrisisReport 추가
legal_assessment: Optional[dict]    # 법적 리스크/제약/권고/판례
```

### 4-4. 인프라 변경

#### [MODIFY] `engine.py`
- `init_infrastructure()`에서 법률 Vector DB 초기화 (별도 ChromaDB collection)
- `build_graph()`에서 `legal_advisor` 노드 추가
- Phase 4에서는 **직렬 연결**

```
... -> Analyst -> LegalAdvisor -> Compiler -> Reviewer -> ...
```

#### [MODIFY] `config.py`
- `LEGAL_CORPUS_DIR` 환경변수 추가

### 4-5. Compiler / Reviewer 수정

#### [MODIFY] `compiler.py`
- 프롬프트에 `legal_assessment` 요약 포함
- 최종 리포트에 직접 주입

#### [MODIFY] `reviewer.py`
- 대응문 초안(`draft_statements`)이 법적 제약(`legal_assessment.constraints`)을 위반하지 않는지 교차 검증

---

## Phase 5: 파이프라인 병렬 구조 개편

### 목표

Strategist와 LegalAdvisor를 **병렬 실행**하여 파이프라인 처리 시간을 단축합니다.

### 현재 (Phase 4 완료 후, 직렬)

```
START -> QueryBuilder -> Retriever -> Analyzer
      -> Baseline Forecaster -> Planner -> Strategist
      -> Mitigated Forecaster -> Analyst -> LegalAdvisor
      -> Compiler -> Reviewer -> END
```

### 변경 (병렬 분기)

```
START -> QueryBuilder -> Retriever -> Analyzer
      -> Baseline Forecaster -> Planner
      -> +-- Strategist -> Mitigated Forecaster -> Analyst --+
         +-- LegalAdvisor ----------------------------------+
      -> Compiler -> Reviewer -> END (or <=3회 Planner)
```

### 5-1. engine.py 변경

#### [MODIFY] `engine.py`

```python
# LangGraph fan-out / fan-in
workflow.add_edge("planner", "strategist")
workflow.add_edge("planner", "legal_advisor")      # 병렬 분기

workflow.add_edge("strategist", "mitigated_forecaster")
workflow.add_edge("mitigated_forecaster", "analyst")

# fan-in: analyst + legal_advisor 모두 완료 시 compiler 실행
workflow.add_edge("analyst", "compiler")
workflow.add_edge("legal_advisor", "compiler")     # 합류
```

### 5-2. 이점

| 항목 | 직렬 | 병렬 |
|:---|:---:|:---:|
| Strategist LLM 호출 | 5회 순차 | 5회 |
| LegalAdvisor LLM 호출 | 1회 (이후 대기) | 1회 (동시) |
| **총 지연** | **6회 순차** | **max(5, 1) = 5회** |

> LegalAdvisor는 Strategist 산출물에 의존하지 않으므로 병렬 실행 가능.
> Planner의 지시만 있으면 법률 분석을 시작할 수 있음.

---

## 통합 구현 순서

| # | Phase | 작업 | 파일 |
|:--|:---:|:---|:---|
| 1 | 4 | 법률 코퍼스 작성 | `data/legal_corpus/*.md` (5개) |
| 2 | 4 | Pydantic 스키마 추가 | `state.py` |
| 3 | 4 | LegalAdvisorAgent 구현 | `agents/legal_advisor.py` |
| 4 | 4 | config.py 수정 | `config.py` |
| 5 | 4 | engine.py 수정 (직렬 연결) | `engine.py` |
| 6 | 4 | Compiler 수정 | `agents/reporter/compiler.py` |
| 7 | 4 | Reviewer 수정 (법적 제약 교차 검증) | `agents/reviewer.py` |
| 8 | 4 | 검증: 직렬 파이프라인 동작 확인 | - |
| 9 | 5 | engine.py 수정 (직렬 -> 병렬 전환) | `engine.py` |
| 10 | 5 | 검증: 병렬 실행 + fan-in 합류 확인 | - |
| 11 | 4+5 | AGENT.md 업데이트 | `AGENT.md` |

---

## Verification Plan

### Automated Tests
```bash
# Phase 4: 직렬 파이프라인 그래프 확인
python engine.py  # ASCII 그래프에 legal_advisor 노드 확인

# Phase 5: 병렬 분기 확인
python engine.py  # planner -> strategist, planner -> legal_advisor 병렬 엣지 확인
```

### Manual Verification
- 법률 RAG 검색: "제조물책임" 등 키워드로 관련 조문 반환 확인
- LegalAssessment 출력 검증: `legal_risks`, `constraints` 필드 존재 확인
- Reviewer가 법적 제약 위반 시 반려하는지 확인
- 병렬 실행 시 Strategist와 LegalAdvisor가 동시에 실행되는지 확인
- fan-in 합류 시 두 산출물이 모두 Compiler에 전달되는지 확인
