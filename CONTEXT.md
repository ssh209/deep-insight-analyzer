# Issue Cracker — 프로젝트 컨텍스트 (시스템 프롬프트)

> **이 문서를 먼저 읽어주세요.** 다른 세션에서 이 프로젝트 작업을 이어갈 때, 이 파일을 Antigravity에 첨부하거나 "이 파일을 읽고 컨텍스트를 파악해줘"라고 요청하세요.

---

## 1. 프로젝트 개요

**Issue Cracker**는 기업 PR 위기 상황에서 **무대응(Baseline) vs 전략 적용(Mitigated)** 시나리오를 시뮬레이션하는 **8-에이전트 순차 파이프라인**입니다.

- **프레임워크**: LangGraph (StateGraph)
- **LLM**: Google Gemini 2.5 Pro (Vertex AI)
- **ML**: LightGBM (NVI 시계열 예측)
- **RAG**: ChromaDB + all-MiniLM-L6-v2
- **UI**: Streamlit
- **DB**: PostgreSQL (asyncpg) — 현재는 CSV 직접 입력 모드로도 동작
- **이론적 기반**: SCCT(Situational Crisis Communication Theory, Coombs 2007)

### 핵심 지표: NVI (Net Valence Index)
- 0.0(최악) ~ 1.0(정상)의 여론 건강도 지수
- 부정비율, 조롱지수, 옹호비율로 산출
- ⚠️ **알려진 이슈**: NVI 가중치가 `analyzer.py`, `make_dataset.py`, `README.md` 3곳에서 불일치. 통합 필요.

### 파이프라인 흐름
```
START → [Analyzer(선택)] → Baseline Forecaster → Planner → Strategist(RAG)
  → Mitigated Forecaster → Analyst → Compiler → Reviewer(CCO)
  → 승인 시 END / 반려 시(≤3회) Planner로 복귀
```

---

## 2. 프로젝트 구조

```
Issue_Cracker/
├── app.py                  # Streamlit 대시보드 (368줄)
├── engine.py               # LangGraph 워크플로우 빌더 (87줄)
├── state.py                # PipelineState(TypedDict) + Pydantic 스키마
├── config.py               # ★ 환경변수 중앙 설정
├── db.py                   # ★ asyncpg DB 커넥션 풀 관리 (신규)
├── main.py                 # CLI 테스트 엔트리포인트
│
├── agents/
│   ├── analyzer.py         # DB 댓글 감성 분석 + NVI 산출 (503줄, 가장 복잡)
│   ├── forecaster.py       # LightGBM + SCCT Dynamic Decay (221줄)
│   ├── reviewer.py         # CCO 가이드라인 검토 + 피드백 루프
│   └── reporter/
│       ├── planner.py      # 기획 지시
│       ├── strategist.py   # RAG 기반 전략 수립 (이중 출력: JSON + 서술형)
│       ├── analyst.py      # Baseline vs Mitigated Gap 분석
│       └── compiler.py     # CrisisReport 최종 취합
│
├── models/
│   └── nvi_forecaster.pkl  # 사전 학습된 LightGBM 모델 (575KB)
│
├── data/
│   ├── pr_crisis_dataset.csv    # 720h 학습용 합성 데이터
│   └── input_crisis_72h.csv     # 72h 실전 입력 데이터
│
├── sql/
│   └── 001_create_tables.sql    # PostgreSQL DDL + Materialized View
│
├── manual/
│   ├── make_dataset.py     # 합성 데이터 생성 스크립트
│   └── seed_db.py          # DB 시드 데이터 투입
│
├── scripts/
│   ├── train_model.py      # LightGBM 학습 스크립트
│   └── make_graph.py       # 파이프라인 시각화
│
├── deploy/                 # ★ AWS Fargate 배포 (신규 생성)
│   ├── setup-aws.sh        # 인프라 초기 구성 (원클릭)
│   └── deploy.sh           # 이미지 빌드 & 배포 자동화
│
├── Dockerfile              # ★ python:3.11-slim 기반 (신규 생성)
├── .dockerignore            # ★ (신규 생성)
├── .streamlit/config.toml   # ★ ALB 뒤 Streamlit 설정 (신규 생성)
├── .env.example             # ★ 환경변수 템플릿 (신규 생성)
├── requirements.txt         # 15개 패키지
├── README.md
└── CLAUDE.md
```

---

## 3. 완료된 작업

### 3-1. 프로젝트 분석 (완료)
- 전체 코드베이스 포렌식 분석 완료
- 10개 취약점 식별 (NVI 불일치, GCP 하드코딩, 테스트 부재 등)
- 분석 보고서 작성 완료

### 3-2. AWS Fargate 배포 준비 (완료)
- `config.py` — 환경변수 중앙 관리 모듈 신규 생성
- `engine.py` — 하드코딩된 GCP 프로젝트 ID/리전을 `config.py` 환경변수로 전환
- `Dockerfile` — python:3.11-slim, graphviz, curl, 헬스체크(120초 start-period)
- `.dockerignore` — manual/, scripts/ 등 런타임 불필요 파일 제외
- `.streamlit/config.toml` — CORS/XSRF 비활성화 (ALB 뒤 동작용)
- `deploy/setup-aws.sh` — ECR, ECS 클러스터, IAM Role, ALB, Security Group, ECS Service 원클릭 구성
- `deploy/deploy.sh` — Docker 빌드 → ECR 푸시 → ECS 강제 재배포 → 안정화 대기
- `.gitignore` — .env, *.pem, deploy/*.json 추가
- `.env.example` — 환경변수 템플릿

### 3-3. DB 연결 로직 구현 (완료)
- `db.py` — asyncpg 커넥션 풀 관리 모듈 신규 생성
  - `create_db_pool()` / `close_db_pool()` / `check_schema()`
  - DATABASE_URL 미설정 시 None 반환 → CSV 모드 폴백
  - SSL 지원 (Neon, Supabase 등 클라우드 DB 호환)
  - 연결 실패 시 앱 크래시 방지 → 경고 로그 + CSV 모드 폴백
  - 스키마 존재 여부 확인 (issue_cracker 스키마 누락 시 경고)
- `config.py` — DB_MIN_CONNECTIONS, DB_MAX_CONNECTIONS, DB_SSL 설정 추가
- `engine.py` — `init_infrastructure()` async 전환, DB 풀 생성·주입
- `app.py` — 비동기 초기화 + 사이드바 DB 연결 상태 표시
- `main.py` — 비동기 초기화 + DB 풀 정리
- `setup-aws.sh` — Task Definition에 DATABASE_URL, DB_SSL 환경변수 추가
- `.env.example` — Neon/Supabase/Aurora/로컬 예시 추가

### 결정된 사항
- **AWS 리전**: `ap-northeast-2` (서울) 고정
- **컨테이너 사양**: 1 vCPU + 4GB 메모리
- **DB**: 외부 PostgreSQL 연결 (DATABASE_URL 환경변수) — 미설정 시 CSV 직접 입력 모드
- **DB 프로비저닝**: 배포 스크립트에서 분리 (콘솔/Terraform/별도 스크립트로 수행)
- **GCP 인증**: 서비스 계정 키를 Secrets Manager에 저장하는 방식 권장 (미구현, 수동 처리 필요)

---

## 4. 미완료 / 후속 작업

### 즉시 필요
- [ ] 로컬 Docker 빌드 테스트 (`docker build -t issue-cracker .`)
- [ ] GCP 서비스 계정 키 준비 (Fargate에서 Gemini API 호출용)
- [ ] `./deploy/setup-aws.sh` 실행하여 AWS 인프라 구성 + 최초 배포

### 단기 개선
- [ ] NVI 산출 가중치 3곳 통일 (`constants.py` 도입)
- [ ] RAG 벡터 DB에 실제 PR 사례/가이드라인 데이터 투입
- [ ] 테스트 코드 작성 (ForecasterAgent SCCT 감쇠 로직 우선)
- [ ] 에러 핸들링 강화 (LLM 호출 실패 시 Fallback)
- [ ] 외부 PostgreSQL 프로비저닝 + DDL 실행 (sql/001_create_tables.sql)

### 중장기
- [ ] 도메인 + HTTPS (ACM + Route 53)
- [ ] Fargate Spot 적용 (비용 최적화)
- [ ] 모니터링/로깅 대시보드 (CloudWatch → Grafana)
- [ ] Streamlit → FastAPI + React 전환 (프로덕션 UI)

---

## 5. 기술적 주의사항

### engine.py 핵심 구조
```python
# config.py에서 환경변수 로드
from config import GCP_PROJECT_ID, GCP_LOCATION, GEMINI_MODEL

# LangGraph StateGraph로 순차 파이프라인 구성
# ForecasterAgent는 mode="baseline"과 mode="mitigated"로 2회 인스턴스화
# Reviewer 반려 시 should_continue() 함수로 Planner 복귀 (최대 3회)
```

### AnalyzerAgent 특수 사항
- `db_pool=None`이면 자동 생략 → CSV 직접 입력 모드
- `async def run()` — 유일한 비동기 에이전트 (asyncpg + asyncio.to_thread)
- 50건 × 5 병렬 배치 + 3회 재시도 (지수 백오프)

### ForecasterAgent SCCT 감쇠 파라미터
```python
CRISIS_DECAY_PARAMS = {
    "victim":       {"decay_rate": 0.06, "neg_ratio_floor": 0.20, ...},  # 빠른 회복
    "accidental":   {"decay_rate": 0.035, "neg_ratio_floor": 0.30, ...}, # 보통 회복
    "preventable":  {"decay_rate": 0.02, "neg_ratio_floor": 0.40, ...},  # 느린 회복 (Sticky)
}
```

### Dockerfile 주요 설계 결정
- `start-period=120s`: sentence-transformers 임베딩 모델 로딩에 ~60초 소요
- `enableCORS=false`: ALB 뒤에서 Streamlit WebSocket 정상 동작 필수
- `graphviz` 시스템 패키지: app.py에서 LangGraph 워크플로우 시각화에 사용

### 배포 스크립트 멱등성
- `setup-aws.sh`는 리소스 존재 여부를 먼저 확인하므로 재실행해도 안전
- `deploy.sh`는 Git SHA 기반 태깅으로 롤백 지원

---

## 6. 커맨드 레퍼런스

```bash
# 로컬 Docker 테스트
docker build -t issue-cracker .
docker run -p 8501:8501 -e GCP_PROJECT_ID=deep-insight-496705 issue-cracker

# AWS 최초 배포
chmod +x deploy/setup-aws.sh && ./deploy/setup-aws.sh

# 이후 배포 (코드 변경 시)
chmod +x deploy/deploy.sh && ./deploy/deploy.sh

# Fargate 비용 절감 (미사용 시)
aws ecs update-service --cluster issue-cracker-cluster --service issue-cracker-service --desired-count 0 --region ap-northeast-2

# Fargate 재시작
aws ecs update-service --cluster issue-cracker-cluster --service issue-cracker-service --desired-count 1 --region ap-northeast-2
```

---

## 7. 비용 예상 (서울 리전, 24/7 기준)

| 구성요소 | 월 비용 |
|:---|:---|
| Fargate (1 vCPU, 4GB) | ~$36 |
| ALB | ~$22 |
| ECR + CloudWatch | ~$2 |
| **합계** | **~$60/월** |
