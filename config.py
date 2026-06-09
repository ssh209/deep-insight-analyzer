"""
중앙 설정 모듈 — 환경변수 기반 설정 관리.

로컬 개발 시: .env 파일 또는 시스템 환경변수
Fargate 배포 시: ECS Task Definition의 환경변수 / Secrets Manager
"""
import os
from dotenv import load_dotenv

# .env 파일 자동 로딩 (없으면 무시)
load_dotenv()

# ==========================================
# 🔑 GCP / Vertex AI
# ==========================================
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "deep-insight-496705")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

# ==========================================
# 🤖 에이전트별 LLM 모델 (멀티모델 전략)
#
# Flash: 고속·저비용 — 단순 분류, 지시서, 정량 비교
# Pro:   고추론·고품질 — 전략 수립, 리포트 융합, 뉘앙스 판별
# ==========================================
FLASH_MODEL = os.environ.get("FLASH_MODEL", "gemini-2.5-flash")
PRO_MODEL = os.environ.get("PRO_MODEL", "gemini-2.5-pro")

# QueryBuilder: grounding(Pro) + JSON 추출(Flash) 2-step
QUERY_BUILDER_GROUNDING_MODEL = os.environ.get("QUERY_BUILDER_GROUNDING_MODEL", PRO_MODEL)
QUERY_BUILDER_MODEL = os.environ.get("QUERY_BUILDER_MODEL", FLASH_MODEL)

# Analyzer: 감성 배치(Flash) + 테마/KOL 심층(Pro)
ANALYZER_MODEL = os.environ.get("ANALYZER_MODEL", FLASH_MODEL)
ANALYZER_DEEP_MODEL = os.environ.get("ANALYZER_DEEP_MODEL", PRO_MODEL)

# Reporter 파이프라인
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", FLASH_MODEL)
STRATEGIST_MODEL = os.environ.get("STRATEGIST_MODEL", PRO_MODEL)
ANALYST_MODEL = os.environ.get("ANALYST_MODEL", FLASH_MODEL)
COMPILER_MODEL = os.environ.get("COMPILER_MODEL", PRO_MODEL)
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", PRO_MODEL)

# ==========================================
# 📂 데이터 / 출력 경로
# ==========================================
TRAIN_CSV_PATH = os.environ.get("TRAIN_CSV_PATH", "data/pr_crisis_dataset.csv")
INPUT_CSV_PATH = os.environ.get("INPUT_CSV_PATH", "data/input_crisis_72h.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")          # Analyzer CSV 출력
REPORT_DIR = os.environ.get("REPORT_DIR", "reports")         # HTML 보고서 로컬 저장

# LightGBM 모델
LGBM_MODEL_PATH = os.environ.get("LGBM_MODEL_PATH", "models/lightgbm/nvi_forecaster.pkl")

# TFT 모델
TFT_MODEL_PATH = os.environ.get("TFT_MODEL_PATH", "models/tft/tft_nvi.ckpt")

# ==========================================
# 🗄️ Database (외부 PostgreSQL 연결)
# ==========================================
DATABASE_URL = os.environ.get("DATABASE_URL", "")
DB_MIN_CONNECTIONS = int(os.environ.get("DB_MIN_CONNECTIONS", "2"))
DB_MAX_CONNECTIONS = int(os.environ.get("DB_MAX_CONNECTIONS", "10"))
DB_SSL = os.environ.get("DB_SSL", "true").lower() in ("true", "1", "yes")

# ==========================================
# 🔤 Embedding Model
# ==========================================
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

# ==========================================
# 🔮 Forecaster Model
# ==========================================
FORECASTER_MODEL = os.environ.get("FORECASTER_MODEL", "lightgbm")  # "lightgbm" | "tft" | "moirai" | "arima"

# ==========================================
# 📄 Report (HTML 보고서 → GCS 업로드)
# ==========================================
GCS_REPORT_BUCKET = os.environ.get("GCS_REPORT_BUCKET", "")  # 비어있으면 로컬 저장만

# ==========================================
# 🌐 Streamlit
# ==========================================
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8501"))
