"""
중앙 설정 모듈 — 환경변수 기반 설정 관리.

로컬 개발 시: .env 파일 또는 시스템 환경변수
Fargate 배포 시: ECS Task Definition의 환경변수 / Secrets Manager
"""
import os

# ==========================================
# 🔑 GCP / Vertex AI
# ==========================================
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "deep-insight-496705")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

# ==========================================
# 📊 데이터 경로
# ==========================================
TRAIN_CSV_PATH = os.environ.get("TRAIN_CSV_PATH", "data/pr_crisis_dataset.csv")
INPUT_CSV_PATH = os.environ.get("INPUT_CSV_PATH", "data/input_crisis_72h.csv")

# ==========================================
# 🗄️ Database (향후 Aurora Serverless 연동 시 사용)
# ==========================================
DATABASE_URL = os.environ.get("DATABASE_URL", "")

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
# 🌐 Streamlit
# ==========================================
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8501"))
