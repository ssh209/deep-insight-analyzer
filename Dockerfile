# ==========================================
# Issue Cracker — Fargate 컨테이너 이미지
# 
# 빌드: docker build -t issue-cracker .
# 실행: docker run -p 8501:8501 -e GCP_PROJECT_ID=xxx issue-cracker
# ==========================================

FROM python:3.11-slim

# 시스템 의존성 (graphviz 바이너리 + curl 헬스체크)
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (Docker 레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# Streamlit 환경변수
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# 헬스체크 (ALB Target Group 용)
# start-period: sentence-transformers 모델 로딩 대기 (최대 120초)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
