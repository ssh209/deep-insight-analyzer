#!/bin/bash
# ==========================================
# Issue Cracker — 이미지 빌드 & 배포 자동화
#
# 사용법:
#   chmod +x deploy/deploy.sh
#   ./deploy/deploy.sh
# ==========================================

set -euo pipefail

AWS_REGION="ap-northeast-2"
APP_NAME="issue-cracker"
ECR_REPO="${APP_NAME}"
ECS_CLUSTER="${APP_NAME}-cluster"
ECS_SERVICE="${APP_NAME}-service"
ECS_TASK_FAMILY="${APP_NAME}-task"

# Git SHA 태그 (롤백용)
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "manual")
IMAGE_TAG="${GIT_SHA}"

echo "============================================"
echo "🚀 Issue Cracker — 배포 시작"
echo "   Tag: ${IMAGE_TAG}"
echo "============================================"

# ==========================================
# Step 1: ECR 정보 조회 & 로그인
# ==========================================
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo ""
echo "🔑 ECR 로그인..."
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ==========================================
# Step 2: Docker 이미지 빌드
# ==========================================
echo ""
echo "🐳 Docker 이미지 빌드..."
docker build -t "${APP_NAME}:${IMAGE_TAG}" -t "${APP_NAME}:latest" .

# ==========================================
# Step 3: ECR 푸시 (SHA 태그 + latest)
# ==========================================
echo ""
echo "📦 ECR 푸시..."
docker tag "${APP_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker tag "${APP_NAME}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:latest"
echo "   ✅ Pushed: ${ECR_URI}:${IMAGE_TAG}"

# ==========================================
# Step 4: ECS 강제 배포
# ==========================================
echo ""
echo "🚢 ECS 서비스 업데이트 (강제 재배포)..."
aws ecs update-service \
    --cluster "${ECS_CLUSTER}" \
    --service "${ECS_SERVICE}" \
    --force-new-deployment \
    --region "${AWS_REGION}" > /dev/null

echo "   ⏳ 배포 안정화 대기 중... (최대 5분)"
aws ecs wait services-stable \
    --cluster "${ECS_CLUSTER}" \
    --services "${ECS_SERVICE}" \
    --region "${AWS_REGION}"

# ==========================================
# 🎉 완료
# ==========================================
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --names "${APP_NAME}-alb" \
    --region "${AWS_REGION}" \
    --query 'LoadBalancers[0].DNSName' --output text 2>/dev/null || echo "N/A")

echo ""
echo "============================================"
echo "🎉 배포 완료!"
echo "============================================"
echo "   Image: ${ECR_URI}:${IMAGE_TAG}"
echo "   URL:   http://${ALB_DNS}"
echo "============================================"
