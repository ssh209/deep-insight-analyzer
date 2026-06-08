#!/bin/bash
# ==========================================
# Issue Cracker — AWS Fargate 인프라 초기 구성
#
# 사전 조건:
#   1. AWS CLI 설치 + 'aws configure' 완료
#   2. Docker 설치
#   3. 적절한 IAM 권한 (ECS, ECR, EC2, ELB, IAM, CloudWatch)
#
# 사용법:
#   chmod +x deploy/setup-aws.sh
#   ./deploy/setup-aws.sh
# ==========================================

set -euo pipefail

# ==========================================
# 🔧 설정 변수 (필요 시 수정)
# ==========================================
AWS_REGION="ap-northeast-2"
APP_NAME="issue-cracker"
ECR_REPO="${APP_NAME}"
ECS_CLUSTER="${APP_NAME}-cluster"
ECS_SERVICE="${APP_NAME}-service"
ECS_TASK_FAMILY="${APP_NAME}-task"
LOG_GROUP="/ecs/${APP_NAME}"
CONTAINER_PORT=8501
TASK_CPU=1024        # 1 vCPU
TASK_MEMORY=4096     # 4 GB

echo "============================================"
echo "🚀 Issue Cracker — Fargate 인프라 구성 시작"
echo "   Region: ${AWS_REGION}"
echo "============================================"

# ==========================================
# Step 1: ECR 리포지토리 생성
# ==========================================
echo ""
echo "📦 [Step 1/8] ECR 리포지토리 생성..."
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" 2>/dev/null \
    || aws ecr create-repository \
        --repository-name "${ECR_REPO}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256

ECR_URI=$(aws ecr describe-repositories \
    --repository-names "${ECR_REPO}" \
    --region "${AWS_REGION}" \
    --query 'repositories[0].repositoryUri' \
    --output text)
echo "   ✅ ECR URI: ${ECR_URI}"

# ==========================================
# Step 2: Docker 이미지 빌드 & ECR 푸시
# ==========================================
echo ""
echo "🐳 [Step 2/8] Docker 이미지 빌드 & ECR 푸시..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)

aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build -t "${APP_NAME}:latest" .
docker tag "${APP_NAME}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
echo "   ✅ 이미지 푸시 완료"

# ==========================================
# Step 3: ECS 클러스터 생성
# ==========================================
echo ""
echo "🏗️ [Step 3/8] ECS 클러스터 생성..."
aws ecs describe-clusters --clusters "${ECS_CLUSTER}" --region "${AWS_REGION}" \
    --query 'clusters[?status==`ACTIVE`].clusterName' --output text | grep -q "${ECS_CLUSTER}" \
    || aws ecs create-cluster \
        --cluster-name "${ECS_CLUSTER}" \
        --region "${AWS_REGION}" \
        --setting name=containerInsights,value=enabled
echo "   ✅ 클러스터: ${ECS_CLUSTER}"

# ==========================================
# Step 4: CloudWatch 로그 그룹 생성
# ==========================================
echo ""
echo "📊 [Step 4/8] CloudWatch 로그 그룹 생성..."
aws logs describe-log-groups --log-group-name-prefix "${LOG_GROUP}" --region "${AWS_REGION}" \
    --query 'logGroups[0].logGroupName' --output text 2>/dev/null | grep -q "${LOG_GROUP}" \
    || aws logs create-log-group \
        --log-group-name "${LOG_GROUP}" \
        --region "${AWS_REGION}"
echo "   ✅ 로그 그룹: ${LOG_GROUP}"

# ==========================================
# Step 5: IAM Task Execution Role 생성
# ==========================================
echo ""
echo "🔑 [Step 5/8] IAM Task Execution Role 생성..."
ROLE_NAME="ecsTaskExecutionRole"
ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text 2>/dev/null || true)

if [ -z "${ROLE_ARN}" ] || [ "${ROLE_ARN}" = "None" ]; then
    cat > /tmp/ecs-trust-policy.json <<'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": { "Service": "ecs-tasks.amazonaws.com" },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
    aws iam create-role \
        --role-name "${ROLE_NAME}" \
        --assume-role-policy-document file:///tmp/ecs-trust-policy.json

    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

    ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text)
    rm /tmp/ecs-trust-policy.json
fi
echo "   ✅ Role ARN: ${ROLE_ARN}"

# ==========================================
# Step 6: 기본 VPC + 서브넷 조회
# ==========================================
echo ""
echo "🌐 [Step 6/8] 기본 VPC + 서브넷 조회..."
VPC_ID=$(aws ec2 describe-vpcs \
    --filters Name=isDefault,Values=true \
    --region "${AWS_REGION}" \
    --query 'Vpcs[0].VpcId' --output text)

SUBNET_IDS=$(aws ec2 describe-subnets \
    --filters Name=vpc-id,Values="${VPC_ID}" \
    --region "${AWS_REGION}" \
    --query 'Subnets[*].SubnetId' --output text | tr '\t' ',')

echo "   VPC: ${VPC_ID}"
echo "   Subnets: ${SUBNET_IDS}"

# ==========================================
# Step 7: Security Group 생성
# ==========================================
echo ""
echo "🔒 [Step 7/8] Security Group 생성..."
SG_NAME="${APP_NAME}-sg"
SG_ID=$(aws ec2 describe-security-groups \
    --filters Name=group-name,Values="${SG_NAME}" Name=vpc-id,Values="${VPC_ID}" \
    --region "${AWS_REGION}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)

if [ -z "${SG_ID}" ] || [ "${SG_ID}" = "None" ]; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name "${SG_NAME}" \
        --description "Issue Cracker Fargate SG" \
        --vpc-id "${VPC_ID}" \
        --region "${AWS_REGION}" \
        --query 'GroupId' --output text)

    # HTTP 80 (ALB)
    aws ec2 authorize-security-group-ingress \
        --group-id "${SG_ID}" \
        --protocol tcp --port 80 --cidr 0.0.0.0/0 \
        --region "${AWS_REGION}"

    # Streamlit 8501 (컨테이너)
    aws ec2 authorize-security-group-ingress \
        --group-id "${SG_ID}" \
        --protocol tcp --port "${CONTAINER_PORT}" --cidr 0.0.0.0/0 \
        --region "${AWS_REGION}"
fi
echo "   ✅ Security Group: ${SG_ID}"

# ==========================================
# Step 8: ALB + Target Group + ECS Service
# ==========================================
echo ""
echo "⚖️ [Step 8/8] ALB + Target Group + ECS Service 생성..."

# ALB 생성
ALB_ARN=$(aws elbv2 describe-load-balancers \
    --names "${APP_NAME}-alb" \
    --region "${AWS_REGION}" \
    --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || true)

if [ -z "${ALB_ARN}" ] || [ "${ALB_ARN}" = "None" ]; then
    # 서브넷을 개별 인자로 전달
    SUBNET_ARRAY=(${SUBNET_IDS//,/ })
    ALB_ARN=$(aws elbv2 create-load-balancer \
        --name "${APP_NAME}-alb" \
        --subnets ${SUBNET_ARRAY[@]} \
        --security-groups "${SG_ID}" \
        --scheme internet-facing \
        --type application \
        --region "${AWS_REGION}" \
        --query 'LoadBalancers[0].LoadBalancerArn' --output text)
fi

# Target Group 생성
TG_ARN=$(aws elbv2 describe-target-groups \
    --names "${APP_NAME}-tg" \
    --region "${AWS_REGION}" \
    --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || true)

if [ -z "${TG_ARN}" ] || [ "${TG_ARN}" = "None" ]; then
    TG_ARN=$(aws elbv2 create-target-group \
        --name "${APP_NAME}-tg" \
        --protocol HTTP \
        --port "${CONTAINER_PORT}" \
        --vpc-id "${VPC_ID}" \
        --target-type ip \
        --health-check-path "/_stcore/health" \
        --health-check-interval-seconds 30 \
        --healthy-threshold-count 2 \
        --unhealthy-threshold-count 5 \
        --health-check-timeout-seconds 10 \
        --region "${AWS_REGION}" \
        --query 'TargetGroups[0].TargetGroupArn' --output text)
fi

# Listener 생성
aws elbv2 describe-listeners \
    --load-balancer-arn "${ALB_ARN}" \
    --region "${AWS_REGION}" \
    --query 'Listeners[0].ListenerArn' --output text 2>/dev/null | grep -q "arn:" \
    || aws elbv2 create-listener \
        --load-balancer-arn "${ALB_ARN}" \
        --protocol HTTP \
        --port 80 \
        --default-actions Type=forward,TargetGroupArn="${TG_ARN}" \
        --region "${AWS_REGION}"

# ECS Task Definition 등록
echo ""
echo "📋 Task Definition 등록..."
cat > /tmp/task-def.json <<TASKEOF
{
    "family": "${ECS_TASK_FAMILY}",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "${TASK_CPU}",
    "memory": "${TASK_MEMORY}",
    "executionRoleArn": "${ROLE_ARN}",
    "containerDefinitions": [
        {
            "name": "${APP_NAME}",
            "image": "${ECR_URI}:latest",
            "portMappings": [
                { "containerPort": ${CONTAINER_PORT}, "protocol": "tcp" }
            ],
            "environment": [
                { "name": "GCP_PROJECT_ID", "value": "${GCP_PROJECT_ID:-deep-insight-496705}" },
                { "name": "GCP_LOCATION", "value": "${GCP_LOCATION:-global}" },
                { "name": "GEMINI_MODEL", "value": "${GEMINI_MODEL:-gemini-2.5-pro}" },
                { "name": "DATABASE_URL", "value": "${DATABASE_URL:-}" },
                { "name": "DB_SSL", "value": "${DB_SSL:-true}" }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "${LOG_GROUP}",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "essential": true
        }
    ]
}
TASKEOF

aws ecs register-task-definition \
    --cli-input-json file:///tmp/task-def.json \
    --region "${AWS_REGION}" > /dev/null
rm /tmp/task-def.json

# ECS Service 생성
echo ""
echo "🚢 ECS Service 생성..."
SUBNET_ARRAY=(${SUBNET_IDS//,/ })
FIRST_TWO_SUBNETS="${SUBNET_ARRAY[0]},${SUBNET_ARRAY[1]}"

aws ecs describe-services \
    --cluster "${ECS_CLUSTER}" \
    --services "${ECS_SERVICE}" \
    --region "${AWS_REGION}" \
    --query 'services[?status==`ACTIVE`].serviceName' --output text 2>/dev/null | grep -q "${ECS_SERVICE}" \
    || aws ecs create-service \
        --cluster "${ECS_CLUSTER}" \
        --service-name "${ECS_SERVICE}" \
        --task-definition "${ECS_TASK_FAMILY}" \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[${FIRST_TWO_SUBNETS}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
        --load-balancers "targetGroupArn=${TG_ARN},containerName=${APP_NAME},containerPort=${CONTAINER_PORT}" \
        --region "${AWS_REGION}" > /dev/null

# ==========================================
# 🎉 완료
# ==========================================
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --names "${APP_NAME}-alb" \
    --region "${AWS_REGION}" \
    --query 'LoadBalancers[0].DNSName' --output text)

echo ""
echo "============================================"
echo "🎉 Fargate 인프라 구성 완료!"
echo "============================================"
echo ""
echo "🌐 대시보드 접속 URL:"
echo "   http://${ALB_DNS}"
echo ""
echo "📊 CloudWatch 로그:"
echo "   https://${AWS_REGION}.console.aws.amazon.com/cloudwatch/home?region=${AWS_REGION}#logsV2:log-groups/log-group/${LOG_GROUP}"
echo ""
echo "⚠️ 참고사항:"
echo "   - ALB가 healthy 상태가 되기까지 2~3분 소요됩니다."
echo "   - GCP 인증: 컨테이너에 서비스 계정 키를 마운트하거나 환경변수로 제공해야 합니다."
echo "   - 비용: Fargate 1 vCPU + 4GB 기준 약 \$0.05/h (서울 리전)"
echo ""
