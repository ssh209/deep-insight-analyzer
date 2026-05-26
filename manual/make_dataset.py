import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_synthetic_dataset(output_path="data/pr_crisis_dataset.csv"):
    print("⏳ PR 위기 시뮬레이션 데이터셋(Raw -> Feature) 생성 중...")
    
    # 1. 기본 시계열 뼈대 생성 (72시간)
    hours = 72
    start_time = datetime.now() - timedelta(hours=hours)
    
    data = {
        'Datetime': [start_time + timedelta(hours=i) for i in range(hours)],
        'Hours_Since_Start': np.arange(hours),
        'Company_Action_Type': np.zeros(hours, dtype=int),
        'Influencer_Impact': np.zeros(hours, dtype=int)
    }
    df = pd.DataFrame(data)

    # 🎯 [이벤트 주입] 특정 시간에 이벤트 발생 시뮬레이션
    df.loc[10, 'Influencer_Impact'] = 2   # 10시간 차: 대형 유튜버의 1차 저격
    df.loc[25, 'Influencer_Impact'] = 1   # 25시간 차: 중소형 유튜버의 2차 확산
    df.loc[30, 'Company_Action_Type'] = 1 # 30시간 차: 회사의 1차 해명문 배포
    df.loc[48, 'Company_Action_Type'] = 2 # 48시간 차: 회사의 전면 사과 및 리콜 공표

    # 2. Raw Data (1차 감성 분류 및 수집 데이터) 시뮬레이션
    # (실제 환경에서는 소셜 리스닝 API나 감성 분류 모델이 채워주는 값)
    total_mentions = []
    negative_mentions = []
    mockery_mentions = []
    advocate_mentions = []
    
    current_total = 100
    for i in range(hours):
        action = df.loc[i, 'Company_Action_Type']
        influencer = df.loc[i, 'Influencer_Impact']
        
        # 유튜버 저격 시 언급량 폭증
        if influencer > 0:
            current_total += np.random.randint(5000, 15000) * influencer
        # 1차 해명문(변명성) 배포 시 오히려 언급량(부정) 일시적 증가
        elif action == 1:
            current_total += np.random.randint(2000, 5000)
        # 2차 사과문(적극적) 배포 후 점진적 진화 (언급량 감소)
        elif action == 2 or (i > 48):
            current_total = max(500, current_total * np.random.uniform(0.7, 0.9))
        else:
            # 자연 증감 노이즈
            current_total = max(100, current_total * np.random.uniform(0.9, 1.2))
            
        total_mentions.append(current_total)
        
        # 감성 분할 (상황에 따라 비율이 달라짐)
        if i < 30: # 무대응/확산기: 부정 및 조롱 극대화
            neg_ratio = np.random.uniform(0.8, 0.95)
            mock_ratio = neg_ratio * np.random.uniform(0.3, 0.6) # 부정 중 조롱의 비율
            adv_ratio = np.random.uniform(0.01, 0.03)
        elif 30 <= i < 48: # 1차 해명 후: 옹호자 약간 생성, 그러나 여전히 조롱 높음
            neg_ratio = np.random.uniform(0.7, 0.85)
            mock_ratio = neg_ratio * np.random.uniform(0.5, 0.8) # 밈화 심화
            adv_ratio = np.random.uniform(0.05, 0.1)
        else: # 2차 사과 후: 옹호자(방어) 증가, 조롱 감소
            neg_ratio = np.random.uniform(0.4, 0.6)
            mock_ratio = neg_ratio * np.random.uniform(0.1, 0.2)
            adv_ratio = np.random.uniform(0.2, 0.35)
            
        negative_mentions.append(int(current_total * neg_ratio))
        mockery_mentions.append(int(current_total * mock_ratio))
        advocate_mentions.append(int(current_total * adv_ratio))

    df['Raw_Total_Mentions'] = total_mentions
    df['Raw_Negative_Mentions'] = negative_mentions
    df['Raw_Mockery_Mentions'] = mockery_mentions
    df['Raw_Advocate_Mentions'] = advocate_mentions

    # ==========================================
    # 🧠 3. Feature Engineering (2차 파생 지표 연산)
    # ==========================================
    
    # [비율 지표] - NVI 산출의 핵심 근거
    df['Negative_Ratio'] = (df['Raw_Negative_Mentions'] / df['Raw_Total_Mentions']).round(3)
    df['Mockery_Index'] = (df['Raw_Mockery_Mentions'] / df['Raw_Total_Mentions']).round(3)
    df['Advocate_Ratio'] = (df['Raw_Advocate_Mentions'] / df['Raw_Total_Mentions']).round(3)
    
    # [가속도/모멘텀 지표] - .diff() 함수로 이전 시간 대비 증감량(기울기) 계산
    df['SNS_Mentions_Velocity'] = df['Raw_Total_Mentions'].diff().fillna(0).astype(int)
    # 부정 모멘텀: 부정 언급이 전 시간 대비 얼마나 폭증했는가?
    df['Negative_Momentum'] = df['Raw_Negative_Mentions'].diff().fillna(0).astype(int)
    
    # 4. Target Variable (Actual_NVI) 산출 로직 시뮬레이션
    # (실제로는 이 값이 Ground Truth가 되며, Forecaster가 이를 학습함)
    nvi_list = [1.0] # 시작은 정상(1.0)
    
    for i in range(1, hours):
        prev_nvi = nvi_list[-1]
        
        # 감점 요인: 높은 부정 비율, 높은 조롱 지수, 부정 모멘텀 폭증
        penalty = (df.loc[i, 'Negative_Ratio'] * 0.1) + \
                  (df.loc[i, 'Mockery_Index'] * 0.15) + \
                  (1 if df.loc[i, 'Negative_Momentum'] > 2000 else 0) * 0.1
                  
        # 가점 요인: 옹호자 비율 증가, 회사의 적극적 대응 액션
        bonus = (df.loc[i, 'Advocate_Ratio'] * 0.2) + \
                (0.05 if df.loc[i, 'Company_Action_Type'] == 1 else 0) + \
                (0.15 if df.loc[i, 'Company_Action_Type'] == 2 else 0)
                
        # NVI 업데이트 (노이즈 약간 추가) 및 클리핑 (0.1 ~ 1.0)
        new_nvi = prev_nvi - penalty + bonus + np.random.uniform(-0.02, 0.02)
        new_nvi = np.clip(new_nvi, 0.1, 1.0)
        nvi_list.append(round(new_nvi, 3))
        
    df['Actual_NVI'] = nvi_list

    # 5. CSV 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Forecaster 모델이 학습하기 좋게 컬럼 순서 정렬
    final_cols = [
        'Datetime', 'Hours_Since_Start', 
        'Company_Action_Type', 'Influencer_Impact',
        'Raw_Total_Mentions', 'Raw_Negative_Mentions', 'Raw_Mockery_Mentions', 'Raw_Advocate_Mentions',
        'Negative_Ratio', 'Mockery_Index', 'Advocate_Ratio', 
        'SNS_Mentions_Velocity', 'Negative_Momentum', 
        'Actual_NVI'
    ]
    df = df[final_cols]
    
    df.to_csv(output_path, index=False)
    print(f"✅ 데이터셋 생성 완료! 저장 위치: {output_path}")
    print(df[['Hours_Since_Start', 'Negative_Ratio', 'Mockery_Index', 'Negative_Momentum', 'Actual_NVI']].head(15))

if __name__ == "__main__":
    create_synthetic_dataset()