import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_synthetic_dataset(output_path="data/pr_crisis_dataset.csv"):
    """
    720시간(30일) PR 위기 시뮬레이션 데이터셋 생성.
    
    5-페이즈 위기 생애주기:
      ① 잠복기 (D0~D2)   → 커뮤니티 발화, 소규모 확산
      ② 폭발기 (D2~D5)   → 유튜버 저격, 주류 미디어 보도
      ③ 확산/바닥 (D5~D10) → 밈화, 해명문 역효과
      ④ 교착기 (D10~D18)  → CEO 사과, 관심 하락
      ⑤ 수습기 (D18~D30)  → 보상/감사, 옹호자 증가
    """
    print("⏳ PR 위기 시뮬레이션 데이터셋(720h/30일) 생성 중...")
    
    hours = 720
    start_time = datetime.now() - timedelta(hours=hours)
    
    data = {
        'Datetime': [start_time + timedelta(hours=i) for i in range(hours)],
        'Hours_Since_Start': np.arange(hours),
        'Company_Action_Type': np.zeros(hours, dtype=int),
        'Influencer_Impact': np.zeros(hours, dtype=int),
    }
    df = pd.DataFrame(data)

    # ==========================================
    # 🎯 14개 이벤트 주입
    # ==========================================
    # ① 잠복기 (0~48h)
    df.loc[24,  'Influencer_Impact'] = 1    # E1: 커뮤니티 폭로 글 바이럴
    df.loc[36,  'Company_Action_Type'] = 1  # E2: 기업 "확인 중" 1줄 공지
    
    # ② 폭발기 (48~120h)
    df.loc[60,  'Influencer_Impact'] = 2    # E3: 대형 유튜버 1차 저격
    df.loc[84,  'Influencer_Impact'] = 2    # E4: 중형 유튜버 2차 저격 + 내부 문건
    df.loc[108, 'Influencer_Impact'] = 1    # E5: 주류 미디어 보도

    # ③ 확산/바닥 (120~240h)
    df.loc[168, 'Company_Action_Type'] = 1  # E6: 1차 해명문 (변명조 → 역효과)
    df.loc[192, 'Influencer_Impact'] = 1    # E7: 패러디/밈 최고 확산기
    
    # ④ 교착기 (240~432h)
    df.loc[240, 'Company_Action_Type'] = 2  # E8: CEO 공식 사과문
    df.loc[312, 'Company_Action_Type'] = 2  # E9: 구체적 개선안 발표
    df.loc[360, 'Influencer_Impact'] = -1   # E10: 중립 유튜버 "사과 분석" (옹호 전환)
    df.loc[408, 'Company_Action_Type'] = 2  # E11: 피해자 보상 프로그램 가동
    
    # ⑤ 수습기 (432~720h)
    df.loc[504, 'Company_Action_Type'] = 2  # E12: 외부 감사 결과 공개
    df.loc[576, 'Influencer_Impact'] = 0    # E13: 새 경쟁사 이슈 (자연 관심 이탈)
    df.loc[648, 'Company_Action_Type'] = 2  # E14: CEO 후속 인터뷰

    # ==========================================
    # 📊 Raw Data 시뮬레이션 (페이즈별 분기)
    # ==========================================
    total_mentions = []
    negative_mentions = []
    mockery_mentions = []
    advocate_mentions = []
    
    current_total = 100
    
    for i in range(hours):
        action = df.loc[i, 'Company_Action_Type']
        influencer = df.loc[i, 'Influencer_Impact']
        
        # --- 언급량 변동 ---
        if influencer == 2:
            current_total += np.random.randint(8000, 25000)
        elif influencer == 1:
            current_total += np.random.randint(1000, 5000)
        elif influencer == -1:  # 옹호 유튜버
            current_total += np.random.randint(500, 2000)
        
        if action == 1 and i < 240:
            # 해명문(변명조) → 오히려 언급량 증가
            current_total += np.random.randint(2000, 5000)
        
        # 페이즈별 자연 증감
        if i < 48:          # ① 잠복기: 완만 증가
            current_total = max(100, current_total * np.random.uniform(1.02, 1.08))
        elif i < 120:       # ② 폭발기: 고수준 유지
            current_total = max(500, current_total * np.random.uniform(0.95, 1.10))
        elif i < 240:       # ③ 확산/바닥: 고원 유지 후 소폭 하락
            current_total = max(500, current_total * np.random.uniform(0.93, 1.02))
        elif i < 432:       # ④ 교착기: 점진 하락
            decay = 0.96 if action == 2 else np.random.uniform(0.94, 0.99)
            current_total = max(300, current_total * decay)
        else:               # ⑤ 수습기: 자연 소멸
            current_total = max(200, current_total * np.random.uniform(0.90, 0.97))
        
        total_mentions.append(current_total)
        
        # --- 감성 분할 (페이즈별 비율) ---
        if i < 48:          # ① 잠복기
            neg_ratio = np.random.uniform(0.40, 0.55)
            mock_of_neg = np.random.uniform(0.10, 0.20)
            adv_ratio = np.random.uniform(0.03, 0.08)
        elif i < 120:       # ② 폭발기
            neg_ratio = np.random.uniform(0.80, 0.95)
            mock_of_neg = np.random.uniform(0.30, 0.60)
            adv_ratio = np.random.uniform(0.01, 0.03)
        elif i < 240:       # ③ 확산/바닥
            neg_ratio = np.random.uniform(0.85, 0.95)
            mock_of_neg = np.random.uniform(0.50, 0.80)
            adv_ratio = np.random.uniform(0.02, 0.05)
        elif i < 432:       # ④ 교착기
            # CEO 사과 이후 감성 뚜렷한 회복
            progress = (i - 240) / (432 - 240)  # 0→1
            neg_base = 0.65 - progress * 0.30   # 0.65 → 0.35
            neg_ratio = np.random.uniform(max(0.30, neg_base - 0.05), neg_base + 0.05)
            mock_of_neg = np.random.uniform(max(0.05, 0.25 - progress * 0.15), max(0.10, 0.40 - progress * 0.25))
            adv_ratio = np.random.uniform(0.10 + progress * 0.15, 0.18 + progress * 0.17)
        else:               # ⑤ 수습기
            progress = (i - 432) / (720 - 432)  # 0→1
            neg_base = 0.35 - progress * 0.15   # 0.35 → 0.20
            neg_ratio = np.random.uniform(max(0.15, neg_base - 0.05), max(0.20, neg_base + 0.05))
            mock_of_neg = np.random.uniform(0.05, max(0.08, 0.15 - progress * 0.10))
            adv_ratio = np.random.uniform(0.25 + progress * 0.10, 0.35 + progress * 0.10)
        
        # 클리핑
        neg_ratio = np.clip(neg_ratio, 0.05, 0.98)
        adv_ratio = np.clip(adv_ratio, 0.01, 0.50)
        
        negative_mentions.append(int(current_total * neg_ratio))
        mockery_mentions.append(int(current_total * neg_ratio * mock_of_neg))
        advocate_mentions.append(int(current_total * adv_ratio))
    
    df['Raw_Total_Mentions'] = total_mentions
    df['Raw_Negative_Mentions'] = negative_mentions
    df['Raw_Mockery_Mentions'] = mockery_mentions
    df['Raw_Advocate_Mentions'] = advocate_mentions

    # ==========================================
    # 🧠 Feature Engineering
    # ==========================================
    df['Negative_Ratio'] = (df['Raw_Negative_Mentions'] / df['Raw_Total_Mentions']).round(3)
    df['Mockery_Index'] = (df['Raw_Mockery_Mentions'] / df['Raw_Total_Mentions']).round(3)
    df['Advocate_Ratio'] = (df['Raw_Advocate_Mentions'] / df['Raw_Total_Mentions']).round(3)
    
    df['SNS_Mentions_Velocity'] = df['Raw_Total_Mentions'].diff().fillna(0).astype(int)
    df['Negative_Momentum'] = df['Raw_Negative_Mentions'].diff().fillna(0).astype(int)

    # ==========================================
    # 🎯 NVI (Target Variable) 산출
    # ==========================================
    nvi_list = [1.0]
    
    for i in range(1, hours):
        prev_nvi = nvi_list[-1]
        neg_r = df.loc[i, 'Negative_Ratio']
        mock_r = df.loc[i, 'Mockery_Index']
        adv_r = df.loc[i, 'Advocate_Ratio']
        momentum = df.loc[i, 'Negative_Momentum']
        action = df.loc[i, 'Company_Action_Type']
        
        # 감점 요인 (시간당 누적이므로 가중치를 낮춤)
        penalty = (neg_r * 0.03) + \
                  (mock_r * 0.04) + \
                  (0.05 if momentum > 2000 else 0)
                  
        # 가점 요인
        bonus = (adv_r * 0.06) + \
                (0.01 if action == 1 else 0) + \
                (0.04 if action == 2 else 0)
        
        # 자연 회귀력: NVI가 극단(0.1 or 1.0)에서 멀수록 변동 작음,
        # 가까울수록 반대 방향으로 약한 복원력 발생 (평형점 0.5)
        reversion = (0.5 - prev_nvi) * 0.002
        
        new_nvi = prev_nvi - penalty + bonus + reversion + np.random.uniform(-0.005, 0.005)
        new_nvi = np.clip(new_nvi, 0.1, 1.0)
        nvi_list.append(round(new_nvi, 3))
        
    df['Actual_NVI'] = nvi_list

    # ==========================================
    # 💾 CSV 저장
    # ==========================================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
    print(f"✅ 데이터셋 생성 완료! ({hours}시간/{hours//24}일)")
    print(f"   저장 위치: {output_path}")
    print(f"   NVI 범위: {df['Actual_NVI'].min():.3f} ~ {df['Actual_NVI'].max():.3f}")
    print(f"\n📊 페이즈별 NVI 평균:")
    for label, s, e in [("① 잠복기", 0,48), ("② 폭발기", 48,120), ("③ 확산/바닥", 120,240), ("④ 교착기", 240,432), ("⑤ 수습기", 432,720)]:
        subset = df[(df['Hours_Since_Start'] >= s) & (df['Hours_Since_Start'] < e)]
        print(f"   {label} (H{s}~H{e}): NVI avg={subset['Actual_NVI'].mean():.3f}, min={subset['Actual_NVI'].min():.3f}")
def create_input_dataset(output_path="data/input_crisis_72h.csv"):
    """
    72시간 실전 입력 데이터셋 생성.
    위기 초기(잠복기~폭발기 초반)만 포함하여 '현재 상황'을 표현합니다.
    ForecasterAgent는 이 데이터의 마지막 시점에서 미래를 예측합니다.
    """
    print("\n--- 72h 실전 입력 데이터 생성 ---")
    
    hours = 72
    start_time = datetime.now() - timedelta(hours=hours)
    
    data = {
        'Datetime': [start_time + timedelta(hours=i) for i in range(hours)],
        'Hours_Since_Start': np.arange(hours),
        'Company_Action_Type': np.zeros(hours, dtype=int),
        'Influencer_Impact': np.zeros(hours, dtype=int),
    }
    df = pd.DataFrame(data)

    # 이벤트: 잠복기 + 폭발기 초반
    df.loc[12,  'Influencer_Impact'] = 1    # 커뮤니티 폭로
    df.loc[24,  'Company_Action_Type'] = 1  # "확인 중" 공지
    df.loc[48,  'Influencer_Impact'] = 2    # 대형 유튜버 저격
    df.loc[60,  'Influencer_Impact'] = 2    # 2차 저격 + 문건 유출

    # Raw Data 시뮬레이션
    total_mentions = []
    negative_mentions = []
    mockery_mentions = []
    advocate_mentions = []
    
    current_total = 100
    for i in range(hours):
        action = df.loc[i, 'Company_Action_Type']
        influencer = df.loc[i, 'Influencer_Impact']
        
        if influencer == 2:
            current_total += np.random.randint(8000, 20000)
        elif influencer == 1:
            current_total += np.random.randint(1000, 4000)
        if action == 1:
            current_total += np.random.randint(1000, 3000)
        
        if i < 24:       # 잠복기
            current_total = max(100, current_total * np.random.uniform(1.02, 1.10))
        elif i < 48:     # 잠복기 후반
            current_total = max(200, current_total * np.random.uniform(1.00, 1.08))
        else:            # 폭발기
            current_total = max(500, current_total * np.random.uniform(0.95, 1.10))
        
        total_mentions.append(current_total)
        
        # 감성 비율
        if i < 24:       # 잠복기 초반
            neg_ratio = np.random.uniform(0.35, 0.50)
            mock_of_neg = np.random.uniform(0.10, 0.20)
            adv_ratio = np.random.uniform(0.05, 0.10)
        elif i < 48:     # 잠복기 후반
            neg_ratio = np.random.uniform(0.55, 0.70)
            mock_of_neg = np.random.uniform(0.20, 0.35)
            adv_ratio = np.random.uniform(0.03, 0.06)
        else:            # 폭발기
            neg_ratio = np.random.uniform(0.80, 0.95)
            mock_of_neg = np.random.uniform(0.35, 0.65)
            adv_ratio = np.random.uniform(0.01, 0.03)
        
        neg_ratio = np.clip(neg_ratio, 0.05, 0.98)
        adv_ratio = np.clip(adv_ratio, 0.01, 0.50)
        
        negative_mentions.append(int(current_total * neg_ratio))
        mockery_mentions.append(int(current_total * neg_ratio * mock_of_neg))
        advocate_mentions.append(int(current_total * adv_ratio))
    
    df['Raw_Total_Mentions'] = total_mentions
    df['Raw_Negative_Mentions'] = negative_mentions
    df['Raw_Mockery_Mentions'] = mockery_mentions
    df['Raw_Advocate_Mentions'] = advocate_mentions
    
    # Feature Engineering
    df['Negative_Ratio'] = (df['Raw_Negative_Mentions'] / df['Raw_Total_Mentions']).round(3)
    df['Mockery_Index'] = (df['Raw_Mockery_Mentions'] / df['Raw_Total_Mentions']).round(3)
    df['Advocate_Ratio'] = (df['Raw_Advocate_Mentions'] / df['Raw_Total_Mentions']).round(3)
    df['SNS_Mentions_Velocity'] = df['Raw_Total_Mentions'].diff().fillna(0).astype(int)
    df['Negative_Momentum'] = df['Raw_Negative_Mentions'].diff().fillna(0).astype(int)
    
    # NVI 산출
    nvi_list = [1.0]
    for i in range(1, hours):
        prev_nvi = nvi_list[-1]
        neg_r = df.loc[i, 'Negative_Ratio']
        mock_r = df.loc[i, 'Mockery_Index']
        adv_r = df.loc[i, 'Advocate_Ratio']
        momentum = df.loc[i, 'Negative_Momentum']
        action = df.loc[i, 'Company_Action_Type']
        
        penalty = (neg_r * 0.03) + (mock_r * 0.04) + (0.05 if momentum > 2000 else 0)
        bonus = (adv_r * 0.06) + (0.01 if action == 1 else 0) + (0.04 if action == 2 else 0)
        reversion = (0.5 - prev_nvi) * 0.002
        
        new_nvi = prev_nvi - penalty + bonus + reversion + np.random.uniform(-0.005, 0.005)
        new_nvi = np.clip(new_nvi, 0.1, 1.0)
        nvi_list.append(round(new_nvi, 3))
    
    df['Actual_NVI'] = nvi_list
    
    final_cols = [
        'Datetime', 'Hours_Since_Start',
        'Company_Action_Type', 'Influencer_Impact',
        'Raw_Total_Mentions', 'Raw_Negative_Mentions', 'Raw_Mockery_Mentions', 'Raw_Advocate_Mentions',
        'Negative_Ratio', 'Mockery_Index', 'Advocate_Ratio',
        'SNS_Mentions_Velocity', 'Negative_Momentum',
        'Actual_NVI'
    ]
    df = df[final_cols]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] 72h input data saved: {output_path}")
    print(f"   NVI: {df['Actual_NVI'].iloc[0]:.3f} -> {df['Actual_NVI'].iloc[-1]:.3f} (last)")

if __name__ == "__main__":
    create_synthetic_dataset()       # 720h 학습 데이터
    create_input_dataset()           # 72h 실전 입력 데이터