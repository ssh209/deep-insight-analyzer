import pandas as pd
import numpy as np

# 1. 시뮬레이션 기간: 2026-06-01 부터 2026-06-07 12:00 (현재 시점) 까지 생성
dates = pd.date_range(start="2026-06-01 00:00:00", end="2026-06-07 12:00:00", freq="h")
df = pd.DataFrame({"Datetime": dates})

np.random.seed(42)
# 🎯 윈도우 환경 dtype 충돌 방지: 생성 시점에 모두 int64로 강제 고정!
df['SNS_Mentions'] = np.random.randint(10, 50, len(df)).astype(np.int64)
df['News_Articles'] = np.random.randint(0, 5, len(df)).astype(np.int64)
df['Influencer_Hit'] = np.int64(0)
df['Victim_Claims'] = np.int64(0)
df['Boycott_Mentions'] = np.int64(0)
df['Neutral_Shares'] = np.random.randint(20, 100, len(df)).astype(np.int64)
df['Company_Action'] = np.int64(0) 

# ==========================================
# 🛑 [Event 1] 최초 이슈 발생 (커뮤니티 발 호소글)
# ==========================================
issue_start = pd.to_datetime("2026-06-05 20:00:00")
m1 = (df['Datetime'] >= issue_start)

# 이제 맘 편하게 np.int64로 변환해서 던져주면 됩니다.
claims_series = pd.Series(np.linspace(10, 300, m1.sum()).astype(np.int64), index=df[m1].index)
mentions_series = pd.Series(np.linspace(100, 1500, m1.sum()).astype(np.int64), index=df[m1].index)

df.loc[m1, 'Victim_Claims'] = df.loc[m1, 'Victim_Claims'] + claims_series
df.loc[m1, 'SNS_Mentions'] = df.loc[m1, 'SNS_Mentions'] + mentions_series

# ==========================================
# 💥 [Event 2] 유튜버 A 저격 영상 업로드
# ==========================================
# 발생: 2026-06-06 10:00:00 (이슈 발생 14시간 뒤)
inf1_time = issue_start + pd.Timedelta(hours=14)
df.loc[df['Datetime'] == inf1_time, 'Influencer_Hit'] = 1

m2 = (df['Datetime'] >= inf1_time)
df.loc[m2, 'SNS_Mentions'] = df.loc[m2, 'SNS_Mentions'] + np.random.randint(2000, 5000, m2.sum())
df.loc[m2, 'Boycott_Mentions'] = df.loc[m2, 'Boycott_Mentions'] + np.random.randint(500, 2000, m2.sum())
df.loc[m2, 'News_Articles'] = df.loc[m2, 'News_Articles'] + np.random.randint(50, 150, m2.sum())

# ==========================================
# 💣 [Event 3] 유튜버 B 확인사살 영상 업로드 (6시간 간격)
# ==========================================
# 발생: 2026-06-06 16:00:00 (유튜버 A 업로드 6시간 뒤)
inf2_time = inf1_time + pd.Timedelta(hours=6)
df.loc[df['Datetime'] == inf2_time, 'Influencer_Hit'] = 1

m3 = (df['Datetime'] >= inf2_time)
# 여론 악화 가속 (언급량 및 보이콧 폭발)
df.loc[m3, 'SNS_Mentions'] = df.loc[m3, 'SNS_Mentions'] + np.random.randint(5000, 12000, m3.sum())
df.loc[m3, 'Boycott_Mentions'] = df.loc[m3, 'Boycott_Mentions'] + np.random.randint(3000, 8000, m3.sum())
df.loc[m3, 'News_Articles'] = df.loc[m3, 'News_Articles'] + np.random.randint(150, 400, m3.sum())

# ==========================================
# 📉 NVI (여론 지수) 산출
# ==========================================
negative_impact = (df['Boycott_Mentions'] * 2.0) + (df['Victim_Claims'] * 3.0) + (df['SNS_Mentions'] * 0.5)
total_volume = df['SNS_Mentions'] + df['News_Articles'] + df['Boycott_Mentions'] + df['Victim_Claims'] + df['Neutral_Shares']

df['Actual_NVI'] = 1.0 - (negative_impact / total_volume.replace(0, 1))

# 인플루언서 타격 시 NVI 추가 페널티 부여
df.loc[df['Datetime'] >= inf1_time, 'Actual_NVI'] -= 0.15
df.loc[df['Datetime'] >= inf2_time, 'Actual_NVI'] -= 0.20

df['Actual_NVI'] = df['Actual_NVI'].clip(lower=0.1, upper=1.0).round(3)

# 모델 호환용 매핑 및 저장
df['Positive'] = df['Neutral_Shares']
df['Negative'] = df['Boycott_Mentions'] + df['Victim_Claims']
df['Neutral'] = df['SNS_Mentions']

df.to_csv("data/pr_crisis_dataset.csv", index=False)
print("✅ 실전 데이터셋 생성 완료: pr_crisis_dataset.csv")
print(f"   - 최종(현재) 시간: {df['Datetime'].max()}")
print(f"   - 최종 NVI 수치: {df['Actual_NVI'].iloc[-1]}")