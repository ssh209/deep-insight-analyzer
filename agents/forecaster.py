import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from state import PipelineState

class ForecasterAgent:
    def __init__(self):
        pass

    def run(self, state: PipelineState) -> dict:
        print("\n▶️ [Agent 2] 실시간 데이터 트렌드 동적 모델링 가동...")
        
        # 1. 인풋으로 전달된 현재 시점까지의 데이터셋 즉석 학습
        df = pd.read_csv(state["input_csv_path"])
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        issue_start = df['Datetime'].min()
        df['Hours_Since_Start'] = (df['Datetime'] - issue_start).dt.total_seconds() / 3600
        
        X_train = df[['Hours_Since_Start', 'Company_Action', 'Influencer_Hit']]
        y_train = df['Actual_NVI']
        
        model = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1, min_data_in_leaf=1)
        model.fit(X_train, y_train)
        
        # 2. 현재 이후 향후 72시간의 시계열 캔버스 매트릭스 구성
        current_hour = int(df['Hours_Since_Start'].max())
        future_hours = np.arange(current_hour + 1, current_hour + 73)
        
        test_X = pd.DataFrame({
            'Hours_Since_Start': future_hours, 
            'Company_Action': df.iloc[-1]['Company_Action'], 
            'Influencer_Hit': df.iloc[-1]['Influencer_Hit']
        })
        
        # 3. Agent 1이 정의한 시나리오 변곡점 오프셋 투영
        for evt in state["timeline_events"]:
            start_idx = current_hour + evt['hour_offset'] 
            mask = test_X['Hours_Since_Start'] >= start_idx
            test_X.loc[mask, 'Company_Action'] = evt['action_type']
            test_X.loc[mask, 'Influencer_Hit'] = evt['influencer_hit']
                
        # 4. 미래 NVI 시뮬레이션 값 추론
        forecast = model.predict(test_X)
        forecast_list = np.clip(forecast, 0.1, 1.0).round(3).tolist()
        
        print(f"   ✅ 미래 NVI 시뮬레이션 완료 (예상 최저점: {min(forecast_list):.3f})")
        return {"nvi_forecast": forecast_list}