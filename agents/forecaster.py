import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from state import PipelineState

class ForecasterAgent:
    """NVI 예측 에이전트.
    
    mode="baseline": 무대응(Do Nothing) 시나리오 — 향후 Company_Action_Type=0 고정
    mode="mitigated": 전략 적용 시나리오 — strategist_timeline 이벤트를 시계열에 주입
    """
    def __init__(self, mode="baseline"):
        self.mode = mode

    def run(self, state: PipelineState) -> dict:
        mode_label = "무대응(Baseline)" if self.mode == "baseline" else "전략 적용(Mitigated)"
        print(f"\n▶️ [Forecaster-{self.mode}] {mode_label} NVI 시뮬레이션 가동...")
        
        # 1. 현재 시점까지의 데이터셋 즉석 학습
        df = pd.read_csv(state["input_csv_path"])
        model, features = self._train(df)
        
        # 2. 모드에 따라 미래 시나리오 분기
        if self.mode == "baseline":
            # 무대응: 아무런 대응 이벤트 없이 예측
            forecast = self._predict(model, df, features, events=[])
            actual_history = df['Actual_NVI'].tolist()
            print(f"   ✅ 무대응 시뮬레이션 완료 (예상 최저점: {min(forecast):.3f})")
            return {
                "nvi_baseline_forecast": forecast,
                "actual_nvi_history": actual_history
            }
        else:
            # 전략 적용: strategist_timeline 이벤트를 시계열에 주입
            events = state.get("strategist_timeline", [])
            forecast = self._predict(model, df, features, events=events)
            print(f"   ✅ 전략 적용 시뮬레이션 완료 (예상 최저점: {min(forecast):.3f})")
            return {
                "nvi_mitigated_forecast": forecast
            }

    def _train(self, df: pd.DataFrame):
        """과거 데이터로 LightGBM 모델 학습"""
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        issue_start = df['Datetime'].min()
        df['Hours_Since_Start'] = (df['Datetime'] - issue_start).dt.total_seconds() / 3600
        
        features = [
            'Hours_Since_Start', 'Company_Action_Type', 'Influencer_Impact', 
            'Negative_Ratio', 'Mockery_Index', 'Advocate_Ratio', 'Negative_Momentum'
        ]
        
        X_train = df[features]
        y_train = df['Actual_NVI']
        
        model = LGBMRegressor(
            n_estimators=100, learning_rate=0.05, 
            random_state=42, verbose=-1, min_data_in_leaf=1
        )
        model.fit(X_train, y_train)
        return model, features

    def _predict(self, model, df: pd.DataFrame, features: list, events: list) -> list:
        """미래 72시간 NVI 예측.
        
        감성 지표에 '동적 감쇠(Dynamic Decay)'를 적용하여,
        시간 흐름에 따른 자연 진화 + 기업 대응에 따른 여론 회복을 시뮬레이션합니다.
        """
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        issue_start = df['Datetime'].min()
        df['Hours_Since_Start'] = (df['Datetime'] - issue_start).dt.total_seconds() / 3600
        
        current_hour = int(df['Hours_Since_Start'].max())
        future_hours = np.arange(current_hour + 1, current_hour + 73)
        
        test_X = pd.DataFrame(index=range(72), columns=features)
        test_X['Hours_Since_Start'] = future_hours
        
        # 1. 이벤트(대응 액션) 먼저 투영 (기본값: 무대응)
        test_X['Company_Action_Type'] = 0
        test_X['Influencer_Impact'] = 0
        
        for evt in events:
            start_idx = current_hour + evt['hour_offset']
            mask = test_X['Hours_Since_Start'] >= start_idx
            test_X.loc[mask, 'Company_Action_Type'] = evt['action_type']
            test_X.loc[mask, 'Influencer_Impact'] = evt['influencer_hit']
        
        # 2. 🎯 미래 감성 지표 '동적 감쇠(Dynamic Decay)' 시뮬레이션
        #    시간이 지나면 관심도가 자연 하락하고, 기업 대응에 따라 여론이 회복됩니다.
        current_neg_ratio = float(df.iloc[-1]['Negative_Ratio'])
        current_mockery = float(df.iloc[-1]['Mockery_Index'])
        current_advocate = float(df.iloc[-1]['Advocate_Ratio'])
        current_momentum = float(df.iloc[-1]['Negative_Momentum'])
        
        for idx in test_X.index:
            action = test_X.loc[idx, 'Company_Action_Type']
            
            # ① 모멘텀은 시간이 지날수록 급격히 0으로 수렴 (관심도 자연 하락)
            current_momentum = current_momentum * 0.6
            
            # ② 자연 감쇠: 아무 대응 없어도 시간이 지나면 여론은 약간씩 진정
            current_neg_ratio = max(0.3, current_neg_ratio * 0.995)
            current_mockery = max(0.03, current_mockery * 0.99)
            current_advocate = min(0.4, current_advocate * 1.002)
            
            # ③ 기업 대응(Action)에 따른 여론 회복 가속
            if action == 2:  # 전면 사과/리콜 — 강력한 여론 회복
                current_neg_ratio = max(0.3, current_neg_ratio * 0.95)
                current_mockery = max(0.03, current_mockery * 0.90)
                current_advocate = min(0.6, current_advocate * 1.05)
            elif action == 1:  # 1차 해명 — 미미한 효과, 조롱만 약간 감소
                current_mockery = max(0.03, current_mockery * 0.98)
            
            test_X.loc[idx, 'Negative_Ratio'] = current_neg_ratio
            test_X.loc[idx, 'Mockery_Index'] = current_mockery
            test_X.loc[idx, 'Advocate_Ratio'] = current_advocate
            test_X.loc[idx, 'Negative_Momentum'] = current_momentum
        
        # 강제 형변환 (LightGBM mixed-type 에러 방지)
        test_X = test_X.astype(float)
        
        forecast = model.predict(test_X)
        return np.clip(forecast, 0.1, 1.0).round(3).tolist()