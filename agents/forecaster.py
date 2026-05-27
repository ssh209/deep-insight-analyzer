import os
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from state import PipelineState

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "nvi_forecaster.pkl")

# ==========================================
# 📊 SCCT 위기 유형별 감쇠 파라미터 프리셋
# Coombs (2007) Situational Crisis Communication Theory 기반
#
# 학술 근거:
#   [1] Coombs, W.T. (2007). Corporate Reputation Review, 10(3), 163-176.
#   [2] Dong et al. (2020). Expert Systems with Applications, 148, 113268.
#   [3] Antonetti & Maklan (2016). Journal of Business Ethics, 135(3), 429-444.
# ==========================================
CRISIS_DECAY_PARAMS = {
    "victim": {
        # 피해자형 (자연재해, 루머, 외부 범행)
        # 책임 귀인 낮음 → 관심 빠르게 소멸, 사과 효과 강력
        "label": "Victim",
        "momentum_decay": 0.45,
        "neg_ratio_decay": 0.98,
        "neg_ratio_floor": 0.25,
        "mockery_decay": 0.95,
        "mockery_floor": 0.02,
        "advocate_growth": 1.01,
        "advocate_ceiling": 0.5,
        "action_1_mockery": 0.95,
        "action_2_neg": 0.90,
        "action_2_mockery": 0.85,
        "action_2_advocate": 1.08,
        "action_2_advocate_ceiling": 0.7,
    },
    "accidental": {
        # 사고형 (리콜, 기술적 결함, 장비 고장)
        # 책임 귀인 보통 → 자연 감쇠 보통, 사과 효과 보통
        "label": "Accidental",
        "momentum_decay": 0.60,
        "neg_ratio_decay": 0.995,
        "neg_ratio_floor": 0.30,
        "mockery_decay": 0.99,
        "mockery_floor": 0.03,
        "advocate_growth": 1.002,
        "advocate_ceiling": 0.4,
        "action_1_mockery": 0.98,
        "action_2_neg": 0.95,
        "action_2_mockery": 0.90,
        "action_2_advocate": 1.05,
        "action_2_advocate_ceiling": 0.6,
    },
    "preventable": {
        # 예방가능형 (경영진 비리, 안전 규정 위반, 의도적 은폐)
        # 책임 귀인 높음 → Sticky Crisis, 도덕적 분노로 감쇠 저항
        "label": "Preventable",
        "momentum_decay": 0.75,
        "neg_ratio_decay": 0.999,
        "neg_ratio_floor": 0.40,
        "mockery_decay": 0.998,
        "mockery_floor": 0.05,
        "advocate_growth": 1.0005,
        "advocate_ceiling": 0.25,
        "action_1_mockery": 0.995,
        "action_2_neg": 0.98,
        "action_2_mockery": 0.96,
        "action_2_advocate": 1.02,
        "action_2_advocate_ceiling": 0.35,
    },
}

class ForecasterAgent:
    """NVI 예측 에이전트.
    
    mode="baseline": 무대응(Do Nothing) 시나리오 — 향후 Company_Action_Type=0 고정
    mode="mitigated": 전략 적용 시나리오 — strategist_timeline 이벤트를 시계열에 주입
    
    감쇠 파라미터는 PipelineState의 crisis_type에 따라 CRISIS_DECAY_PARAMS에서 동적으로 로드됩니다.
    """
    def __init__(self, mode="baseline"):
        self.mode = mode

    def run(self, state: PipelineState) -> dict:
        crisis_type = state.get("crisis_type", "accidental")
        params = CRISIS_DECAY_PARAMS.get(crisis_type, CRISIS_DECAY_PARAMS["accidental"])
        
        mode_label = "Baseline" if self.mode == "baseline" else "Mitigated"
        print(f"\n>> [Forecaster-{self.mode}] {mode_label} NVI simulation start...")
        print(f"   Crisis Type: {params['label']} (momentum_decay={params['momentum_decay']})")
        
        # 1. 사전 학습 모델 로드 (pkl 없으면 train_csv로 학습 후 자동 저장)
        train_path = state.get("train_csv_path", state["input_csv_path"])
        model, features = self._load_or_train(train_path)
        
        # 2. 실전 입력 데이터(72h 현재 위기)로 예측 출발점 결정
        input_df = pd.read_csv(state["input_csv_path"])
        
        # 3. 모드에 따라 미래 시나리오 분기
        if self.mode == "baseline":
            # 무대응: 아무런 대응 이벤트 없이 예측
            forecast = self._predict(model, input_df, features, events=[], params=params)
            actual_history = input_df['Actual_NVI'].tolist()
            print(f"   [OK] Baseline done (min NVI: {min(forecast):.3f})")
            return {
                "nvi_baseline_forecast": forecast,
                "actual_nvi_history": actual_history
            }
        else:
            # 전략 적용: strategist_timeline 이벤트를 시계열에 주입
            events = state.get("strategist_timeline", [])
            forecast = self._predict(model, input_df, features, events=events, params=params)
            print(f"   [OK] Mitigated done (min NVI: {min(forecast):.3f})")
            return {
                "nvi_mitigated_forecast": forecast
            }

    def _load_or_train(self, train_csv_path: str):
        """models/nvi_forecaster.pkl 로드. 없으면 학습 후 자동 저장."""
        model_path = os.path.normpath(MODEL_PATH)
        
        if os.path.exists(model_path):
            bundle = joblib.load(model_path)
            print(f"   [LOAD] Pre-trained model: {model_path}")
            return bundle["model"], bundle["features"]
        
        # pkl 없으면 학습 후 저장
        print(f"   [TRAIN] No pre-trained model found. Training from {train_csv_path}...")
        df = pd.read_csv(train_csv_path)
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
            n_estimators=200, learning_rate=0.05, max_depth=6,
            num_leaves=31, random_state=42, verbose=-1, min_data_in_leaf=3
        )
        model.fit(X_train, y_train)
        
        # 자동 저장
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({"model": model, "features": features}, model_path)
        print(f"   [SAVE] Model saved: {model_path}")
        
        return model, features

    def _predict(self, model, df: pd.DataFrame, features: list, events: list, params: dict) -> list:
        """미래 168시간(1주) NVI 예측.
        
        감성 지표에 SCCT 위기 유형별 '동적 감쇠(Dynamic Decay)'를 적용하여,
        시간 흐름에 따른 자연 진화 + 기업 대응에 따른 여론 회복을 시뮬레이션합니다.
        
        감쇠 속도는 params(CRISIS_DECAY_PARAMS)에 의해 결정됩니다:
          - victim: 빠른 감쇠 (관심 급감, 사과 효과 강력)
          - accidental: 보통 감쇠 (기본값)
          - preventable: 느린 감쇠 (Sticky Crisis, 사과 효과 미미)
        """
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        issue_start = df['Datetime'].min()
        df['Hours_Since_Start'] = (df['Datetime'] - issue_start).dt.total_seconds() / 3600
        
        current_hour = int(df['Hours_Since_Start'].max())
        forecast_horizon = 168  # 1주
        future_hours = np.arange(current_hour + 1, current_hour + forecast_horizon + 1)
        
        test_X = pd.DataFrame(index=range(forecast_horizon), columns=features)
        test_X['Hours_Since_Start'] = future_hours
        
        # 1. 이벤트(대응 액션) 먼저 투영 (기본값: 무대응)
        test_X['Company_Action_Type'] = 0
        test_X['Influencer_Impact'] = 0
        
        for evt in events:
            start_idx = current_hour + evt['hour_offset']
            mask = test_X['Hours_Since_Start'] >= start_idx
            test_X.loc[mask, 'Company_Action_Type'] = evt['action_type']
            test_X.loc[mask, 'Influencer_Impact'] = evt['influencer_hit']
        
        # 2. 🎯 SCCT 위기 유형별 '동적 감쇠(Dynamic Decay)' 시뮬레이션
        current_neg_ratio = float(df.iloc[-1]['Negative_Ratio'])
        current_mockery = float(df.iloc[-1]['Mockery_Index'])
        current_advocate = float(df.iloc[-1]['Advocate_Ratio'])
        current_momentum = float(df.iloc[-1]['Negative_Momentum'])
        
        for idx in test_X.index:
            action = test_X.loc[idx, 'Company_Action_Type']
            
            # ① 모멘텀 감쇠 (위기 유형에 따라 감쇠 속도 차등)
            current_momentum = current_momentum * params["momentum_decay"]
            
            # ② 자연 감쇠 (위기 유형에 따라 감쇠 속도/하한 차등)
            current_neg_ratio = max(params["neg_ratio_floor"], current_neg_ratio * params["neg_ratio_decay"])
            current_mockery = max(params["mockery_floor"], current_mockery * params["mockery_decay"])
            current_advocate = min(params["advocate_ceiling"], current_advocate * params["advocate_growth"])
            
            # ③ 기업 대응(Action)에 따른 여론 회복 가속 (위기 유형에 따라 효과 차등)
            if action == 2:  # 전면 사과/리콜
                current_neg_ratio = max(params["neg_ratio_floor"], current_neg_ratio * params["action_2_neg"])
                current_mockery = max(params["mockery_floor"], current_mockery * params["action_2_mockery"])
                current_advocate = min(params["action_2_advocate_ceiling"], current_advocate * params["action_2_advocate"])
            elif action == 1:  # 1차 해명
                current_mockery = max(params["mockery_floor"], current_mockery * params["action_1_mockery"])
            
            test_X.loc[idx, 'Negative_Ratio'] = current_neg_ratio
            test_X.loc[idx, 'Mockery_Index'] = current_mockery
            test_X.loc[idx, 'Advocate_Ratio'] = current_advocate
            test_X.loc[idx, 'Negative_Momentum'] = current_momentum
        
        # 강제 형변환 (LightGBM mixed-type 에러 방지)
        test_X = test_X.astype(float)
        
        forecast = model.predict(test_X)
        return np.clip(forecast, 0.1, 1.0).round(3).tolist()