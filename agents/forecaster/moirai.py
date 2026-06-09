"""
MoiraiForecasterAgent — MOIRAI 2.0 Foundation Model 기반 NVI 예측

Salesforce의 MOIRAI 2.0 (decoder-only time-series foundation model)을 사용한
zero-shot 또는 fine-tuned NVI 예측 에이전트.

LightGBM/TFT ForecasterAgent와 동일한 run(state) -> dict 인터페이스.

특징:
  - Zero-shot 가능 (학습 데이터 불필요)
  - Quantile loss → 확률적 예측 (신뢰구간 자동 제공)
  - Decoder-only → 이전 MOIRAI 대비 2배 빠르고 30배 작음
  - 외생변수 → past_feat_dynamic_real로 전달 (Known Future 구분 없음)

제약:
  - Known Future vs Observed 구분 불가 (TFT 대비 약점)
  - SCCT 감쇠 파라미터 직접 통합 어려움 → 후처리로 보완
  - GPU 필수

출력 형식:
  {"point": [0.42, 0.40, ...], "lower": [0.38, 0.36, ...], "upper": [0.46, 0.44, ...]}

의존: uni2ts, gluonts, torch
"""
import os
import warnings

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 설정
# ==========================================
# HuggingFace 모델 ID (small/base/large)
MOIRAI_MODEL_ID = os.environ.get("MOIRAI_MODEL_ID", "Salesforce/moirai-2.0-R-small")
CONTEXT_LENGTH = 720       # 과거 참조 길이 (최대 720시간 = 30일)
FORECAST_HORIZON = 168     # 예측 구간 (1주, 기존과 동일)
BATCH_SIZE = 32

# 과거 피처 (past_feat_dynamic_real로 전달)
PAST_FEATURES = [
    "Negative_Ratio", "Mockery_Index", "Advocate_Ratio",
    "Negative_Momentum", "Company_Action_Type", "Influencer_Impact",
]


class MoiraiForecasterAgent:
    """MOIRAI 2.0 기반 NVI 예측 에이전트.

    mode="baseline": 무대응 시나리오
    mode="mitigated": 전략 적용 시나리오

    LightGBM/TFT ForecasterAgent와 동일한 run() 인터페이스 유지.
    """

    def __init__(self, mode="baseline"):
        self.mode = mode
        self.model = None

    def run(self, state: dict) -> dict:
        crisis_type = state.get("crisis_type", "accidental")
        mode_label = "Baseline" if self.mode == "baseline" else "Mitigated"
        print(f"\n>> [MOIRAI-Forecaster-{self.mode}] {mode_label} NVI prediction start...")
        print(f"   Crisis Type: {crisis_type}")
        print(f"   Model: {MOIRAI_MODEL_ID}")

        # 1. 모델 로드
        self._load_model()

        # 2. 입력 데이터 준비
        input_df = pd.read_csv(state["input_csv_path"])
        input_df["Datetime"] = pd.to_datetime(input_df["Datetime"])

        # 3. 모드별 과거 피처 구성
        if self.mode == "mitigated":
            events = state.get("strategist_timeline", [])
            input_df = self._apply_events(input_df, events)

        # 4. 예측
        forecast_result = self._predict(input_df)

        # 5. 결과 반환
        if self.mode == "baseline":
            actual_history = input_df["Actual_NVI"].tolist()
            point = forecast_result["point"]
            print(f"   [OK] Baseline done (min NVI: {min(point):.3f}, "
                  f"CI: [{min(forecast_result['lower']):.3f}, {max(forecast_result['upper']):.3f}])")
            return {
                "nvi_baseline_forecast": forecast_result,
                "actual_nvi_history": actual_history,
            }
        else:
            point = forecast_result["point"]
            print(f"   [OK] Mitigated done (min NVI: {min(point):.3f}, "
                  f"CI: [{min(forecast_result['lower']):.3f}, {max(forecast_result['upper']):.3f}])")
            return {
                "nvi_mitigated_forecast": forecast_result,
            }

    # ==========================================
    # 모델 로드
    # ==========================================
    def _load_model(self):
        """HuggingFace에서 MOIRAI 2.0 모델 로드 (첫 호출 시 다운로드)."""
        if self.model is not None:
            return

        print(f"   [LOAD] Downloading/loading {MOIRAI_MODEL_ID}...")
        module = MoiraiModule.from_pretrained(MOIRAI_MODEL_ID)

        self.model = MoiraiForecast(
            module=module,
            prediction_length=FORECAST_HORIZON,
            context_length=CONTEXT_LENGTH,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=len(PAST_FEATURES),
        )
        print(f"   [OK] Model loaded (context={CONTEXT_LENGTH}, horizon={FORECAST_HORIZON})")

    # ==========================================
    # 이벤트(대응 액션) 적용
    # ==========================================
    def _apply_events(self, df: pd.DataFrame, events: list) -> pd.DataFrame:
        """Strategist 타임라인 이벤트를 입력 데이터에 반영.

        MOIRAI는 Known Future를 직접 지원하지 않으므로,
        과거 데이터의 마지막 부분에 이벤트를 반영하여 모델이 패턴을 읽도록 함.
        """
        df = df.copy()
        if not events:
            return df

        # 이벤트를 시간 기준으로 과거 데이터에 투영
        max_hour = df["Hours_Since_Start"].max() if "Hours_Since_Start" in df.columns else len(df)
        for evt in events:
            target_hour = max_hour + evt.get("hour_offset", 0)
            # 과거 데이터에 이벤트 흔적 남기기 (MOIRAI가 패턴으로 학습)
            mask = df["Hours_Since_Start"] >= target_hour
            if mask.any():
                df.loc[mask, "Company_Action_Type"] = evt.get("action_type", 0)
                df.loc[mask, "Influencer_Impact"] = evt.get("influencer_hit", 0)

        return df

    # ==========================================
    # MOIRAI 예측
    # ==========================================
    def _predict(self, df: pd.DataFrame) -> dict:
        """MOIRAI 2.0으로 NVI 예측 수행.

        Returns:
            {"point": list, "lower": list, "upper": list}
        """
        # GluonTS PandasDataset 구성
        ts_df = df.set_index("Datetime")[["Actual_NVI"] + PAST_FEATURES].copy()
        ts_df.index = pd.DatetimeIndex(ts_df.index, freq="h")

        # PandasDataset 생성
        ds = PandasDataset.from_long_dataframe(
            ts_df.reset_index().assign(item_id="nvi"),
            target="Actual_NVI",
            timestamp="Datetime",
            item_id="item_id",
            feat_dynamic_real=PAST_FEATURES,
        )

        # 예측 수행
        predictor = self.model.create_predictor(batch_size=BATCH_SIZE)
        forecasts = list(predictor.predict(ds))

        if not forecasts:
            # fallback
            empty = [0.5] * FORECAST_HORIZON
            return {"point": empty, "lower": empty, "upper": empty}

        forecast = forecasts[0]

        # Quantile 추출
        point = np.clip(forecast.quantile(0.5), 0.1, 1.0).round(3).tolist()
        lower = np.clip(forecast.quantile(0.1), 0.1, 1.0).round(3).tolist()
        upper = np.clip(forecast.quantile(0.9), 0.1, 1.0).round(3).tolist()

        return {"point": point, "lower": lower, "upper": upper}
