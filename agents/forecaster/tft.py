"""
TFTForecasterAgent — Temporal Fusion Transformer 기반 NVI 예측

LightGBM ForecasterAgent와 동일한 인터페이스(run(state) -> dict)를 제공하되,
Direct multi-horizon 예측 + Quantile 기반 신뢰구간을 반환합니다.

피처 분류 (TFT의 핵심 차별점):
  Known Future:  Company_Action_Type, Influencer_Impact
  Observed:      Negative_Ratio, Mockery_Index, Advocate_Ratio, Negative_Momentum
  Static:        crisis_type (victim / accidental / preventable)
  Target:        Actual_NVI

출력 형식:
  {"point": [0.42, 0.40, ...], "lower": [0.38, 0.36, ...], "upper": [0.46, 0.44, ...]}
  - point: 중앙값(q=0.5)
  - lower: 하한(q=0.1)
  - upper: 상한(q=0.9)

의존: pytorch-forecasting, pytorch-lightning
"""
import os
import warnings

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

warnings.filterwarnings("ignore", category=UserWarning)

from config import TFT_MODEL_PATH
TFT_CHECKPOINT = TFT_MODEL_PATH

# ==========================================
# 피처 정의
# ==========================================
TARGET = "Actual_NVI"

# TFT가 미래까지 알고 있는 변수 (Strategist 타임라인으로 제공)
KNOWN_FUTURE_REALS = ["Hours_Since_Start"]
KNOWN_FUTURE_CATEGORICALS = ["Company_Action_Type", "Influencer_Impact"]

# 과거만 관측 가능한 변수
OBSERVED_REALS = ["Negative_Ratio", "Mockery_Index", "Advocate_Ratio", "Negative_Momentum"]

# 시계열 간 고정 속성
STATIC_CATEGORICALS = ["crisis_type"]

# SCCT 위기 유형 → 정수 인코딩
CRISIS_TYPE_MAP = {"victim": 0, "accidental": 1, "preventable": 2}

FORECAST_HORIZON = 168  # 1주 (기존 LightGBM과 동일)
MAX_ENCODER_LENGTH = 72  # 과거 72시간 참조


class TFTForecasterAgent:
    """Temporal Fusion Transformer 기반 NVI 예측 에이전트.

    mode="baseline": 무대응 시나리오 (Company_Action_Type=0 고정)
    mode="mitigated": 전략 적용 시나리오 (strategist_timeline 이벤트 주입)

    LightGBM ForecasterAgent와 동일한 run() 인터페이스 유지.
    """

    def __init__(self, mode="baseline"):
        self.mode = mode
        self.model = None

    def run(self, state: dict) -> dict:
        crisis_type = state.get("crisis_type", "accidental")
        mode_label = "Baseline" if self.mode == "baseline" else "Mitigated"
        print(f"\n>> [TFT-Forecaster-{self.mode}] {mode_label} NVI prediction start...")
        print(f"   Crisis Type: {crisis_type}")

        # 1. 모델 로드
        self._load_model()

        # 2. 입력 데이터 준비
        input_df = pd.read_csv(state["input_csv_path"])
        input_df["Datetime"] = pd.to_datetime(input_df["Datetime"])
        issue_start = input_df["Datetime"].min()
        input_df["Hours_Since_Start"] = (
            (input_df["Datetime"] - issue_start).dt.total_seconds() / 3600
        )

        # 3. 미래 피처 구성
        events = [] if self.mode == "baseline" else state.get("strategist_timeline", [])
        future_df = self._build_future_features(input_df, events, crisis_type)

        # 4. 예측
        forecast_result = self._predict(input_df, future_df, crisis_type)

        # 5. 결과 반환 (LightGBM과 동일한 키)
        if self.mode == "baseline":
            actual_history = input_df[TARGET].tolist()
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
        """학습된 TFT 체크포인트를 로드."""
        if self.model is not None:
            return

        checkpoint_path = os.path.normpath(TFT_CHECKPOINT)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"TFT 체크포인트를 찾을 수 없습니다: {checkpoint_path}\n"
                f"먼저 python scripts/train_tft.py 를 실행하세요."
            )

        self.model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        print(f"   [LOAD] TFT model: {checkpoint_path}")

    # ==========================================
    # 미래 피처 구성
    # ==========================================
    def _build_future_features(
        self, df: pd.DataFrame, events: list, crisis_type: str
    ) -> pd.DataFrame:
        """미래 168시간의 Known Future 피처를 구성."""
        current_hour = int(df["Hours_Since_Start"].max())
        future_hours = np.arange(current_hour + 1, current_hour + FORECAST_HORIZON + 1)

        future_df = pd.DataFrame({
            "Hours_Since_Start": future_hours,
            "Company_Action_Type": 0,
            "Influencer_Impact": 0,
            "crisis_type": CRISIS_TYPE_MAP.get(crisis_type, 1),
        })

        # 이벤트(대응 액션) 투영
        for evt in events:
            start_hour = current_hour + evt["hour_offset"]
            mask = future_df["Hours_Since_Start"] >= start_hour
            future_df.loc[mask, "Company_Action_Type"] = evt["action_type"]
            future_df.loc[mask, "Influencer_Impact"] = evt["influencer_hit"]

        return future_df

    # ==========================================
    # TFT 예측
    # ==========================================
    def _predict(
        self, history_df: pd.DataFrame, future_df: pd.DataFrame, crisis_type: str
    ) -> dict:
        """TFT 모델로 Direct multi-horizon 예측 수행.

        Returns:
            {"point": list, "lower": list, "upper": list}
        """
        crisis_type_int = CRISIS_TYPE_MAP.get(crisis_type, 1)

        # 과거 + 미래 결합 (TFT는 encoder+decoder 입력 필요)
        history = history_df.copy()
        history["time_idx"] = range(len(history))
        history["series_id"] = "current"
        history["crisis_type"] = crisis_type_int

        # 미래 프레임 (target=NaN, observed=NaN)
        future = future_df.copy()
        future["time_idx"] = range(len(history), len(history) + len(future))
        future["series_id"] = "current"
        future[TARGET] = 0.0  # placeholder (예측 대상)
        for col in OBSERVED_REALS:
            if col not in future.columns:
                future[col] = 0.0  # observed 변수는 미래에 없으므로 0

        combined = pd.concat([history, future], ignore_index=True)

        # 카테고리컬 타입 변환
        combined["Company_Action_Type"] = combined["Company_Action_Type"].astype(int).astype(str)
        combined["Influencer_Impact"] = combined["Influencer_Impact"].astype(int).astype(str)
        combined["crisis_type"] = combined["crisis_type"].astype(int).astype(str)
        combined["series_id"] = combined["series_id"].astype(str)

        # TimeSeriesDataSet 구성 (prediction mode)
        dataset = TimeSeriesDataSet(
            combined,
            time_idx="time_idx",
            target=TARGET,
            group_ids=["series_id"],
            min_encoder_length=min(MAX_ENCODER_LENGTH, len(history)),
            max_encoder_length=MAX_ENCODER_LENGTH,
            min_prediction_length=1,
            max_prediction_length=FORECAST_HORIZON,
            static_categoricals=STATIC_CATEGORICALS,
            time_varying_known_categoricals=KNOWN_FUTURE_CATEGORICALS,
            time_varying_known_reals=KNOWN_FUTURE_REALS,
            time_varying_unknown_reals=[TARGET] + OBSERVED_REALS,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        # 예측 수행
        with torch.no_grad():
            predictions = self.model.predict(
                dataloader,
                mode="quantiles",
                return_x=False,
            )

        # quantile 결과 파싱 (shape: [1, horizon, n_quantiles])
        # pytorch-forecasting 기본 quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        pred_np = predictions.cpu().numpy()

        if pred_np.ndim == 3:
            # [batch, horizon, quantiles]
            lower = np.clip(pred_np[0, :, 1], 0.1, 1.0).round(3).tolist()  # q=0.1
            point = np.clip(pred_np[0, :, 3], 0.1, 1.0).round(3).tolist()  # q=0.5
            upper = np.clip(pred_np[0, :, 5], 0.1, 1.0).round(3).tolist()  # q=0.9
        else:
            # fallback: 단일 quantile
            point = np.clip(pred_np.flatten(), 0.1, 1.0).round(3).tolist()
            lower = point
            upper = point

        return {"point": point, "lower": lower, "upper": upper}
