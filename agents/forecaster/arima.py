"""
ArimaForecasterAgent — SARIMAX 기반 NVI 예측 (통계적 베이스라인)

statsmodels의 SARIMAX를 사용한 전통 통계 모델 기반 예측.
다른 모델(LightGBM, TFT, MOIRAI) 대비 벤치마크 베이스라인으로 활용.

특징:
  - GPU 불필요, 학습 데이터 소량 OK
  - 신뢰구간 자동 제공
  - 외생변수 제한적 (ARIMAX로 Company_Action_Type만 투입)
  - 비선형 패턴(급변) 포착 불가 → NVI 예측 정밀도 낮음

출력 형식 (TFT/MOIRAI와 동일):
  {"point": [0.42, 0.40, ...], "lower": [0.38, 0.36, ...], "upper": [0.46, 0.44, ...]}

의존: statsmodels
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

FORECAST_HORIZON = 168  # 1주 (기존과 동일)

# SARIMAX 기본 하이퍼파라미터
# (p,d,q): AR=2, 차분=1, MA=1
ARIMA_ORDER = (2, 1, 1)
# 외생변수: Company_Action_Type만 사용 (범주형이지만 0/1/2 정수로 투입)
EXOG_COLUMNS = ["Company_Action_Type"]


class ArimaForecasterAgent:
    """SARIMAX 기반 NVI 예측 에이전트 (통계적 베이스라인).

    mode="baseline": 무대응 시나리오 (Company_Action_Type=0 고정)
    mode="mitigated": 전략 적용 시나리오 (strategist_timeline 반영)

    LightGBM/TFT/MOIRAI ForecasterAgent와 동일한 run() 인터페이스 유지.
    """

    def __init__(self, mode="baseline"):
        self.mode = mode

    def run(self, state: dict) -> dict:
        crisis_type = state.get("crisis_type", "accidental")
        mode_label = "Baseline" if self.mode == "baseline" else "Mitigated"
        print(f"\n>> [ARIMA-Forecaster-{self.mode}] {mode_label} NVI prediction start...")
        print(f"   Crisis Type: {crisis_type}")
        print(f"   Order: {ARIMA_ORDER}")

        # 1. 입력 데이터 로드
        input_df = pd.read_csv(state["input_csv_path"])
        input_df["Datetime"] = pd.to_datetime(input_df["Datetime"])

        # 2. 미래 외생변수 구성
        events = [] if self.mode == "baseline" else state.get("strategist_timeline", [])
        future_exog = self._build_future_exog(input_df, events)

        # 3. SARIMAX 학습 + 예측
        forecast_result = self._fit_and_predict(input_df, future_exog)

        # 4. 결과 반환
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
    # 미래 외생변수 구성
    # ==========================================
    def _build_future_exog(self, df: pd.DataFrame, events: list) -> pd.DataFrame:
        """미래 168시간의 외생변수를 구성."""
        current_hour = int(df["Hours_Since_Start"].max()) if "Hours_Since_Start" in df.columns else len(df)
        future_hours = np.arange(current_hour + 1, current_hour + FORECAST_HORIZON + 1)

        future_exog = pd.DataFrame({
            "Hours_Since_Start": future_hours,
            "Company_Action_Type": 0,
        })

        for evt in events:
            start_hour = current_hour + evt.get("hour_offset", 0)
            mask = future_exog["Hours_Since_Start"] >= start_hour
            future_exog.loc[mask, "Company_Action_Type"] = evt.get("action_type", 0)

        return future_exog[EXOG_COLUMNS]

    # ==========================================
    # SARIMAX 학습 + 예측
    # ==========================================
    def _fit_and_predict(self, df: pd.DataFrame, future_exog: pd.DataFrame) -> dict:
        """SARIMAX 학습 후 미래 예측.

        Returns:
            {"point": list, "lower": list, "upper": list}
        """
        endog = df["Actual_NVI"].values
        exog = df[EXOG_COLUMNS].values if EXOG_COLUMNS[0] in df.columns else None

        try:
            model = SARIMAX(
                endog,
                exog=exog,
                order=ARIMA_ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)
            print(f"   [FIT] AIC: {result.aic:.1f}, BIC: {result.bic:.1f}")

            # 미래 예측 + 신뢰구간
            forecast = result.get_forecast(
                steps=FORECAST_HORIZON,
                exog=future_exog.values,
            )

            point = np.clip(forecast.predicted_mean, 0.1, 1.0).round(3).tolist()
            conf_int = forecast.conf_int(alpha=0.2)  # 80% 신뢰구간 (q=0.1~0.9에 대응)
            lower = np.clip(conf_int[:, 0], 0.1, 1.0).round(3).tolist()
            upper = np.clip(conf_int[:, 1], 0.1, 1.0).round(3).tolist()

        except Exception as e:
            # ARIMA 수렴 실패 시 naive fallback (마지막 값 유지)
            print(f"   [WARN] SARIMAX fit failed: {e}")
            print(f"   [WARN] Falling back to naive forecast (last value)")
            last_val = float(endog[-1]) if len(endog) > 0 else 0.5
            point = [round(last_val, 3)] * FORECAST_HORIZON
            lower = [round(max(0.1, last_val - 0.1), 3)] * FORECAST_HORIZON
            upper = [round(min(1.0, last_val + 0.1), 3)] * FORECAST_HORIZON

        return {"point": point, "lower": lower, "upper": upper}
