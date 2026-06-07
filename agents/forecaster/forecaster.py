"""
ForecasterAgent — 예측 모델 라우터 (루트 컨트롤러)

PipelineState의 forecaster_model 값에 따라 적절한 서브 에이전트로 라우팅합니다.
사용자 입력으로 모델을 선택하며, config.FORECASTER_MODEL은 기본값으로 사용됩니다.

지원 모델:
  - "lightgbm" (기본): LightGBM + SCCT 감쇠 시뮬레이션
  - "tft":              TFT (Direct multi-horizon + quantile)
  - "moirai":           MOIRAI 2.0 (zero-shot foundation model)
  - "arima":            SARIMAX (통계적 베이스라인)

모든 서브 에이전트는 동일한 출력 형식:
  - LightGBM: list[float] (point forecast만)
  - TFT/MOIRAI/ARIMA: {"point": list, "lower": list, "upper": list}
"""
from config import FORECASTER_MODEL

from agents.forecaster.lightgbm import LightGBMForecasterAgent
from agents.forecaster.tft import TFTForecasterAgent
from agents.forecaster.moirai import MoiraiForecasterAgent
from agents.forecaster.arima import ArimaForecasterAgent

# 모델명 → 에이전트 클래스 매핑
_AGENT_REGISTRY = {
    "lightgbm": LightGBMForecasterAgent,
    "tft": TFTForecasterAgent,
    "moirai": MoiraiForecasterAgent,
    "arima": ArimaForecasterAgent,
}

SUPPORTED_MODELS = list(_AGENT_REGISTRY.keys())


class ForecasterAgent:
    """예측 모델 라우터.

    mode="baseline" 또는 "mitigated"로 생성하고,
    run() 호출 시 state["forecaster_model"]에 따라 적절한 서브 에이전트에 위임.
    """

    def __init__(self, mode="baseline"):
        self.mode = mode
        self._cache = {}  # 모델별 에이전트 캐시 (재생성 방지)

    def run(self, state: dict) -> dict:
        model_name = state.get("forecaster_model", FORECASTER_MODEL)
        agent = self._get_agent(model_name)
        return agent.run(state)

    def _get_agent(self, model_name: str):
        """모델명에 해당하는 서브 에이전트를 반환 (캐시 사용)."""
        if model_name not in self._cache:
            agent_cls = _AGENT_REGISTRY.get(model_name)
            if agent_cls is None:
                print(f"   [WARN] Unknown forecaster model: '{model_name}'. "
                      f"Supported: {SUPPORTED_MODELS}. Falling back to lightgbm.")
                agent_cls = LightGBMForecasterAgent
            self._cache[model_name] = agent_cls(mode=self.mode)
        return self._cache[model_name]
