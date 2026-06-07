"""
agents/forecaster 패키지.

ForecasterAgent를 패키지 레벨에서 바로 import 가능하도록 노출.
  from agents.forecaster import ForecasterAgent
"""
from agents.forecaster.forecaster import ForecasterAgent

__all__ = ["ForecasterAgent"]
