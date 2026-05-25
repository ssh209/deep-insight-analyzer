import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# ==========================================
# 2. 노드(Node) 실행 함수: LightGBM 시뮬레이션
# ==========================================
def forecasting_node(state: dict) -> dict:
    print("\n▶️ [Agent 2] LightGBM 여론(NVI) 예측 시뮬레이션 가동...")
    
    # 1. Analyzer(Agent 1)가 넘겨준 파라미터 꺼내기
    params = state.get("extracted_params", {})
    action_type = params.get("action_type", 1)
    impact_weight = params.get("impact_weight", 1.0)
    duration_hours = params.get("duration_hours", 48)
    
    # ---------------------------------------------------------
    # ⚙️ [가상 학습 구간] 실무에서는 미리 학습된 model.pkl을 불러옵니다.
    # 여기서는 PoC를 위해 다양한 위기 상황 패턴을 즉석에서 학습시킵니다.
    # 피처: [경과시간, 대응수준, 충격가중치] / 정답: [NVI 지수]
    # ---------------------------------------------------------
    train_X = np.array([
        [1, 0, 1.0], [24, 0, 1.0], [72, 0, 1.0],  # 무대응 (천천히 하락)
        [1, 1, 2.0], [24, 1, 2.0], [72, 1, 2.0],  # 1차 변명 + 큰 충격 (급격히 하락)
        [1, 2, 1.5], [24, 2, 1.5], [72, 2, 1.5]   # 적극 사과 (초반 하락 후 반등)
    ])
    # 위 상황에 맞는 가상의 정답 NVI
    train_y = np.array([
        0.80, 0.60, 0.40,  
        0.75, 0.30, 0.15,  
        0.70, 0.65, 0.85   
    ])
    
    # 모델 초고속 학습 (n_estimators는 작게 설정)
    model = LGBMRegressor(n_estimators=50, random_state=42)
    model.fit(train_X, train_y)
    # ---------------------------------------------------------

    # 2. 미래 예측용 피처 데이터셋 만들기 (1시간 뒤부터 duration_hours까지)
    future_hours = np.arange(1, duration_hours + 1)
    
    test_X = pd.DataFrame({
        'Hour': future_hours,
        'Action': action_type,         # LLM이 판단한 대응 수준
        'Impact': impact_weight        # LLM이 판단한 파급력
    })
    
    # 3. LightGBM 추론 실행
    forecast_array = model.predict(test_X)
    
    # 4. 현실성 부여 (약간의 지그재그 노이즈 추가 및 범위 0.1~1.0 보정)
    noise = np.random.normal(0, 0.015, duration_hours)
    final_forecast = np.clip(forecast_array + noise, 0.1, 1.0).round(3).tolist()
    
    print("   ✅ LightGBM 시뮬레이션 완료!")
    print(f"   - 시뮬레이션 기간: 향후 {duration_hours}시간")
    print(f"   - 최저 NVI 예상치: {min(final_forecast):.3f}")
    
    # LangGraph 상태(State) 업데이트 반환
    return {"nvi_forecast": final_forecast}


# --- 테스트 실행부 ---
if __name__ == "__main__":
    # 1번 에이전트(LLM)가 뱉어냈던 JSON을 그대로 주입해 봅니다.
    test_state = {
        "extracted_params": {
            'action_type': 1,            # 1차 입장문 발표
            'impact_weight': 1.8,        # 대형 유튜버 공론화 (큰 충격)
            'duration_hours': 72
        }
    }
    
    result = forecasting_node(test_state)
    print(f"\n📊 시간별 NVI 예측 궤적 (처음 10시간):")
    print(result['nvi_forecast'][:10])