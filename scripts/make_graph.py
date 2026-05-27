"""
Baseline Forecast 그래프 단독 실행 스크립트.
LLM / 임베딩 벡터 초기화 없이 ForecasterAgent만 실행하여 NVI 차트를 렌더링합니다.

실행: streamlit run scripts/make_graph.py
"""
import sys
import os

# 프로젝트 루트를 모듈 검색 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
from agents.forecaster import ForecasterAgent, CRISIS_DECAY_PARAMS

st.set_page_config(page_title="NVI Forecast Preview", page_icon="📈", layout="wide")
st.title("📈 NVI Baseline Forecast — Quick Preview")
st.caption("LLM 파이프라인 없이 ForecasterAgent만 단독 실행합니다. (1주 예측)")

# ==========================================
# 사이드바: 파라미터 설정
# ==========================================
with st.sidebar:
    st.header("⚙️ 파라미터")
    
    train_csv = st.text_input("학습 데이터 (720h)", value="data/pr_crisis_dataset.csv")
    input_csv = st.text_input("입력 데이터 (72h)", value="data/input_crisis_72h.csv")
    
    CRISIS_TYPE_OPTIONS = {
        "피해자형 (Victim) — 빠른 회복": "victim",
        "사고형 (Accidental) — 보통 회복": "accidental",
        "예방가능형 (Preventable) — 느린 회복": "preventable",
    }
    crisis_label = st.selectbox(
        "⚠️ 위기 유형 (SCCT)",
        options=list(CRISIS_TYPE_OPTIONS.keys()),
        index=1,
    )
    crisis_type = CRISIS_TYPE_OPTIONS[crisis_label]
    
    # 전체 유형 비교 모드
    compare_all = st.checkbox("🔄 3개 유형 동시 비교", value=False)
    
    st.divider()
    
    # 현재 선택된 프리셋 표시
    params = CRISIS_DECAY_PARAMS[crisis_type]
    st.subheader(f"📋 감쇠 파라미터 ({params['label']})")
    for key, val in params.items():
        if key != "label":
            st.text(f"{key}: {val}")

    run_btn = st.button("🚀 예측 실행", type="primary", use_container_width=True)

# ==========================================
# 실행
# ==========================================
if run_btn:
    for p in [train_csv, input_csv]:
        if not os.path.exists(p):
            st.error(f"파일을 찾을 수 없습니다: `{p}`")
            st.stop()

    if compare_all:
        # ==========================================
        # 3개 유형 동시 비교 모드
        # ==========================================
        st.subheader("📊 SCCT 위기 유형별 NVI 무대응 예측 비교")
        
        results = {}
        actual = None
        
        progress = st.progress(0, text="예측 중...")
        for i, (ctype, cparams) in enumerate(CRISIS_DECAY_PARAMS.items()):
            state = {"train_csv_path": train_csv, "input_csv_path": input_csv, "crisis_type": ctype}
            agent = ForecasterAgent(mode="baseline")
            result = agent.run(state)
            results[cparams["label"]] = result["nvi_baseline_forecast"]
            if actual is None:
                actual = result["actual_nvi_history"]
            progress.progress((i + 1) / 3, text=f"{cparams['label']} 완료")
        
        progress.empty()
        
        # 메트릭 카드
        cols = st.columns(3)
        for i, (label, forecast) in enumerate(results.items()):
            cols[i].metric(f"{label} 최저점", f"{min(forecast):.3f}")
        
        # 차트 구성 (일 단위 x축)
        forecast_len = len(list(results.values())[0])
        total_len = len(actual) + forecast_len
        hours_arr = np.arange(total_len)
        days_arr = hours_arr / 24
        
        chart_data = {"일(Day)": days_arr, "실제 NVI (Actual)": actual + [None] * forecast_len}
        
        colors = ["#3b82f6"]  # 실제: 파란색
        type_colors = {"Victim": "#22c55e", "Accidental": "#f59e0b", "Preventable": "#ef4444"}
        
        for label, forecast in results.items():
            chart_data[label] = [None] * (len(actual) - 1) + [actual[-1]] + forecast
            colors.append(type_colors.get(label, "#888"))
        
        chart_df = pd.DataFrame(chart_data).set_index("일(Day)")
        st.line_chart(chart_df, color=colors)
        
        st.caption("🔵 실제 | 🟢 Victim (빠른 회복) | 🟡 Accidental (보통) | 🔴 Preventable (느린 회복)")
    
    else:
        # ==========================================
        # 단일 유형 모드
        # ==========================================
        state = {"train_csv_path": train_csv, "input_csv_path": input_csv, "crisis_type": crisis_type}
        
        with st.spinner("LightGBM 학습 + Baseline 예측 중..."):
            agent = ForecasterAgent(mode="baseline")
            result = agent.run(state)

        actual = result["actual_nvi_history"]
        baseline = result["nvi_baseline_forecast"]

        # 메트릭 카드
        col1, col2, col3 = st.columns(3)
        col1.metric("현재 NVI", f"{actual[-1]:.3f}")
        col2.metric("무대응 시 최저점", f"{min(baseline):.3f}")
        col3.metric("무대응 시 1주 후", f"{baseline[-1]:.3f}")

        # 차트 (일 단위 x축)
        st.subheader(f"📊 NVI 추이 — {params['label']}")
        
        total_len = len(actual) + len(baseline)
        days_arr = np.arange(total_len) / 24
        
        chart_df = pd.DataFrame({
            "일(Day)": days_arr,
            "실제 NVI (Actual)": actual + [None] * len(baseline),
            "무대응 예측 (Baseline)": [None] * (len(actual) - 1) + [actual[-1]] + baseline,
        }).set_index("일(Day)")

        st.line_chart(chart_df, color=["#3b82f6", "#ef4444"])
        st.caption("🔵 파란색: 과거 실제 여론 | 🔴 빨간색: 무대응(Do Nothing) 예측")

    # 원시 데이터 접기
    with st.expander("📄 예측 원시 데이터"):
        if compare_all:
            raw_data = {"Day": np.arange(1, forecast_len + 1) / 24}
            for label, forecast in results.items():
                raw_data[label] = forecast
            st.dataframe(pd.DataFrame(raw_data), use_container_width=True, hide_index=True)
        else:
            raw_df = pd.DataFrame({
                "Day": np.arange(1, len(baseline) + 1) / 24,
                "Predicted NVI": baseline,
            })
            st.dataframe(raw_df, use_container_width=True, hide_index=True)
