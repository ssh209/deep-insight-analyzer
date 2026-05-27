import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
import time
import numpy as np
import graphviz
from engine import init_infrastructure, build_graph

# ==========================================
# ⚙️ 백엔드 엔진 로드 및 캐싱
# ==========================================
@st.cache_resource
def setup_backend():
    client, vector_db = init_infrastructure()
    app_graph = build_graph(client, vector_db)
    return app_graph

app_graph = setup_backend()

# ==========================================
# 🔄 파이프라인 노드 정의 (공통 상수)
# ==========================================
PIPELINE_ORDER = [
    "baseline_forecaster", "planner", "strategist", 
    "mitigated_forecaster", "analyst", "compiler", "reviewer"
]

NODE_LABELS = {
    "baseline_forecaster": "Baseline Forecaster",
    "planner": "Report Planner",
    "strategist": "Strategist",
    "mitigated_forecaster": "Mitigated Forecaster",
    "analyst": "Analyst (Gap)",
    "compiler": "Report Compiler",
    "reviewer": "Reviewer (CCO)",
}

NODE_ICONS = {
    "waiting": "⬜",
    "running": "🔵",
    "done": "✅",
    "rejected": "❌",
}

# ==========================================
# 📊 Graphviz 다이어그램 (컴팩트 버전)
# ==========================================
def render_pipeline_graph(completed_nodes, current_nodes, is_rejected=False):
    dot = graphviz.Digraph(engine='dot')
    dot.attr(rankdir='LR')  # 좌→우 방향으로 변경 (컴팩트)
    dot.attr('graph', size='6,1.5', ratio='compress', nodesep='0.3', ranksep='0.4')
    dot.attr('node', fontname='Malgun Gothic', fontsize='9', width='0.6', height='0.4', margin='0.08,0.04')
    dot.attr('edge', arrowsize='0.5')

    nodes = {
        "S": ("", "circle"),
        "BF": ("Baseline\nForecaster", "box"),
        "PL": ("Planner", "box"),
        "ST": ("Strategist", "box"),
        "MF": ("Mitigated\nForecaster", "box"),
        "AN": ("Analyst", "box"),
        "CP": ("Compiler", "box"),
        "RV": ("Reviewer", "box"),
        "E": ("", "doublecircle"),
    }
    
    # 노드 ID → 파이프라인 키 매핑
    id_map = {
        "S": "START", "BF": "baseline_forecaster", "PL": "planner",
        "ST": "strategist", "MF": "mitigated_forecaster", 
        "AN": "analyst", "CP": "compiler", "RV": "reviewer", "E": "END"
    }

    for node_id, (label, shape) in nodes.items():
        pipeline_key = id_map[node_id]
        color = "#e2e8f0"
        fontcolor = "#64748b"
        penwidth = "1"
        
        if pipeline_key in current_nodes:
            if pipeline_key == "reviewer" and is_rejected:
                color = "#fca5a5"
                fontcolor = "#991b1b"
                penwidth = "2"
            else:
                color = "#93c5fd"
                fontcolor = "#1e40af"
                penwidth = "2"
        elif pipeline_key in completed_nodes or (pipeline_key == "START" and completed_nodes):
            color = "#86efac"
            fontcolor = "#166534"

        dot.node(node_id, label, style='filled,rounded', fillcolor=color, 
                 fontcolor=fontcolor, shape=shape, penwidth=penwidth)

    # 순차 엣지
    dot.edge("S", "BF")
    dot.edge("BF", "PL")
    dot.edge("PL", "ST")
    dot.edge("ST", "MF")
    dot.edge("MF", "AN")
    dot.edge("AN", "CP")
    dot.edge("CP", "RV")
    dot.edge("RV", "E", label="OK", fontsize="8", color="#16a34a", fontcolor="#16a34a", fontname='Malgun Gothic')
    dot.edge("RV", "PL", label="NG", fontsize="8", color="#dc2626", fontcolor="#dc2626", 
             fontname='Malgun Gothic', style="dashed", constraint="false")

    return dot

# ==========================================
# 📋 리스트 전이 방식 파이프라인 상태 렌더링
# ==========================================
def render_pipeline_list(completed_nodes, current_nodes, is_rejected=False, loop_count=0):
    """파이프라인 단계를 리스트 형태로 렌더링. 각 노드의 상태를 아이콘으로 표시."""
    lines = []
    if loop_count > 0:
        lines.append(f"🔄 **Loop {loop_count + 1}**")
    
    for node_key in PIPELINE_ORDER:
        label = NODE_LABELS[node_key]
        if node_key in current_nodes:
            if node_key == "reviewer" and is_rejected:
                icon = NODE_ICONS["rejected"]
                lines.append(f"{icon} ~~{label}~~ — 반려")
            else:
                icon = NODE_ICONS["running"]
                lines.append(f"{icon} **{label}** ⏳")
        elif node_key in completed_nodes:
            icon = NODE_ICONS["done"]
            lines.append(f"{icon} {label}")
        else:
            icon = NODE_ICONS["waiting"]
            lines.append(f"{icon} {label}")
    
    return "\n\n".join(lines)

# ==========================================
# 🎨 Streamlit 대시보드 UI
# ==========================================
st.set_page_config(page_title="PR 위기 대응 대시보드", page_icon="🚨", layout="wide")

st.title("🚨 실시간 PR 위기 시뮬레이션 대시보드")
st.markdown("무대응(Do Nothing) vs 전략 적용(Mitigated) **이원화 시뮬레이션** 기반 위기 대응 시스템")

# ==========================================
# 📝 사이드바: 입력 파라미터 + Agent I/O 로그
# ==========================================
with st.sidebar:
    st.header("📝 입력 파라미터")
    target_csv = st.text_input("입력 데이터 (72h 현재 위기)", value="data/input_crisis_72h.csv")
    
    # 🎯 SCCT 위기 유형 선택 (감쇠 파라미터 결정)
    CRISIS_TYPE_OPTIONS = {
        "피해자형 — 자연재해, 루머, 외부 범행 (빠른 회복)": "victim",
        "사고형 — 리콜, 기술적 결함, 장비 고장 (보통 회복)": "accidental",
        "예방가능형 — 경영진 비리, 안전 위반, 의도적 은폐 (느린 회복)": "preventable",
    }
    crisis_type_label = st.selectbox(
        "⚠️ 위기 유형 (SCCT)",
        options=list(CRISIS_TYPE_OPTIONS.keys()),
        index=1,  # 기본값: 사고형
        help="Coombs(2007) SCCT 이론 기반. 위기 유형에 따라 여론 감쇠 속도와 대응 효과가 달라집니다."
    )
    selected_crisis_type = CRISIS_TYPE_OPTIONS[crisis_type_label]
    
    default_meta = "[현재 상황] 초기 대응 지연 및 무대응 기간 유튜버 2연타 타격으로 여론 최악 직면."
    
    input_metadata = st.text_area("위기 메타 정보 및 대응 계획", value=default_meta, height=200)
    start_btn = st.button("🚀 파이프라인 가동", type="primary", use_container_width=True)
    
    st.divider()
    
    # 🎯 Agent I/O 로그 영역 (사이드바 하단)
    st.header("🔍 Agent I/O 로그")
    agent_log_container = st.container()

if start_btn:
    # 초기 상태
    current_state = {
        "train_csv_path": "data/pr_crisis_dataset.csv",
        "input_csv_path": target_csv,
        "crisis_context": input_metadata,
        "crisis_type": selected_crisis_type,
        "actual_nvi_history": [],
        "nvi_baseline_forecast": [],
        "nvi_mitigated_forecast": [],
        "strategist_timeline": [],
        "strategist_draft": "",
        "planner_instructions": "",
        "analyst_draft": "",
        "draft_report": "", 
        "review_feedback": "", 
        "is_approved": False, 
        "loop_count": 0
    }

    st.subheader("⚙️ 워크플로우 진행 현황")
    
    # 플로우 차트(좌) + 파이프라인 리스트(우) — 비슷한 크기
    col_graph, col_pipeline = st.columns([2, 1])
    
    completed_nodes = set()
    active_nodes = ["baseline_forecaster"]
    is_rejected = False
    
    with col_graph:
        st.caption("📊 워크플로우 다이어그램")
        graph_placeholder = st.empty()
    
    with col_pipeline:
        st.caption("📋 파이프라인 상태")
        pipeline_placeholder = st.empty()
    
    # 초기 렌더링
    graph_placeholder.graphviz_chart(
        render_pipeline_graph(completed_nodes, active_nodes), 
        use_container_width=True
    )
    pipeline_placeholder.markdown(
        render_pipeline_list(completed_nodes, active_nodes)
    )
    
    # ==========================================
    # 🚀 파이프라인 실행 루프
    # ==========================================
    for event in app_graph.stream(current_state):
        finished_nodes = list(event.keys())
        is_rejected = False
        
        for node_name, node_output in event.items():
            current_state.update(node_output)
            
            # 🎯 사이드바에 Agent I/O 로깅
            with agent_log_container:
                with st.expander(f"{'❌' if node_name == 'reviewer' and not node_output.get('is_approved', False) else '✅'} {NODE_LABELS.get(node_name, node_name)}", expanded=False):
                    for key, value in node_output.items():
                        if isinstance(value, list) and len(value) > 10:
                            st.markdown(f"**{key}**: `[{len(value)} items]` min={min(value):.3f}, max={max(value):.3f}")
                        elif isinstance(value, str) and len(value) > 200:
                            st.markdown(f"**{key}**:")
                            st.text(value[:500] + ("..." if len(value) > 500 else ""))
                        else:
                            st.markdown(f"**{key}**: `{value}`")
            
            if node_name == "reviewer" and not node_output.get("is_approved", False):
                is_rejected = True
        
        completed_nodes.update(finished_nodes)
        
        # 다음 노드 예측
        if "reviewer" in finished_nodes:
            if is_rejected:
                # 반려 시: 잠시 빨간색 표시 후 planner로 복귀
                graph_placeholder.graphviz_chart(
                    render_pipeline_graph(completed_nodes, ["reviewer"], is_rejected=True),
                    use_container_width=True
                )
                pipeline_placeholder.markdown(
                    render_pipeline_list(completed_nodes, ["reviewer"], is_rejected=True, loop_count=current_state.get("loop_count", 0))
                )
                time.sleep(1.5)
                active_nodes = ["planner"]
                completed_nodes.difference_update({
                    "planner", "strategist", "mitigated_forecaster", 
                    "analyst", "compiler", "reviewer"
                })
            else:
                active_nodes = []
        else:
            for finished in finished_nodes:
                if finished in PIPELINE_ORDER:
                    idx = PIPELINE_ORDER.index(finished)
                    if idx + 1 < len(PIPELINE_ORDER):
                        active_nodes = [PIPELINE_ORDER[idx + 1]]
                    else:
                        active_nodes = []

        # UI 업데이트
        graph_placeholder.graphviz_chart(
            render_pipeline_graph(completed_nodes, active_nodes),
            use_container_width=True
        )
        pipeline_placeholder.markdown(
            render_pipeline_list(completed_nodes, active_nodes, loop_count=current_state.get("loop_count", 0))
        )
    
    # 최종 완료 상태
    graph_placeholder.graphviz_chart(
        render_pipeline_graph(completed_nodes, []),
        use_container_width=True
    )
    pipeline_placeholder.markdown(
        render_pipeline_list(completed_nodes, [], loop_count=current_state.get("loop_count", 0))
    )

    st.divider()

    # ==========================================
    # 📊 결과 대시보드
    # ==========================================
    if current_state.get("draft_report"):
        try:
            report_data = json.loads(current_state["draft_report"])
            st.header("📊 시뮬레이션 종합 분석 결과")
            
            # 무대응 vs 전략 적용 비교 메트릭
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="위기 경보 등급", value=report_data.get('alert_level', 'UNKNOWN'))
            col2.metric(label="무대응 시 최저점", value=f"{report_data.get('baseline_nvi_bottom', 0.0):.3f}")
            col3.metric(label="전략 적용 시 최저점", value=f"{report_data.get('mitigated_nvi_bottom', 0.0):.3f}")
            col4.metric(
                label="🛡️ 방어 효과", 
                value=f"+{report_data.get('defense_effect', 0.0):.3f}p",
                delta=f"{report_data.get('defense_effect', 0.0):.3f} 포인트 방어"
            )

            st.info(f"**경영진 핵심 요약:** {report_data.get('executive_summary', '')}")

            st.divider()

            # 3-라인 NVI 비교 차트
            st.subheader("📈 NVI 여론 지수 추이 (실제 vs 무대응 vs 전략 적용)")
            
            actual_data = current_state.get("actual_nvi_history", [])
            baseline_data = current_state.get("nvi_baseline_forecast", [])
            mitigated_data = current_state.get("nvi_mitigated_forecast", [])
            
            if actual_data and baseline_data:
                total_len = len(actual_data) + len(baseline_data)
                
                chart_df = pd.DataFrame({
                    "시간(Hour)": np.arange(total_len),
                    "실제 여론 지수(Actual)": actual_data + [None] * len(baseline_data),
                    "무대응 시나리오(Baseline)": [None] * (len(actual_data) - 1) + [actual_data[-1]] + baseline_data,
                    "전략 적용 시나리오(Mitigated)": [None] * (len(actual_data) - 1) + [actual_data[-1]] + mitigated_data,
                }).set_index("시간(Hour)")

                st.line_chart(chart_df, color=["#3b82f6", "#ef4444", "#10b981"])
                st.caption("※ 🔵 파란색: 실제 여론 | 🔴 빨간색: 무대응(Do Nothing) | 🟢 초록색: 전략 적용(Mitigated)")

            st.divider()

            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("⚖️ 법무 및 PR 잠재 리스크 진단")
                st.warning(report_data.get('legal_and_pr_risk', ''))
                
            with col_right:
                st.subheader("⏱️ 시간대별 정밀 액션 플랜")
                if 'immediate_action_items' in report_data:
                    action_df = pd.DataFrame(report_data['immediate_action_items'])
                    action_df.columns = ["실행 시점", "전략적 액션 내용"]
                    st.dataframe(action_df, use_container_width=True, hide_index=True)
                    
        except json.JSONDecodeError:
            st.error("최종 컴파일러 아웃풋이 규격 JSON 포맷이 아닙니다. 원시 스트링 데이터를 로드합니다.")
            st.write(current_state["draft_report"])