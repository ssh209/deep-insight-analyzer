import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
import time
import base64
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
# 📊 Graphviz 다이어그램 렌더링 헬퍼 함수
# ==========================================
def render_pipeline_graph(completed_nodes, current_nodes, is_rejected=False):
    # Graphviz Digraph (방향성 그래프) 객체 생성
    dot = graphviz.Digraph(engine='dot')
    dot.attr(rankdir='TD')  # Top to Bottom (위에서 아래로)
    
    # 폰트 깨짐 방지 (한글 폰트 지정)
    dot.attr('node', fontname='Malgun Gothic') 

    # 노드 정의 (id: 라벨)
    nodes = {
        "START": "시작",
        "analyzer": "1. 상황 분석기",
        "forecaster": "2. NVI 시뮬레이터",
        "planner": "3. TF 기획자",
        "analyst": "4-A. 데이터 분석가",
        "strategist": "4-B. PR 전략가",
        "compiler": "5. 보고서 취합자",
        "reviewer": "6. CCO 레드팀",
        "END": "종료"
    }

    # 동적 색상 할당
    for node_id, label in nodes.items():
        color = "#f8fafc"       # 기본 배경색 (회백색)
        fontcolor = "#334155"   # 기본 글자색
        shape = 'box' if node_id not in ['START', 'END'] else 'ellipse'

        if node_id in current_nodes:
            if node_id == "reviewer" and is_rejected:
                color = "#ef4444"  # 빨간색 (반려)
                fontcolor = "white"
            else:
                color = "#3b82f6"  # 파란색 (현재 실행 중)
                fontcolor = "white"
        elif node_id in completed_nodes or (node_id == "START" and completed_nodes):
            color = "#10b981"      # 초록색 (완료)
            fontcolor = "white"

        # 노드 추가
        dot.node(node_id, label, style='filled,rounded', fillcolor=color, fontcolor=fontcolor, shape=shape)

    # 엣지(화살표) 연결
    dot.edge("START", "analyzer")
    dot.edge("analyzer", "forecaster")
    dot.edge("forecaster", "planner")
    dot.edge("planner", "analyst")
    dot.edge("planner", "strategist")
    dot.edge("analyst", "compiler")
    dot.edge("strategist", "compiler")
    dot.edge("compiler", "reviewer")
    
    # 조건부 엣지
    dot.edge("reviewer", "planner", label="반려", color="#ef4444", fontcolor="#ef4444", fontname='Malgun Gothic')
    dot.edge("reviewer", "END", label="승인", color="#10b981", fontcolor="#10b981", fontname='Malgun Gothic')

    return dot

# ==========================================
# 🎨 Streamlit 대시보드 UI
# ==========================================
st.set_page_config(page_title="PR 위기 대응 대시보드", page_icon="🚨", layout="wide")

st.title("🚨 실시간 PR 위기 시뮬레이션 대시보드")
st.markdown("분산형 에이전트 그룹의 협업 워크플로우를 활용한 실시간 여론(NVI) 추론 시스템")

with st.sidebar:
    st.header("📝 입력 파라미터")
    target_csv = st.text_input("데이터셋 경로", value="data/pr_crisis_dataset.csv")
    
    default_meta = """[현재 상황] 초기 대응 지연 및 무대응 기간 유튜버 2연타 타격으로 여론 최악 직면.
[향후 대응 계획]
- 4시간 뒤: '사실무근이며 깊은 유감이다'라는 1차 해명문 배포 (action_type: 1)
- 24시간 뒤: 전면 리콜 공표 및 대표이사 명의의 2차 대고객 사과문 발표 (action_type: 2)"""
    
    input_metadata = st.text_area("위기 메타 정보 및 대응 계획", value=default_meta, height=250)
    start_btn = st.button("🚀 파이프라인 가동", type="primary", use_container_width=True)

if start_btn:
    current_state = {
        "input_csv_path": target_csv,
        "crisis_context": input_metadata,
        "timeline_events": [], 
        "nvi_forecast": [], 
        "planner_instructions": "",
        "analyst_draft": "",
        "strategist_draft": "",
        "draft_report": "", 
        "review_feedback": "", 
        "is_approved": False, 
        "loop_count": 0
    }

    st.subheader("⚙️ 멀티 에이전트 워크플로우 진행 현황")
    col_viz, col_log = st.columns([1.5, 1])
    
    completed_nodes = set()
    
    with col_viz:
        graph_placeholder = st.empty()
        
    with col_log:
        with st.status("에이전트 연산 인프라 기동 중...", expanded=True) as status:
            
            # 🎯 1. 시작 직후: 첫 번째 에이전트(analyzer)에 미리 파란불 켜기
            active_nodes = ["analyzer"]
            graph_placeholder.graphviz_chart(render_pipeline_graph(completed_nodes, active_nodes), use_container_width=True)
            
            for event in app_graph.stream(current_state):
                finished_nodes = list(event.keys()) # '방금 연산이 끝난' 노드들
                is_rejected = False
                
                # 방금 끝난 노드들의 결과를 전체 상태에 업데이트
                for node_name, node_output in event.items():
                    current_state.update(node_output)
                    
                    if node_name == "reviewer" and not node_output.get("is_approved", False):
                        st.error(f"❌ **{node_name}** 반려: {node_output.get('review_feedback', '규정 미달')}")
                        is_rejected = True
                    else:
                        st.success(f"✅ **{node_name}** 오퍼레이션 완료")
                
                # 완료 목록에 방금 끝난 녀석들 추가 (초록색으로 변할 준비)
                completed_nodes.update(finished_nodes)
                
                # 🎯 2. 토폴로지 기반 '다음 활성 노드' 예측 로직
                if "reviewer" in finished_nodes:
                    if is_rejected:
                        # 반려 시: 리뷰어를 잠시 빨간색으로 보여준 뒤
                        graph_placeholder.graphviz_chart(render_pipeline_graph(completed_nodes, ["reviewer"], is_rejected=True), use_container_width=True)
                        time.sleep(1.5) 
                        # 기획자(planner)부터 다시 시작하도록 상태 리셋
                        active_nodes = ["planner"]
                        completed_nodes.difference_update({"planner", "analyst", "strategist", "compiler", "reviewer"})
                    else:
                        active_nodes = [] # 승인 시 종료
                        
                elif "compiler" in finished_nodes:
                    active_nodes = ["reviewer"]
                    
                elif "analyst" in finished_nodes or "strategist" in finished_nodes:
                    # 병렬 처리 분기: 둘 다 끝났으면 compiler로 넘어가고, 하나만 끝났으면 남은 녀석만 파란불 유지
                    if "analyst" in completed_nodes and "strategist" in completed_nodes:
                        active_nodes = ["compiler"]
                    else:
                        active_nodes = [n for n in ["analyst", "strategist"] if n not in completed_nodes]
                        
                elif "planner" in finished_nodes:
                    active_nodes = ["analyst", "strategist"] # 두 에이전트 동시 가동
                    
                elif "forecaster" in finished_nodes:
                    active_nodes = ["planner"]
                    
                elif "analyzer" in finished_nodes:
                    active_nodes = ["forecaster"]

                # 🎯 3. 엔진이 다음 연산에 들어가기 직전(대기 상태), 예측된 다음 노드에 파란불 켜기
                graph_placeholder.graphviz_chart(render_pipeline_graph(completed_nodes, active_nodes), use_container_width=True)
                
            status.update(label="파이프라인 실행 프로세스 종료", state="complete", expanded=False)
            
            # 최종 종료 시 다이어그램 전체 완료 상태로 렌더링
            graph_placeholder.graphviz_chart(render_pipeline_graph(completed_nodes, []), use_container_width=True)

    st.divider()

    # 결과 대시보드 데이터 바인딩 로직 (기존과 동일하게 유지)
    if current_state.get("draft_report"):
        try:
            report_data = json.loads(current_state["draft_report"])
            st.header("📊 시뮬레이션 종합 분석 결과")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(label="위기 경보 등급", value=report_data.get('alert_level', 'UNKNOWN'))
            col2.metric(label="예상 NVI 최저점", value=f"{report_data.get('expected_nvi_bottom', 0.0)} / 1.0")
            col3.metric(label="총 연산 루프 횟수", value=f"{current_state.get('loop_count', 0)} 회")

            st.info(f"**경영진 핵심 요약:** {report_data.get('executive_summary', '')}")

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