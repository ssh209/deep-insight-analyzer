import json
from engine import init_infrastructure, build_graph

# ==========================================
# 🚀 CLI 통합 테스트 엔트리포인트
# engine.py의 인프라 + 그래프 빌더를 재사용합니다.
# ==========================================

if __name__ == "__main__":
    # 1. 인프라 초기화 (engine.py 재사용)
    print("⏳ 인프라 레이어: 지식베이스 벡터 DB 인스턴스 초기화 중...")
    client, vector_db = init_infrastructure()
    app = build_graph(client, vector_db)
    
    # 2. 테스트 입력 데이터
    target_csv = "data/input_crisis_72h.csv"
    
    # 1회차 루프에서 CCO 반려를 확실히 유도하도록 '유감스럽다' 키워드 가상 계획에 주입
    input_metadata = """
    [현재 상황] 초기 대응 지연 및 무대응 기간 유튜버 2연타 타격으로 여론 최악 직면.
    [향후 대응 계획]
    - 4시간 뒤: '사실무근이며 깊은 유감이다'라는 1차 해명문 배포 (action_type: 1)
    - 24시간 뒤: 전면 리콜 공표 및 대표이사 명의의 2차 대고객 사과문 발표 (action_type: 2)
    """
    
    # 3. 초기 상태 (새로운 PipelineState 구조)
    initial_state = {
        "train_csv_path": "data/pr_crisis_dataset.csv",
        "input_csv_path": target_csv,
        "crisis_context": input_metadata,
        "crisis_type": "accidental",
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
    
    print("\n" + "="*70)
    print("🚀 [Dual-Forecast Pipeline] 무대응 vs 전략 적용 이원화 워크플로우 가동")
    print("="*70)
    
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*70)
    print("🎉 [파이프라인 터미널 아웃풋 - 최종 정제된 대시보드용 JSON 데이터]")
    print("="*70)
    
    try:
        parsed_report = json.loads(final_state["draft_report"])
        print(json.dumps(parsed_report, indent=4, ensure_ascii=False))
        
        # 핵심 비교 지표 출력
        print("\n" + "-"*50)
        print("📊 핵심 비교 지표:")
        print(f"   무대응 시 NVI 최저점: {parsed_report.get('baseline_nvi_bottom', 'N/A')}")
        print(f"   전략 적용 시 NVI 최저점: {parsed_report.get('mitigated_nvi_bottom', 'N/A')}")
        print(f"   🛡️ 방어 효과: +{parsed_report.get('defense_effect', 0):.3f} 포인트")
    except json.JSONDecodeError:
        print(final_state["draft_report"])