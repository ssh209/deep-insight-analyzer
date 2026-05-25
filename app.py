from graph import app

# ==========================================
# 5. 파이프라인 실행 테스트
# ==========================================
if __name__ == "__main__":
    print("🚀 PR 위기 대응 에이전트 파이프라인 시작\n" + "-"*50)
    
    initial_state = {
        "crisis_context": "A일보 배터리 발화 의혹 단독 보도 발생. 오후 2시 1차 입장문 배포 예정."
    }
    
    # 파이프라인 스트리밍 실행 (각 노드의 상태 변화를 관찰)
    for output in app.stream(initial_state):
        for key, value in output.items():
            pass # 진행 과정을 콘솔에서 확인 가능 (위 print 문들)
            
    print("-" * 50)
    print("🎉 최종 리포트 산출 완료!")