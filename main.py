import json
import os
import sys
import asyncio

print("⏳ 모듈 로딩 중...", end="", flush=True)
from engine import init_infrastructure, build_graph
from db import close_db_pool
from config import TRAIN_CSV_PATH, INPUT_CSV_PATH, FORECASTER_MODEL
print(" ✅", flush=True)

# ==========================================
# 🚀 CLI 엔트리포인트 (DB 모드 + CSV 모드 자동 전환)
#
# DB 모드: QueryBuilder → Retriever → Analyzer → Forecaster → ...
# CSV 모드: Forecaster부터 시작 (DB 미연결 시 자동 폴백)
# ==========================================

CRISIS_TYPES = {
    "1": ("victim", "피해자형 — 자연재해, 루머, 무고 등 외부 요인"),
    "2": ("accidental", "우발적 — 기술 결함, 제품 사고 등"),
    "3": ("preventable", "예방가능형 — 은폐, 고의, 법규 위반 등"),
}


def collect_user_input(db_mode: bool) -> dict:
    """사용자로부터 위기 상황 정보를 대화형으로 수집합니다."""

    print("\n" + "=" * 70)
    print("📋 위기 상황 입력")
    print("=" * 70)

    # 1. 위기 상황 설명 (자유 텍스트)
    print("\n[1/3] 위기 상황을 설명해주세요.")
    print("      (여러 줄 입력 가능, 빈 줄 입력 시 완료)")
    print("-" * 40)

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            if lines:
                break
            # 빈 줄만 있으면 계속 입력 대기
            continue
        lines.append(line)

    crisis_context = "\n".join(lines)

    # 2. SCCT 위기 유형 선택
    print(f"\n[2/3] 위기 유형을 선택하세요:")
    for key, (_, desc) in CRISIS_TYPES.items():
        print(f"      {key}. {desc}")

    while True:
        choice = input("\n   선택 (1/2/3): ").strip()
        if choice in CRISIS_TYPES:
            crisis_type = CRISIS_TYPES[choice][0]
            print(f"   → {CRISIS_TYPES[choice][1]}")
            break
        print("   ⚠️ 1, 2, 3 중 하나를 입력하세요.")

    # 3. CSV 모드일 때만 CSV 경로 확인
    if not db_mode:
        print(f"\n[3/3] CSV 모드 — 입력 데이터 경로")
        print(f"      기본값: {INPUT_CSV_PATH}")
        custom_path = input("   경로 (Enter=기본값): ").strip()
        input_csv = custom_path if custom_path else INPUT_CSV_PATH
    else:
        input_csv = INPUT_CSV_PATH  # DB 모드에서는 Analyzer가 CSV를 생성
        print(f"\n[3/3] DB 모드 — Analyzer가 수집 데이터로부터 CSV를 자동 생성합니다.")

    # 요약 출력
    print("\n" + "-" * 40)
    print("📝 입력 요약:")
    print(f"   위기 상황: {crisis_context[:80]}{'...' if len(crisis_context) > 80 else ''}")
    print(f"   위기 유형: {crisis_type}")
    print(f"   동작 모드: {'DB 모드' if db_mode else 'CSV 모드'}")
    print("-" * 40)

    confirm = input("\n   이대로 진행할까요? (Y/n): ").strip().lower()
    if confirm in ("n", "no"):
        print("   ❌ 취소됨. 다시 입력해주세요.")
        return collect_user_input(db_mode)

    return {
        "crisis_context": crisis_context,
        "crisis_type": crisis_type,
        "input_csv_path": input_csv,
    }


if __name__ == "__main__":
    async def _main():
        # 1. 인프라 초기화
        print("⏳ 인프라 초기화 중 (LLM, 임베딩, DB 연결)...")
        client, vector_db, embeddings, db_pool = await init_infrastructure()
        app = build_graph(client, vector_db, embeddings, db_pool)

        db_mode = db_pool is not None
        mode_label = "DB 모드" if db_mode else "CSV 모드"
        print(f"✅ 인프라 준비 완료 — {mode_label}")

        # 2. 사용자 입력 수집
        user_input = collect_user_input(db_mode)

        # 3. Issue 레코드 생성 (DB 모드: INSERT → UUID 반환, CSV 모드: uuid4)
        if db_mode:
            async with db_pool.acquire() as conn:
                issue_id = await conn.fetchval("""
                    INSERT INTO deep_insight.issues (user_input, issue_type, status)
                    VALUES ($1, $2, 'analyzing')
                    RETURNING issue_id::text
                """, user_input["crisis_context"], user_input["crisis_type"])
            print(f"✅ Issue 생성 완료: {issue_id}")
        else:
            import uuid
            issue_id = str(uuid.uuid4())
            print(f"✅ Issue ID (임시): {issue_id}")

        # 4. 초기 상태 구성
        initial_state = {
            # 입력
            "issue_id": issue_id,
            "crisis_context": user_input["crisis_context"],
            "crisis_type": user_input["crisis_type"],
            "train_csv_path": TRAIN_CSV_PATH,
            "input_csv_path": user_input["input_csv_path"],
            "forecaster_model": FORECASTER_MODEL,

            # QueryBuilder 산출물 (DB 모드에서 채워짐)
            "search_keywords": [],
            "search_queries": [],
            "search_embeddings": [],
            "search_time_hint": "",

            # Retriever 산출물 (DB 모드에서 채워짐)
            "retrieved_doc_ids": [],
            "retrieved_docs": [],
            "retrieved_comments": [],
            "retrieved_comment_count": 0,

            # Analyzer 산출물
            "sentiment_landscape": {},
            "sentiment_timeline": {},
            "key_opinion_leaders": [],

            # Forecaster 산출물
            "actual_nvi_history": [],
            "nvi_baseline_forecast": [],
            "nvi_mitigated_forecast": [],

            # Strategist 산출물
            "strategist_timeline": [],
            "strategist_draft": "",
            "risk_matrix": [],
            "draft_statements": [],
            "benchmark_cases": [],

            # Reporter 산출물
            "planner_instructions": "",
            "analyst_draft": "",
            "draft_report": "",
            "review_feedback": "",
            "is_approved": False,
            "loop_count": 0,

            # ReportPublisher 산출물
            "report_url": "",
        }

        # 4. 파이프라인 실행
        print("\n" + "=" * 70)
        print(f"🚀 [{mode_label}] Dual-Forecast Pipeline 가동")
        print("=" * 70)

        final_state = await app.ainvoke(initial_state)

        # 5. 결과 출력
        print("\n" + "=" * 70)
        print("🎉 [파이프라인 완료 — 최종 정제 리포트]")
        print("=" * 70)

        try:
            parsed_report = json.loads(final_state["draft_report"])
            print(json.dumps(parsed_report, indent=4, ensure_ascii=False))

            print("\n" + "-" * 50)
            print("📊 핵심 비교 지표:")
            print(f"   위기 유형: {user_input['crisis_type']}")
            print(f"   무대응 시 NVI 최저점: {parsed_report.get('baseline_nvi_bottom', 'N/A')}")
            print(f"   전략 적용 시 NVI 최저점: {parsed_report.get('mitigated_nvi_bottom', 'N/A')}")
            defense = parsed_report.get('defense_effect', 0)
            print(f"   🛡️ 방어 효과: +{defense:.3f} 포인트")
            print(f"   ⚠️ 위험 등급: {parsed_report.get('alert_level', 'N/A')}")
            print(f"   루프 횟수: {final_state.get('loop_count', 0)}회")
        except (json.JSONDecodeError, TypeError):
            print(final_state.get("draft_report", "(리포트 없음)"))

        # 6. HTML 보고서 URL (노드에서 자동 생성)
        report_url = final_state.get("report_url", "")
        if report_url:
            print(f"\n📄 HTML 보고서: {report_url}")

        # 7. DB 풀 정리
        await close_db_pool(db_pool)

    asyncio.run(_main())