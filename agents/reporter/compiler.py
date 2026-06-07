import json
from google.genai import types
from state import CrisisReport

class CompilerAgent:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def run(self, state):
        print("▶️ [Compiler] 에이전트 결과물 취합 및 JSON 포맷팅 중...")
        
        # NVI 수치 추출 (모델별 list 또는 dict 대응)
        baseline = state['nvi_baseline_forecast']
        mitigated = state['nvi_mitigated_forecast']
        
        baseline_points = baseline["point"] if isinstance(baseline, dict) else baseline
        mitigated_points = mitigated["point"] if isinstance(mitigated, dict) else mitigated
        
        baseline_min = min(baseline_points)
        mitigated_min = min(mitigated_points)
        defense = mitigated_min - baseline_min
        
        # 신뢰구간 정보 (있을 때만)
        ci_info = ""
        if isinstance(baseline, dict) and "lower" in baseline:
            ci_info = f"""
        - Baseline 90% CI: [{min(baseline['lower']):.3f}, {max(baseline['upper']):.3f}]
        - Mitigated 90% CI: [{min(mitigated['lower']):.3f}, {max(mitigated['upper']):.3f}]"""
        
        # 여론 지형도 요약
        landscape = state.get('sentiment_landscape', {})
        landscape_text = ""
        if landscape:
            overview = landscape.get("overview", {})
            landscape_text = f"""
        [여론 지형도]
        - 분석 완료: {overview.get('total_analyzed', 0)}건
        - 부정 비율: {overview.get('negative_ratio', 0):.1%}
        - 조롱/멸 비율: {overview.get('mockery_ratio', 0):.1%}
        - 옹호 비율: {overview.get('advocate_ratio', 0):.1%}
        - 부정 핵심 주제: {json.dumps(landscape.get('top_negative_themes', [])[:3], ensure_ascii=False)[:300]}
        - 조롱 핵심 주제: {json.dumps(landscape.get('top_mockery_themes', [])[:2], ensure_ascii=False)[:200]}"""
        
        # 감성 타임라인 요약
        timeline = state.get('sentiment_timeline', {})
        timeline_text = ""
        if timeline:
            history = timeline.get("history", [])
            events = timeline.get("events", [])
            if history:
                timeline_text = f"""
        [감성 타임라인]
        - 총 {len(history)}시간 구간 데이터
        - 초기 부정 비율: {history[0].get('negative_ratio', 0):.1%}
        - 최종 부정 비율: {history[-1].get('negative_ratio', 0):.1%}
        - 주요 이벤트 {len(events)}건: {', '.join(e['label'][:30] for e in events[:3])}"""
        # 리스크 매트릭스 요약
        risk_matrix = state.get('risk_matrix', [])
        risk_text = ""
        if risk_matrix:
            risk_items = []
            for r in risk_matrix[:6]:
                risk_items.append(
                    f"  - [{r.get('category', '?')}] {r.get('risk', '?')} "
                    f"(P:{r.get('probability', '?')}, I:{r.get('impact', '?')}) → {r.get('mitigation', '')[:50]}"
                )
            risk_text = f"""
        [리스크 매트릭스] ({len(risk_matrix)}건)
{chr(10).join(risk_items)}"""
        
        # KOL 요약
        kols = state.get('key_opinion_leaders', [])
        kol_text = ""
        if kols:
            kol_items = []
            for k in kols[:5]:
                kol_items.append(
                    f"  - {k.get('author_name', '?')} ({k.get('platform', '?')}) "
                    f"followers={k.get('followers', 0):,} stance={k.get('stance', '?')} "
                    f"- {k.get('key_content', '')[:40]}"
                )
            kol_text = f"""
        [KOL 식별] ({len(kols)}명)
{chr(10).join(kol_items)}"""
        
        # 벤치마킹 요약
        benchmarks = state.get('benchmark_cases', [])
        benchmark_text = ""
        if benchmarks:
            bm_items = []
            for b in benchmarks[:3]:
                bm_items.append(
                    f"  - {b.get('case_name', '?')} (유사도:{b.get('similarity_score', 0):.0%}) "
                    f"NVI최저:{b.get('nvi_bottom', '?')} 회복:{b.get('recovery_days', '?')}일 "
                    f"\"{ b.get('lesson', '')[:50]}\""
                )
            benchmark_text = f"""
        [유사사례 벤치마킹] ({len(benchmarks)}건)
{chr(10).join(bm_items)}"""
        
        # 대응문 초안 요약
        statements = state.get('draft_statements', [])
        statement_text = ""
        if statements:
            st_items = []
            for s in statements[:2]:
                st_items.append(
                    f"  - [{s.get('type', '?')}] 대상:{s.get('target_audience', '?')} "
                    f"어조:{s.get('tone', '?')} ({len(s.get('draft', ''))}자)"
                )
            statement_text = f"""
        [대응문 초안] ({len(statements)}건)
{chr(10).join(st_items)}"""
        
        prompt = f"""
        [데이터 분석가의 Gap 분석]: {state['analyst_draft']}
        [PR 전략가의 플랜]: {state['strategist_draft']}
        
        [핵심 수치]
        - 무대응(Baseline) 시 NVI 최저점: {baseline_min:.3f}
        - 전략 적용(Mitigated) 시 NVI 최저점: {mitigated_min:.3f}
        - 방어 효과(defense_effect): {defense:.3f}{ci_info}
        {landscape_text}
        {timeline_text}
        {risk_text}
        {kol_text}
        {benchmark_text}
        {statement_text}
        
        당신은 최종 보고서 취합자입니다. 두 전문가의 초안을 완벽하게 융합하여, 
        위 핵심 수치를 정확히 반영한 JSON 스키마 규격에 맞춰 출력하세요.
        baseline_nvi_bottom, mitigated_nvi_bottom, defense_effect 값은 반드시 위 수치를 사용하세요.
        
        다음 데이터가 있으면 해당 필드에 원본 데이터를 포함하세요:
        - sentiment_landscape: 여론 지형도
        - sentiment_timeline: 감성 타임라인
        - risk_matrix: 리스크 매트릭스
        - key_opinion_leaders: KOL 식별 결과
        - draft_statements: 대응문 초안
        - benchmark_cases: 유사사례 벤치마킹
        """
        res = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CrisisReport,
                temperature=0.1
            )
        )
        
        # LLM 출력에 구조화 데이터가 누락될 수 있으므로 직접 주입
        report = json.loads(res.text)
        if landscape:
            report["sentiment_landscape"] = landscape
        if timeline:
            report["sentiment_timeline"] = timeline
        if risk_matrix:
            report["risk_matrix"] = risk_matrix
        if kols:
            report["key_opinion_leaders"] = kols
        if benchmarks:
            report["benchmark_cases"] = benchmarks
        if statements:
            report["draft_statements"] = statements
        
        return {"draft_report": json.dumps(report, ensure_ascii=False), "loop_count": state.get("loop_count", 0) + 1}