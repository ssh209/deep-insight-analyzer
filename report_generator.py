"""
report_generator.py — 최종 파이프라인 JSON → HTML 보고서 변환 + GCS 발행

LangGraph 노드로 동작:
  Reviewer(승인) → ReportPublisher → END

1. CrisisReport JSON을 Chart.js 기반 프리미엄 HTML로 렌더링
2. GCS에 업로드하여 공개 URL 생성 (GCS_REPORT_BUCKET 미설정 시 로컬 저장)
3. report_url을 PipelineState에 반환 → LangSmith에서 URL 직접 확인 가능
"""
import os
import json
from datetime import datetime

from config import GCS_REPORT_BUCKET, REPORT_DIR


class ReportPublisherAgent:
    """LangGraph 노드 — HTML 보고서 생성 + GCS 발행."""

    def run(self, state: dict) -> dict:
        print("▶️ [ReportPublisher] HTML 보고서 생성 중...")

        # 1) HTML 렌더링
        html_content, filename = _render_html(state)

        # 2) 로컬 저장 (디버깅/백업)
        local_dir = REPORT_DIR
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 3) GCS 업로드 또는 로컬 폴백
        if GCS_REPORT_BUCKET:
            report_url = _upload_to_gcs(html_content, filename)
            print(f"✅ [ReportPublisher] GCS 업로드 완료: {report_url}")
        else:
            abs_path = os.path.abspath(local_path)
            report_url = f"file:///{abs_path}"
            print(f"✅ [ReportPublisher] 로컬 저장 완료: {abs_path}")
            # 로컬에서는 브라우저 자동 열기
            try:
                import webbrowser
                webbrowser.open(report_url)
            except Exception:
                pass

        return {"report_url": report_url}


def _upload_to_gcs(html_content: str, filename: str) -> str:
    """GCS에 HTML을 업로드하고 공개 URL을 반환합니다."""
    from google.cloud import storage
    from config import GCP_PROJECT_ID

    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_REPORT_BUCKET)
    blob = bucket.blob(f"{REPORT_DIR}/{filename}")
    blob.upload_from_string(html_content, content_type="text/html; charset=utf-8")

    # 공개 URL 반환 (버킷이 공개 설정이면 직접 접근 가능)
    return f"https://storage.googleapis.com/{GCS_REPORT_BUCKET}/{REPORT_DIR}/{filename}"


def _render_html(final_state: dict) -> tuple[str, str]:
    """파이프라인 상태를 받아 (html_string, filename) 튜플을 반환합니다."""
    # 보고서 데이터 파싱
    try:
        report = json.loads(final_state["draft_report"])
    except (json.JSONDecodeError, TypeError, KeyError):
        report = {}

    issue_id = final_state.get("issue_id", "unknown")
    crisis_context = final_state.get("crisis_context", "")
    crisis_type = final_state.get("crisis_type", "accidental")

    # NVI 데이터 추출
    actual = final_state.get("actual_nvi_history", [])
    baseline_raw = final_state.get("nvi_baseline_forecast", [])
    mitigated_raw = final_state.get("nvi_mitigated_forecast", [])

    baseline = baseline_raw["point"] if isinstance(baseline_raw, dict) else baseline_raw
    mitigated = mitigated_raw["point"] if isinstance(mitigated_raw, dict) else mitigated_raw

    # 핵심 지표
    alert_level = report.get("alert_level", "UNKNOWN")
    baseline_bottom = report.get("baseline_nvi_bottom", min(baseline) if baseline else 0)
    mitigated_bottom = report.get("mitigated_nvi_bottom", min(mitigated) if mitigated else 0)
    defense_effect = report.get("defense_effect", mitigated_bottom - baseline_bottom)
    executive_summary = report.get("executive_summary", "")
    legal_risk = report.get("legal_and_pr_risk", "")

    # 서브 데이터
    action_items = report.get("immediate_action_items", [])
    risk_matrix = report.get("risk_matrix", [])
    kols = report.get("key_opinion_leaders", [])
    statements = report.get("draft_statements", [])
    benchmarks = report.get("benchmark_cases", [])
    landscape = report.get("sentiment_landscape", {})
    timeline = report.get("sentiment_timeline", {})

    # 위기 유형 한글 매핑
    crisis_type_kr = {
        "victim": "피해자형 (Victim)",
        "accidental": "사고형 (Accidental)",
        "preventable": "예방가능형 (Preventable)",
    }.get(crisis_type, crisis_type)

    # 경보 색상
    alert_colors = {
        "RED": ("#dc2626", "#fef2f2", "#991b1b"),
        "ORANGE": ("#ea580c", "#fff7ed", "#9a3412"),
        "YELLOW": ("#ca8a04", "#fefce8", "#854d0e"),
    }
    alert_bg, alert_bg_light, alert_text = alert_colors.get(
        alert_level, ("#6b7280", "#f9fafb", "#374151")
    )

    # Chart.js 데이터 준비
    total_hours = len(actual) + len(baseline)
    chart_labels = json.dumps(list(range(total_hours)))
    chart_actual = json.dumps(actual + [None] * len(baseline))
    chart_baseline = json.dumps(
        [None] * (len(actual) - 1) + [actual[-1]] + baseline if actual else baseline
    )
    chart_mitigated = json.dumps(
        [None] * (len(actual) - 1) + [actual[-1]] + mitigated if actual else mitigated
    )

    # 신뢰구간 데이터 (있을 때만)
    has_ci = isinstance(baseline_raw, dict) and "lower" in baseline_raw
    if has_ci:
        ci_lower = json.dumps(
            [None] * len(actual) + baseline_raw.get("lower", [])
        )
        ci_upper = json.dumps(
            [None] * len(actual) + baseline_raw.get("upper", [])
        )
    else:
        ci_lower = "[]"
        ci_upper = "[]"

    # 여론 지형도 HTML
    landscape_html = _build_landscape_html(landscape)

    # 액션 플랜 HTML
    action_html = _build_action_table(action_items)

    # 리스크 매트릭스 HTML
    risk_html = _build_risk_table(risk_matrix)

    # KOL HTML
    kol_html = _build_kol_table(kols)

    # 대응문 초안 HTML
    statement_html = _build_statement_html(statements)

    # 벤치마킹 HTML
    benchmark_html = _build_benchmark_html(benchmarks)

    # 감성 타임라인 이벤트 마커
    events = timeline.get("events", [])
    event_annotations = _build_event_annotations(events, len(actual))

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PR 위기 대응 보고서 — {issue_id[:8]}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

  :root {{
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-secondary: #64748b;
    --accent: #3b82f6;
    --red: {alert_bg};
    --red-light: {alert_bg_light};
    --green: #10b981;
    --orange: #f59e0b;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Noto Sans KR', 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
    padding: 0;
  }}

  .container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 24px;
  }}

  /* 헤더 */
  .header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f8fafc;
    padding: 48px 0 36px;
    margin-bottom: 32px;
  }}
  .header .container {{ padding-top: 0; padding-bottom: 0; }}
  .header h1 {{
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
  }}
  .header .subtitle {{
    font-size: 14px;
    color: #94a3b8;
    font-weight: 400;
  }}
  .header .meta {{
    display: flex;
    gap: 24px;
    margin-top: 20px;
    font-size: 13px;
    color: #cbd5e1;
  }}
  .header .meta span {{ display: flex; align-items: center; gap: 6px; }}

  /* 경보 배너 */
  .alert-banner {{
    background: {alert_bg_light};
    border: 2px solid {alert_bg};
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 28px;
    display: flex;
    align-items: flex-start;
    gap: 16px;
  }}
  .alert-badge {{
    background: {alert_bg};
    color: white;
    font-weight: 700;
    font-size: 14px;
    padding: 6px 16px;
    border-radius: 6px;
    white-space: nowrap;
    flex-shrink: 0;
  }}
  .alert-text {{
    font-size: 15px;
    color: {alert_text};
    line-height: 1.6;
  }}

  /* 카드 */
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .card h2 {{
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid #f1f5f9;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  /* 지표 그리드 */
  .metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .metric-card {{
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
  }}
  .metric-label {{
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    font-weight: 500;
  }}
  .metric-value {{
    font-size: 32px;
    font-weight: 700;
    font-family: 'Inter', monospace;
  }}
  .metric-value.red {{ color: var(--red); }}
  .metric-value.green {{ color: var(--green); }}
  .metric-value.blue {{ color: var(--accent); }}

  /* 차트 */
  .chart-container {{
    position: relative;
    height: 360px;
    margin: 16px 0;
  }}

  /* 테이블 */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }}
  th {{
    background: #f1f5f9;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 2px solid var(--border);
  }}
  td {{
    padding: 12px 16px;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: top;
  }}
  tr:hover {{ background: #fafbfd; }}

  /* 뱃지 */
  .badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }}
  .badge-high {{ background: #fef2f2; color: #dc2626; }}
  .badge-medium {{ background: #fff7ed; color: #ea580c; }}
  .badge-low {{ background: #f0fdf4; color: #16a34a; }}
  .badge-critical {{ background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }}
  .badge-legal {{ background: #eef2ff; color: #4338ca; }}
  .badge-reputation {{ background: #fdf4ff; color: #a21caf; }}
  .badge-competitive {{ background: #fff7ed; color: #c2410c; }}
  .badge-operational {{ background: #f0fdf4; color: #15803d; }}

  /* 대응문 */
  .statement {{
    background: #f8fafc;
    border-left: 4px solid var(--accent);
    padding: 20px 24px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
  }}
  .statement-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }}
  .statement-type {{
    background: var(--accent);
    color: white;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
  }}
  .statement-meta {{ font-size: 12px; color: var(--text-secondary); }}
  .statement-body {{
    font-size: 14px;
    line-height: 1.8;
    white-space: pre-wrap;
    color: #334155;
  }}

  /* 벤치마킹 카드 */
  .benchmark-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
  }}
  .benchmark-item {{
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    background: #fafbfd;
  }}
  .benchmark-item h4 {{
    font-size: 15px;
    margin-bottom: 10px;
    color: var(--text);
  }}
  .benchmark-stats {{
    display: flex;
    gap: 16px;
    margin: 12px 0;
    font-size: 12px;
    color: var(--text-secondary);
  }}
  .benchmark-stats span {{ font-weight: 600; color: var(--text); }}
  .benchmark-lesson {{
    font-size: 13px;
    color: #475569;
    font-style: italic;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
  }}

  /* 여론 지형도 */
  .landscape-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  .landscape-bar {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 8px 0;
  }}
  .landscape-bar-label {{
    width: 60px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    text-align: right;
  }}
  .landscape-bar-track {{
    flex: 1;
    height: 24px;
    background: #f1f5f9;
    border-radius: 4px;
    overflow: hidden;
  }}
  .landscape-bar-fill {{
    height: 100%;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    font-size: 11px;
    font-weight: 600;
    color: white;
    min-width: 36px;
  }}

  /* 2열 레이아웃 */
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }}
  @media (max-width: 768px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
    .landscape-grid {{ grid-template-columns: 1fr; }}
  }}

  /* 푸터 */
  .footer {{
    text-align: center;
    padding: 32px;
    font-size: 12px;
    color: var(--text-secondary);
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }}

  /* 인쇄 최적화 */
  @media print {{
    body {{ background: white; }}
    .header {{ background: #1e293b !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .card {{ break-inside: avoid; }}
    .chart-container {{ height: 300px; }}
  }}
</style>
</head>
<body>

<!-- 헤더 -->
<div class="header">
  <div class="container">
    <h1>🚨 PR 위기 대응 분석 보고서</h1>
    <p class="subtitle">Deep Insight Analyzer — Dual-Forecast Simulation Report</p>
    <div class="meta">
      <span>📅 생성일시: {now}</span>
      <span>🔑 Issue: {issue_id[:8]}...</span>
      <span>⚠️ 위기 유형: {crisis_type_kr}</span>
      <span>🔄 검토 루프: {final_state.get('loop_count', 0)}회</span>
    </div>
  </div>
</div>

<div class="container">

  <!-- 경보 배너 -->
  <div class="alert-banner">
    <div class="alert-badge">⚠️ {alert_level}</div>
    <div class="alert-text">{executive_summary}</div>
  </div>

  <!-- 핵심 지표 -->
  <div class="metrics">
    <div class="metric-card">
      <div class="metric-label">무대응 시 NVI 최저점</div>
      <div class="metric-value red">{baseline_bottom:.3f}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">전략 적용 시 NVI 최저점</div>
      <div class="metric-value green">{mitigated_bottom:.3f}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">🛡️ 방어 효과</div>
      <div class="metric-value blue">+{defense_effect:.3f}p</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">위기 유형 (SCCT)</div>
      <div class="metric-value" style="font-size:18px;">{crisis_type_kr}</div>
    </div>
  </div>

  <!-- NVI 차트 -->
  <div class="card">
    <h2>📈 NVI 여론 지수 추이</h2>
    <p style="font-size:13px; color:var(--text-secondary); margin-bottom:12px;">
      과거 실측(파란색) · 무대응 시나리오(빨간색) · 전략 적용 시나리오(초록색)
    </p>
    <div class="chart-container">
      <canvas id="nviChart"></canvas>
    </div>
  </div>

  <!-- 여론 지형도 + 법적 리스크 -->
  <div class="grid-2">
    <div class="card">
      <h2>📊 여론 지형도</h2>
      {landscape_html}
    </div>
    <div class="card">
      <h2>⚖️ 법무 및 PR 리스크</h2>
      <p style="font-size:14px; line-height:1.8; color:#475569;">{legal_risk}</p>
    </div>
  </div>

  <!-- 액션 플랜 -->
  <div class="card">
    <h2>⏱️ 시간대별 정밀 액션 플랜</h2>
    {action_html}
  </div>

  <!-- 리스크 매트릭스 -->
  {f'<div class="card"><h2>⚠️ 리스크 매트릭스</h2>{risk_html}</div>' if risk_matrix else ''}

  <!-- 대응문 초안 -->
  {f'<div class="card"><h2>📝 대응문 초안</h2>{statement_html}</div>' if statements else ''}

  <!-- 유사사례 벤치마킹 -->
  {f'<div class="card"><h2>📚 유사사례 벤치마킹</h2>{benchmark_html}</div>' if benchmarks else ''}

  <!-- KOL -->
  {f'<div class="card"><h2>🔍 주요 오피니언 리더 (KOL)</h2>{kol_html}</div>' if kols else ''}

  <div class="footer">
    Deep Insight Analyzer · 자동 생성 보고서 · Issue {issue_id[:8]} · {now}
  </div>

</div>

<script>
const ctx = document.getElementById('nviChart').getContext('2d');

const annotations = {event_annotations};

new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {chart_labels},
    datasets: [
      {{
        label: '실제 여론 (Actual)',
        data: {chart_actual},
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.08)',
        borderWidth: 2.5,
        pointRadius: 0,
        fill: true,
        spanGaps: false,
      }},
      {{
        label: '무대응 시나리오 (Baseline)',
        data: {chart_baseline},
        borderColor: '#ef4444',
        borderWidth: 2,
        borderDash: [6, 3],
        pointRadius: 0,
        fill: false,
        spanGaps: false,
      }},
      {{
        label: '전략 적용 (Mitigated)',
        data: {chart_mitigated},
        borderColor: '#10b981',
        borderWidth: 2.5,
        pointRadius: 0,
        fill: false,
        spanGaps: false,
      }},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{
        position: 'top',
        labels: {{ font: {{ family: 'Inter, Noto Sans KR', size: 12 }}, usePointStyle: true, padding: 20 }}
      }},
      annotation: {{
        annotations: annotations
      }},
      tooltip: {{
        backgroundColor: '#1e293b',
        titleFont: {{ family: 'Noto Sans KR', size: 13 }},
        bodyFont: {{ family: 'Inter', size: 12 }},
        padding: 12,
        cornerRadius: 8,
      }}
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: '시간 (Hour)', font: {{ size: 12 }} }},
        grid: {{ display: false }},
        ticks: {{ maxTicksLimit: 20 }}
      }},
      y: {{
        title: {{ display: true, text: 'NVI (Net Valence Index)', font: {{ size: 12 }} }},
        min: 0, max: 1,
        grid: {{ color: '#f1f5f9' }},
      }}
    }}
  }}
}});
</script>

</body>
</html>"""

    filename = f"report_{issue_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    return html, filename


def generate_html_report(final_state: dict, output_dir: str = None) -> str:
    """호환 래퍼 — HTML 생성 후 로컬 파일로 저장."""
    if output_dir is None:
        output_dir = REPORT_DIR
    html, filename = _render_html(final_state)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return filepath


# ==========================================
# 내부 헬퍼 함수
# ==========================================

def _build_landscape_html(landscape: dict) -> str:
    if not landscape:
        return '<p style="color:#94a3b8;">여론 지형도 데이터 없음</p>'

    overview = landscape.get("overview", {})
    neg = overview.get("negative_ratio", 0)
    mock = overview.get("mockery_ratio", 0)
    adv = overview.get("advocate_ratio", 0)
    neutral = max(0, 1 - neg - adv)

    themes = landscape.get("top_negative_themes", [])
    theme_html = ""
    if themes:
        items = "".join(
            f'<li style="margin:4px 0;font-size:13px;">{t}</li>'
            if isinstance(t, str) else
            f'<li style="margin:4px 0;font-size:13px;">{t.get("theme", t)}</li>'
            for t in themes[:5]
        )
        theme_html = f'<div style="margin-top:16px;"><strong style="font-size:13px;">부정 핵심 주제</strong><ul style="padding-left:20px;margin-top:6px;">{items}</ul></div>'

    return f"""
    <div class="landscape-bar">
      <span class="landscape-bar-label">부정</span>
      <div class="landscape-bar-track">
        <div class="landscape-bar-fill" style="width:{neg*100:.0f}%;background:#ef4444;">{neg:.0%}</div>
      </div>
    </div>
    <div class="landscape-bar">
      <span class="landscape-bar-label">조롱</span>
      <div class="landscape-bar-track">
        <div class="landscape-bar-fill" style="width:{mock*100:.0f}%;background:#f59e0b;">{mock:.0%}</div>
      </div>
    </div>
    <div class="landscape-bar">
      <span class="landscape-bar-label">옹호</span>
      <div class="landscape-bar-track">
        <div class="landscape-bar-fill" style="width:{adv*100:.0f}%;background:#10b981;">{adv:.0%}</div>
      </div>
    </div>
    <div class="landscape-bar">
      <span class="landscape-bar-label">중립</span>
      <div class="landscape-bar-track">
        <div class="landscape-bar-fill" style="width:{neutral*100:.0f}%;background:#94a3b8;">{neutral:.0%}</div>
      </div>
    </div>
    {theme_html}
    """


def _build_action_table(items: list) -> str:
    if not items:
        return '<p style="color:#94a3b8;">액션 플랜 데이터 없음</p>'

    rows = ""
    for i, item in enumerate(items, 1):
        tf = item.get("timeframe", "")
        action = item.get("action", "")
        rows += f"<tr><td style='white-space:nowrap;font-weight:600;'>{tf}</td><td>{action}</td></tr>"

    return f"""<table>
    <thead><tr><th style="width:140px;">실행 시점</th><th>전략적 액션</th></tr></thead>
    <tbody>{rows}</tbody>
    </table>"""


def _build_risk_table(risks: list) -> str:
    if not risks:
        return ""

    rows = ""
    for r in risks:
        prob = r.get("probability", "")
        impact = r.get("impact", "")
        cat = r.get("category", "")
        prob_class = f"badge-{prob}" if prob in ("high", "medium", "low") else "badge-medium"
        impact_class = f"badge-{impact}" if impact in ("high", "medium", "low", "critical") else "badge-medium"
        cat_class = f"badge-{cat}" if cat in ("legal", "reputation", "competitive", "operational") else "badge-medium"
        rows += f"""<tr>
          <td>{r.get('risk', '')}</td>
          <td><span class="badge {cat_class}">{cat}</span></td>
          <td><span class="badge {prob_class}">{prob}</span></td>
          <td><span class="badge {impact_class}">{impact}</span></td>
          <td style="font-size:13px;">{r.get('mitigation', '')}</td>
        </tr>"""

    return f"""<table>
    <thead><tr><th>리스크</th><th>범주</th><th>확률</th><th>영향도</th><th>완화 전략</th></tr></thead>
    <tbody>{rows}</tbody>
    </table>"""


def _build_kol_table(kols: list) -> str:
    if not kols:
        return ""

    rows = ""
    for k in kols:
        stance = k.get("stance", "")
        stance_color = "#ef4444" if stance in ("hostile", "negative") else "#10b981" if stance in ("supportive", "positive") else "#94a3b8"
        followers = k.get("followers", 0)
        followers_str = f"{followers:,}" if isinstance(followers, (int, float)) else str(followers)
        rows += f"""<tr>
          <td style="font-weight:500;">{k.get('author_name', 'N/A')}</td>
          <td>{k.get('platform', '')}</td>
          <td>{followers_str}</td>
          <td style="color:{stance_color};font-weight:600;">{stance}</td>
          <td style="font-size:13px;">{k.get('key_content', '')[:80]}</td>
        </tr>"""

    return f"""<table>
    <thead><tr><th>이름</th><th>플랫폼</th><th>팔로워</th><th>성향</th><th>핵심 콘텐츠</th></tr></thead>
    <tbody>{rows}</tbody>
    </table>"""


def _build_statement_html(statements: list) -> str:
    if not statements:
        return ""

    html = ""
    for s in statements:
        html += f"""
        <div class="statement">
          <div class="statement-header">
            <span class="statement-type">{s.get('type', '입장문')}</span>
            <span class="statement-meta">대상: {s.get('target_audience', '')} · 어조: {s.get('tone', '')}</span>
          </div>
          <div class="statement-body">{s.get('draft', '')}</div>
        </div>"""
    return html


def _build_benchmark_html(benchmarks: list) -> str:
    if not benchmarks:
        return ""

    html = '<div class="benchmark-grid">'
    for b in benchmarks:
        sim = b.get("similarity_score", 0)
        sim_pct = f"{sim:.0%}" if isinstance(sim, (int, float)) else str(sim)
        html += f"""
        <div class="benchmark-item">
          <h4>{b.get('case_name', 'N/A')}</h4>
          <div class="benchmark-stats">
            <div>유사도 <span>{sim_pct}</span></div>
            <div>NVI 최저 <span>{b.get('nvi_bottom', 'N/A')}</span></div>
            <div>회복 <span>{b.get('recovery_days', 'N/A')}일</span></div>
            <div>첫 대응 <span>{b.get('initial_response_hours', 'N/A')}h</span></div>
          </div>
          <div class="benchmark-lesson">"{b.get('lesson', '')}"</div>
        </div>"""
    html += "</div>"
    return html


def _build_event_annotations(events: list, actual_len: int) -> str:
    """Chart.js annotation plugin용 이벤트 마커를 생성합니다."""
    if not events:
        return "{}"

    annotations = {}
    for i, evt in enumerate(events[:6]):  # 최대 6개만 표시
        hour = evt.get("hour", 0)
        label = evt.get("label", "")[:25]
        evt_type = evt.get("type", "")

        color = "#ef4444" if "influencer" in evt_type else "#f59e0b"

        annotations[f"event{i}"] = {
            "type": "line",
            "xMin": hour,
            "xMax": hour,
            "borderColor": color,
            "borderWidth": 1.5,
            "borderDash": [4, 4],
            "label": {
                "display": True,
                "content": label,
                "position": "start",
                "backgroundColor": color,
                "color": "white",
                "font": {"size": 10, "family": "Noto Sans KR"},
                "padding": 4,
                "borderRadius": 4,
            }
        }

    return json.dumps(annotations, ensure_ascii=False)
