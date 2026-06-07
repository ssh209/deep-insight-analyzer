"""
Issue Cracker — DB 시드 데이터 생성기
issue_cracker 스키마에 맞는 더미 데이터를 PostgreSQL에 직접 삽입합니다.

시나리오: "XX전자 배터리 발화 은폐 의혹" (preventable 유형)

5-페이즈 위기 생애주기 (72시간 = 3일):
  ① 잠복기  (H0~H12)   → 커뮤니티 호소글, 소규모 확산
  ② 폭발기  (H12~H30)  → 유튜버 저격, 주류 미디어 보도
  ③ 바닥    (H30~H48)  → 밈화, 여론 최악
  ④ 교착기  (H48~H60)  → 1차 해명문, 느린 반등
  ⑤ 미결    (H60~H72)  → 아직 수습 안 됨 (예측 필요)

사용법:
  python manual/seed_db.py --dsn postgresql://user:pass@localhost:5432/dbname
"""
import sys
import os
import asyncio
import argparse
import random
import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EMBEDDING_MODEL

PASSAGE_PREFIX = "passage: "

# ==========================================
# 시나리오 정의
# ==========================================
ISSUE_ID = "issue-battery-001"
ISSUE_TITLE = "XX전자 배터리 발화 은폐 의혹"
ISSUE_USER_INPUT = (
    "XX전자의 스마트폰 배터리가 충전 중 발화하는 사례가 3건 연속 접수되었으나, "
    "기업이 이를 2개월간 은폐한 정황이 내부 문건을 통해 폭로됨. "
    "커뮤니티 호소글에서 시작, 대형 유튜버 저격으로 여론 폭발."
)
ISSUE_TYPE = "preventable"

KST = timezone(timedelta(hours=9))
START_TIME = datetime(2025, 6, 1, 9, 0, 0, tzinfo=KST)

# ==========================================
# 포스트 시나리오 (11개 원문)
# ==========================================
POSTS = [
    # (hour_offset, platform, content_type, title, author_id, author_name, followers, views)
    (0,   "community", "post",    "XX전자 폰 충전하다가 이불에 불 붙었는데 서비스센터에서 유상수리래요",
     "user_001", "피해자A", 50, 2400),
    (3,   "community", "post",    "저도요... 배터리 부풀어오름 3번째인데 자비수리만 안내받음",
     "user_002", "피해자B", 30, 1800),
    (8,   "twitter",   "tweet",   "XX전자 배터리 발화 은폐 의혹... 내부 문건 유출이라는데 이거 실화?",
     "user_003", "IT기자_김", 45000, 28000),
    (12,  "youtube",   "video",   "[긴급] XX전자 배터리 발화 은폐 내부문건 단독입수!! 충격적 내용 공개",
     "ytb_001", "IT리뷰어_박프로", 520000, 1800000),
    (18,  "youtube",   "video",   "XX전자 배터리 폭발 피해자 직접 만나봤습니다 | 은폐 타임라인 정리",
     "ytb_002", "뉴스탐사_이기자", 180000, 650000),
    (24,  "news",      "article", "[단독] XX전자, 배터리 발화 사실 2개월 전 내부 보고서로 인지...은폐 정황",
     "news_001", "전자신문", 0, 350000),
    (30,  "community", "post",    "ㅋㅋㅋ XX전자 \"확인 중입니다\" 이거 밈 되겠다",
     "user_010", "밈장인", 200, 45000),
    (36,  "youtube",   "short",   "XX전자 배터리 발화 실험해봤습니다 #shorts",
     "ytb_003", "과학실험TV", 95000, 2200000),
    (42,  "twitter",   "tweet",   "XX전자 배터리 피해자 연대 모임 결성합니다. 관련 피해 겪으신 분 DM 주세요",
     "user_020", "소비자권익연대", 12000, 58000),
    (48,  "news",      "article", "XX전자, 공식 입장문 발표... \"조사 중이며 안전에 만전\"",
     "news_002", "연합뉴스", 0, 180000),
    (60,  "youtube",   "video",   "XX전자 입장문 분석: 이게 사과야 변명이야? | 위기관리 전문가 의견",
     "ytb_004", "경제읽어주는남자", 310000, 890000),
]

# ==========================================
# 댓글 템플릿 (감성별)
# ==========================================
NEGATIVE_COMMENTS = [
    "이게 나라냐 이게 기업이냐", "불매 시작합니다", "은폐하다 걸린 거잖아 ㅋㅋ",
    "진짜 실망이다 XX전자", "소비자를 뭘로 보는 건지", "안전 불감증의 극치",
    "사람이 다칠 뻔했는데 은폐를? 미쳤나", "경영진 전원 사퇴해라", "리콜 안 하면 고소할 거임",
    "내부 문건 유출된 거 보면 조직적 은폐 맞네", "이 회사 폰 쓰는데 갈아타야겠다",
    "주주입니다. 경영진 책임 물을 겁니다", "내 가족이 다쳤으면 어쩔 뻔했어",
    "해명문 보고 더 화남", "소비자 기만이 이 정도면 범죄 아닌가",
    "아직도 이 회사 폰 쓰는 사람 있어?", "2개월 은폐... 소름 돋는다",
    "진상 규명 때까지 불매", "피해자한테 사과부터 해", "이건 단순 결함이 아니라 범죄",
    "국감 가야 할 사안", "기업 윤리 완전 바닥", "충전기 근처에 소화기 놔야겠다 ㅋㅋ",
    "이번엔 진짜 끝이다 XX전자", "CEO 나와서 직접 사과해", "피해보상 제대로 해줘라",
]

MOCKERY_COMMENTS = [
    "XX전자: 불이야? 유상수리입니다 ㅋㅋㅋㅋ", "배터리 아니고 화염방사기 ㅋㅋ",
    "핫한 신제품 출시했네 ㅋㅋ 리얼 핫", "\"확인 중\" << 이거 올해의 밈 ㅋㅋ",
    "XX전자 폰 = 휴대용 난로 ㅋㅋㅋ", "캠핑갈 때 XX전자 폰 들고가면 불 안 피워도 되겠다",
    "XX전자 새 슬로건: We Light Your Life 🔥", "갤럭시 노트 7 후배 등장?? ㅋㅋ",
    "배터리 발화? 그건 프리미엄 온열 기능입니다 ㅋㅋ", "은폐 스킬은 세계 최고 기술력 ㅋㅋ",
]

ADVOCATE_COMMENTS = [
    "다른 회사도 배터리 이슈 있었는데 XX전자만 까는 건 좀...",
    "아직 조사 중인데 너무 단정짓는 거 아닌가요", "이 회사 기술력은 인정해야지",
    "내부 고발자 덕에 빨리 알려진 거라 오히려 다행", "과거에도 잘 수습했으니 이번에도 잘 하겠죠",
]

NEUTRAL_COMMENTS = [
    "정확한 조사 결과 기다려봐야 할 듯", "배터리 이슈는 전기차 쪽에서도 많은데",
    "이런 건 공정위가 나서야 하는 거 아닌가", "팩트 확인 후 판단하겠습니다",
    "둘 다 잘못이 있을 수 있지 않나", "좀 더 지켜보자",
]

# ==========================================
# 시간대별 댓글 분포
# ==========================================
def _comment_distribution(hour: int) -> dict:
    """시간 오프셋에 따른 댓글 수와 감성 비율 반환."""
    if hour < 12:      # 잠복기
        return {"count": random.randint(5, 15),  "neg": 0.40, "mock": 0.05, "adv": 0.10, "neu": 0.45}
    elif hour < 30:    # 폭발기
        return {"count": random.randint(30, 80), "neg": 0.55, "mock": 0.20, "adv": 0.03, "neu": 0.22}
    elif hour < 48:    # 바닥
        return {"count": random.randint(40, 100),"neg": 0.45, "mock": 0.30, "adv": 0.02, "neu": 0.23}
    elif hour < 60:    # 교착기
        return {"count": random.randint(15, 40), "neg": 0.35, "mock": 0.15, "adv": 0.10, "neu": 0.40}
    else:              # 미결
        return {"count": random.randint(10, 25), "neg": 0.30, "mock": 0.10, "adv": 0.15, "neu": 0.45}


def _pick_comment(sentiment: str) -> str:
    if sentiment == "negative":
        return random.choice(NEGATIVE_COMMENTS)
    elif sentiment == "mockery":
        return random.choice(MOCKERY_COMMENTS)
    elif sentiment == "advocate":
        return random.choice(ADVOCATE_COMMENTS)
    return random.choice(NEUTRAL_COMMENTS)


def _pick_sentiment(dist: dict) -> str:
    r = random.random()
    if r < dist["neg"]:
        return "negative"
    elif r < dist["neg"] + dist["mock"]:
        return "mockery"
    elif r < dist["neg"] + dist["mock"] + dist["adv"]:
        return "advocate"
    return "neutral"


# ==========================================
# 포스트별 댓글 생성
# ==========================================
def generate_comments(posts_data: list[dict]) -> list[dict]:
    """각 포스트의 시간대에 따라 사실적인 댓글 분포 생성."""
    comments = []
    
    for post in posts_data:
        hour = post["hour_offset"]
        dist = _comment_distribution(hour)
        count = dist["count"]
        
        for j in range(count):
            sentiment_label = _pick_sentiment(dist)
            body = _pick_comment(sentiment_label)
            
            # 댓글은 포스트 이후 0~6시간 사이에 생성
            minutes_offset = random.randint(0, 360)
            created_at = START_TIME + timedelta(hours=hour, minutes=minutes_offset)
            
            comments.append({
                "comment_id": f"cmt-{uuid.uuid4().hex[:12]}",
                "post_id": post["post_id"],
                "created_at": created_at,
                "body": body,
                "author_id": f"anon_{random.randint(1000, 9999)}",
                "author_name": None,
                "author_followers": random.choice([None, 0, random.randint(10, 500)]),
                "like_count": max(0, int(random.gauss(20, 30))) if sentiment_label in ("negative", "mockery") else random.randint(0, 10),
                "reply_count": random.randint(0, 5),
            })
    
    return comments


# ==========================================
# DB 삽입
# ==========================================
async def seed(dsn: str):
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)
    
    async with pool.acquire() as conn:
        # 1. Issue
        await conn.execute("""
            INSERT INTO issue_cracker.issues
                (issue_id, user_input, issue_type, status)
            VALUES ($1, $2, $3, 'active')
            ON CONFLICT (issue_id) DO UPDATE SET
                user_input = EXCLUDED.user_input,
                updated_at = NOW()
        """, ISSUE_ID, ISSUE_USER_INPUT, ISSUE_TYPE)
        print(f"✅ Issue 생성: {ISSUE_ID}")
        
        # 2. Posts (issue 독립 — issue_id 컬럼 없음)
        posts_data = []
        for i, p in enumerate(POSTS):
            hour, platform, ctype, title, author_id, author_name, followers, views = p
            post_id = f"post-{i+1:03d}"
            created_at = START_TIME + timedelta(hours=hour)
            
            await conn.execute("""
                INSERT INTO issue_cracker.posts
                    (post_id, created_at, platform, content_type,
                     title, author_id, author_name, author_followers,
                     view_count, like_count, comment_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (post_id) DO NOTHING
            """, post_id, created_at, platform, ctype,
                 title, author_id, author_name, followers,
                 views, int(views * random.uniform(0.02, 0.08)), 0)
            
            posts_data.append({"post_id": post_id, "hour_offset": hour})
        
        print(f"✅ Posts 생성: {len(POSTS)}건")
        
        # 3. Comments (issue 독립 — issue_id 컬럼 없음)
        comments = generate_comments(posts_data)
        
        for c in comments:
            await conn.execute("""
                INSERT INTO issue_cracker.comments
                    (comment_id, post_id, created_at,
                     body, author_id, author_name, author_followers,
                     like_count, reply_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (comment_id) DO NOTHING
            """, c["comment_id"], c["post_id"], c["created_at"],
                 c["body"], c["author_id"], c["author_name"], c["author_followers"],
                 c["like_count"], c["reply_count"])
        
        # 포스트별 comment_count 업데이트
        await conn.execute("""
            UPDATE issue_cracker.posts p
            SET comment_count = sub.cnt
            FROM (
                SELECT post_id, COUNT(*) as cnt
                FROM issue_cracker.comments
                GROUP BY post_id
            ) sub
            WHERE p.post_id = sub.post_id
        """)
        
        print(f"✅ Comments 생성: {len(comments)}건")
    
    # ==========================================
    # 4. 임베딩 생성 (E5: passage: prefix)
    # ==========================================
    print(f"\n>> 임베딩 모델 로드: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"   디바이스: {model.device}")
    
    async with pool.acquire() as conn:
        # 4-1. Posts 임베딩
        post_rows = await conn.fetch("""
            SELECT post_id, title, body FROM issue_cracker.posts WHERE embedding IS NULL
        """)
        if post_rows:
            post_texts = [
                PASSAGE_PREFIX + " ".join(filter(None, [r["title"], (r["body"] or "")[:500]]))
                for r in post_rows
            ]
            post_embs = model.encode(post_texts, normalize_embeddings=True, show_progress_bar=False)
            for row, emb in zip(post_rows, post_embs):
                vec_str = "[" + ",".join(str(float(v)) for v in emb) + "]"
                await conn.execute("""
                    UPDATE issue_cracker.posts SET embedding = $2::vector WHERE post_id = $1
                """, row["post_id"], vec_str)
            print(f"✅ Posts 임베딩 생성: {len(post_rows)}건")
        
        # 4-2. Comments 임베딩
        comment_rows = await conn.fetch("""
            SELECT comment_id, body FROM issue_cracker.comments WHERE embedding IS NULL
        """)
        if comment_rows:
            comment_texts = [PASSAGE_PREFIX + (r["body"] or "")[:500] for r in comment_rows]
            # 배치 처리 (100건씩)
            batch_size = 100
            for i in range(0, len(comment_rows), batch_size):
                batch_rows = comment_rows[i:i + batch_size]
                batch_texts = comment_texts[i:i + batch_size]
                batch_embs = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
                for row, emb in zip(batch_rows, batch_embs):
                    vec_str = "[" + ",".join(str(float(v)) for v in emb) + "]"
                    await conn.execute("""
                        UPDATE issue_cracker.comments SET embedding = $2::vector WHERE comment_id = $1
                    """, row["comment_id"], vec_str)
                print(f"   Comments 임베딩 [{i+1}~{i+len(batch_rows)}] 완료")
            print(f"✅ Comments 임베딩 생성: {len(comment_rows)}건")
    
    # ==========================================
    # 5. 요약 통계
    # ==========================================
    async with pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_comments,
                MIN(created_at) as earliest,
                MAX(created_at) as latest
            FROM issue_cracker.comments
        """)
        emb_stats = await conn.fetchrow("""
            SELECT
                (SELECT COUNT(*) FROM issue_cracker.posts WHERE embedding IS NOT NULL) as posts_embedded,
                (SELECT COUNT(*) FROM issue_cracker.comments WHERE embedding IS NOT NULL) as comments_embedded
        """)
    
    print(f"\n📊 시드 데이터 요약:")
    print(f"   Issue: {ISSUE_TITLE} ({ISSUE_TYPE})")
    print(f"   Posts:  {len(POSTS)}건 (임베딩: {emb_stats['posts_embedded']}건)")
    print(f"   Comments: {stats['total_comments']}건 (임베딩: {emb_stats['comments_embedded']}건)")
    print(f"   기간: {stats['earliest']} ~ {stats['latest']}")
    print(f"   모델: {EMBEDDING_MODEL}")
    print(f"\n   ⚠️  analysis_results는 비어 있습니다.")
    print(f"   → AnalyzerAgent 실행 시 issue_id 기준으로 감성 분석 결과가 채워집니다.")
    print(f"\n   ✅ pgVector 검색 준비 완료!")
    
    await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Issue Cracker DB 시드 데이터 삽입")
    parser.add_argument("--dsn", required=True, help="PostgreSQL DSN (예: postgresql://user:pass@localhost:5432/db)")
    args = parser.parse_args()
    
    asyncio.run(seed(args.dsn))
