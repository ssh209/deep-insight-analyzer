import asyncio
import sys
import os
import uuid
from datetime import datetime, timezone, timedelta
import asyncpg

# Windows 환경에서 asyncpg 사용 시 발생하는 [Errno 42] Illegal byte sequence 오류 해결
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# langchain_community 대신 langchain_huggingface를 사용하는 최신 방식으로 작성합니다.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# 테스트용 DB URL (실제 환경에 맞게 변경하세요)
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vibe_x")

# 임베딩 모델
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# ==========================================
# 1. Mock 데이터 (게시글)
# ==========================================
MOCK_POSTS = [
    {
        "platform": "community",
        "content_type": "post",
        "title": "이번 스벅 탱크데이 완전 실망임;;",
        "body": "새벽 6시부터 줄섰는데 굿즈 수량도 부족하고, 받은 텀블러는 로고가 벗겨져있음. 이럴거면 왜 이벤트를 한건지 모르겠네. 진짜 스벅 폼 많이 죽었다.",
        "author_name": "스벅호갱탈출",
        "view_count": 12500,
        "like_count": 432,
        "comment_count": 85,
    },
    {
        "platform": "twitter",
        "content_type": "tweet",
        "title": "스벅 탱크데이 현황",
        "body": "현재 강남역 스벅 상황... 대기줄 관리가 전혀 안돼서 사람들 차도로 밀려나고 난리남. 경찰까지 출동함. #스타벅스 #탱크데이_폭망",
        "author_name": "coffee_lover_99",
        "view_count": 45000,
        "like_count": 1205,
        "comment_count": 320,
    },
    {
        "platform": "news",
        "content_type": "article",
        "title": "[단독] 스타벅스 '탱크데이' 행사, 부실 운영 논란... 굿즈 불량 속출",
        "body": "스타벅스코리아가 야심차게 준비한 '탱크데이' 행사가 부실한 운영과 굿즈 품질 논란으로 도마 위에 올랐다. 일각에서는 불매 운동 움직임까지 보이고 있다...",
        "author_name": "김기자",
        "view_count": 89000,
        "like_count": 560,
        "comment_count": 412,
    },
    {
        "platform": "youtube",
        "content_type": "video",
        "title": "스벅 탱크데이 굿즈 언박싱하다 빡친 영상 🤬",
        "body": "탱크데이 한정판 텀블러 받았는데 뚜껑이 안닫힙니다 ㅋㅋㅋ 이거 불량률 실화인가요? 고객센터 연결도 안되고 어이없네요.",
        "author_name": "리뷰하는아재",
        "view_count": 250000,
        "like_count": 8500,
        "comment_count": 1200,
    },
    {
        "platform": "community",
        "content_type": "post",
        "title": "스타벅스 탱크데이 포스터 디자인 논란 (5.18 비하?)",
        "body": "이번 탱크데이 포스터에 들어간 문구랑 탱크 이미지가 5.18 민주화 운동을 조롱하는 극우 커뮤니티 밈이랑 너무 똑같은데? 이거 기획한 사람 누구냐...",
        "author_name": "역사학도",
        "view_count": 56000,
        "like_count": 3400,
        "comment_count": 890,
    }
]

# ==========================================
# 2. Mock 데이터 (댓글)
# ==========================================
MOCK_COMMENTS_TEMPLATES = [
    "진짜 퀄리티 실화냐... 예전의 스벅이 아님",
    "이거 환불 안되나요? 고객센터 전화 폭주해서 연결도 안됨 ㅠㅠ",
    "불매운동 갑시다! 소비자를 개돼지로 아는듯",
    "포스터 디자인은 진짜 선 넘었지. 담당자 징계해야함",
    "아침에 연차쓰고 간 내 시간이 너무 아깝다",
    "프리퀀시 때부터 알아봤다 ㅉㅉ",
]

async def seed_data():
    print(f"1. DB 연결 시도 중... ({DATABASE_URL})")
    try:
        # 윈도우 환경 로컬 DB의 경우 SSL 협상 오류([Errno 42])가 빈번하여 ssl=False를 명시합니다.
        pool = await asyncpg.create_pool(DATABASE_URL, ssl=False)
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        print("-> DATABASE_URL 환경변수를 올바르게 설정하거나 로컬 DB를 기동해주세요.")
        return

    print("2. 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("3. 데이터 임베딩 및 DB Insert 시작...")
    now = datetime.now(timezone.utc)
    
    async with pool.acquire() as conn:
        for i, post in enumerate(MOCK_POSTS):
            post_id = f"mock_post_{uuid.uuid4().hex[:8]}"
            
            # 임베딩 생성 (문맥을 잘 잡도록 title과 body 결합, E5 모델에 맞게 passage: prefix 추가)
            text_to_embed = f"passage: {post['title']} {post['body']}"
            vector = embeddings.embed_query(text_to_embed)
            vec_str = "[" + ",".join(str(v) for v in vector) + "]"
            
            # 게시글 삽입
            await conn.execute("""
                INSERT INTO issue_cracker.posts 
                (post_id, created_at, platform, content_type, url, title, body, author_id, author_name, view_count, like_count, comment_count, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::vector)
            """, 
            post_id, 
            now - timedelta(hours=i), # 시간차를 조금씩 둠
            post["platform"], 
            post["content_type"],
            f"https://example.com/posts/{post_id}",
            post["title"],
            post["body"],
            f"user_{uuid.uuid4().hex[:6]}",
            post["author_name"],
            post["view_count"],
            post["like_count"],
            post["comment_count"],
            vec_str
            )
            print(f"  [+] Post Inserted: {post['title']} (post_id: {post_id})")
            
            # 댓글 삽입 (각 게시글당 3개씩)
            for j in range(3):
                comment_id = f"mock_comment_{uuid.uuid4().hex[:8]}"
                comment_body = MOCK_COMMENTS_TEMPLATES[(i + j) % len(MOCK_COMMENTS_TEMPLATES)]
                await conn.execute("""
                    INSERT INTO issue_cracker.comments 
                    (comment_id, post_id, created_at, body, author_id, like_count)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                comment_id,
                post_id,
                now - timedelta(hours=i, minutes=j*10),
                comment_body,
                f"user_{uuid.uuid4().hex[:6]}",
                10 + j*5
                )
    
    await pool.close()
    print("\n✅ Mock 데이터 적재가 완료되었습니다!")

if __name__ == "__main__":
    asyncio.run(seed_data())
