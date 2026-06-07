"""
Issue Cracker — 기존 posts/comments에 임베딩 벡터 배치 생성

DB의 embedding IS NULL인 행을 찾아 multilingual-e5-small 벡터를 생성 후 UPDATE.
E5 모델은 문서에 'passage: ' prefix를 붙여야 합니다.

사용법:
  python scripts/generate_embeddings.py --dsn postgresql://user:pass@host:5432/db
  python scripts/generate_embeddings.py --dsn ... --batch-size 200 --dry-run
"""
import sys
import os
import argparse
import asyncio
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncpg
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DIM

# ==========================================
# 설정
# ==========================================
BATCH_SIZE = 100
PASSAGE_PREFIX = "passage: "


def build_text_for_post(row: dict) -> str:
    """posts 행 → 임베딩 입력 텍스트 구성."""
    parts = []
    if row.get("title"):
        parts.append(row["title"])
    if row.get("body"):
        parts.append(row["body"][:500])
    return PASSAGE_PREFIX + " ".join(parts) if parts else ""


def build_text_for_comment(row: dict) -> str:
    """comments 행 → 임베딩 입력 텍스트 구성."""
    body = row.get("body", "")
    return PASSAGE_PREFIX + body[:500] if body else ""


async def generate_embeddings(dsn: str, batch_size: int, dry_run: bool):
    print(f">> 임베딩 모델 로드: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"   차원: {EMBEDDING_DIM}, 디바이스: {model.device}")

    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)

    # ==========================================
    # 1. Posts 임베딩 생성
    # ==========================================
    async with pool.acquire() as conn:
        pending_posts = await conn.fetch("""
            SELECT post_id, title, body
            FROM issue_cracker.posts
            WHERE embedding IS NULL
            ORDER BY created_at
        """)

    print(f"\n>> Posts: 미처리 {len(pending_posts)}건")

    for i in range(0, len(pending_posts), batch_size):
        batch = pending_posts[i:i + batch_size]
        texts = [build_text_for_post(dict(r)) for r in batch]
        texts = [t for t in texts if t]  # 빈 텍스트 제거

        if not texts:
            continue

        start = time.time()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        elapsed = time.time() - start

        print(f"   [{i+1}~{i+len(batch)}] {len(texts)}건 인코딩 완료 ({elapsed:.1f}s)")

        if not dry_run:
            async with pool.acquire() as conn:
                for j, row in enumerate(batch):
                    if j < len(embeddings):
                        vec_str = "[" + ",".join(str(float(v)) for v in embeddings[j]) + "]"
                        await conn.execute("""
                            UPDATE issue_cracker.posts
                            SET embedding = $2::vector
                            WHERE post_id = $1
                        """, row["post_id"], vec_str)

    # ==========================================
    # 2. Comments 임베딩 생성
    # ==========================================
    async with pool.acquire() as conn:
        pending_comments = await conn.fetch("""
            SELECT comment_id, body
            FROM issue_cracker.comments
            WHERE embedding IS NULL
            ORDER BY created_at
        """)

    print(f"\n>> Comments: 미처리 {len(pending_comments)}건")

    for i in range(0, len(pending_comments), batch_size):
        batch = pending_comments[i:i + batch_size]
        texts = [build_text_for_comment(dict(r)) for r in batch]
        texts = [t for t in texts if t]

        if not texts:
            continue

        start = time.time()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        elapsed = time.time() - start

        print(f"   [{i+1}~{i+len(batch)}] {len(texts)}건 인코딩 완료 ({elapsed:.1f}s)")

        if not dry_run:
            async with pool.acquire() as conn:
                for j, row in enumerate(batch):
                    if j < len(embeddings):
                        vec_str = "[" + ",".join(str(float(v)) for v in embeddings[j]) + "]"
                        await conn.execute("""
                            UPDATE issue_cracker.comments
                            SET embedding = $2::vector
                            WHERE comment_id = $1
                        """, row["comment_id"], vec_str)

    # ==========================================
    # 3. 요약
    # ==========================================
    async with pool.acquire() as conn:
        post_stats = await conn.fetchrow("""
            SELECT
                COUNT(*) AS total,
                COUNT(embedding) AS embedded
            FROM issue_cracker.posts
        """)
        comment_stats = await conn.fetchrow("""
            SELECT
                COUNT(*) AS total,
                COUNT(embedding) AS embedded
            FROM issue_cracker.comments
        """)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}📊 임베딩 현황:")
    print(f"   Posts:    {post_stats['embedded']}/{post_stats['total']} 완료")
    print(f"   Comments: {comment_stats['embedded']}/{comment_stats['total']} 완료")
    print(f"   모델: {EMBEDDING_MODEL} ({EMBEDDING_DIM}d)")

    await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Issue Cracker 임베딩 배치 생성")
    parser.add_argument("--dsn", required=True, help="PostgreSQL DSN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"배치 크기 (기본: {BATCH_SIZE})")
    parser.add_argument("--dry-run", action="store_true", help="DB 업데이트 없이 인코딩만 테스트")
    args = parser.parse_args()

    asyncio.run(generate_embeddings(args.dsn, args.batch_size, args.dry_run))
