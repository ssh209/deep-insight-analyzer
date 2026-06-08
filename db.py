"""
DB 연결 풀 관리 모듈 — asyncpg 기반 PostgreSQL 연결.

설계 원칙:
  - RDB는 외부에서 관리 (Neon, Supabase, Aurora 등)
  - 이 모듈은 연결만 담당, 프로비저닝은 별도 프로세스
  - DATABASE_URL이 비어있으면 None 반환 → CSV 모드 폴백
  - 연결 실패 시 앱 크래시 방지 → 경고 로그 + CSV 모드 폴백
"""
import ssl
import logging
from typing import Optional

import asyncpg

from config import DATABASE_URL, DB_MIN_CONNECTIONS, DB_MAX_CONNECTIONS, DB_SSL

logger = logging.getLogger(__name__)


async def create_db_pool() -> Optional[asyncpg.Pool]:
    """DATABASE_URL 환경변수 기반으로 asyncpg 커넥션 풀을 생성합니다.

    Returns:
        asyncpg.Pool: 연결 성공 시 풀 객체
        None: DATABASE_URL 미설정 또는 연결 실패 시 (CSV 모드 폴백)
    """
    if not DATABASE_URL:
        logger.info("DATABASE_URL 미설정 → CSV 직접 입력 모드로 동작합니다.")
        return None

    try:
        # SSL 컨텍스트 설정 (Neon, Supabase 등 클라우드 DB 필수)
        ssl_context = None
        if DB_SSL:
            ssl_context = ssl.create_default_context()
            # 클라우드 DB는 자체 서명 인증서를 사용하므로 검증 완화
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=DB_MIN_CONNECTIONS,
            max_size=DB_MAX_CONNECTIONS,
            ssl=ssl_context,
            command_timeout=30,
            # 연결 풀에서 커넥션 획득 대기 최대 시간
            timeout=10,
        )

        # 연결 테스트
        async with pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            logger.info(f"✅ PostgreSQL 연결 성공: {version[:60]}...")

        logger.info(
            f"✅ DB 풀 생성 완료 (min={DB_MIN_CONNECTIONS}, max={DB_MAX_CONNECTIONS})"
        )
        return pool

    except asyncpg.InvalidCatalogNameError as e:
        logger.error(f"❌ DB 연결 실패 — 데이터베이스가 존재하지 않습니다: {e}")
        logger.warning("→ CSV 직접 입력 모드로 폴백합니다.")
        return None

    except asyncpg.InvalidPasswordError as e:
        logger.error(f"❌ DB 연결 실패 — 인증 오류: {e}")
        logger.warning("→ CSV 직접 입력 모드로 폴백합니다.")
        return None

    except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as e:
        logger.error(f"❌ DB 연결 실패: {type(e).__name__}: {e}")
        logger.warning("→ CSV 직접 입력 모드로 폴백합니다.")
        return None

    except Exception as e:
        logger.error(f"❌ DB 연결 중 예상치 못한 오류: {type(e).__name__}: {e}")
        logger.warning("→ CSV 직접 입력 모드로 폴백합니다.")
        return None


async def close_db_pool(pool: Optional[asyncpg.Pool]) -> None:
    """앱 종료 시 커넥션 풀을 안전하게 해제합니다."""
    if pool is not None:
        await pool.close()
        logger.info("✅ DB 풀 종료 완료")


async def check_schema(pool: asyncpg.Pool) -> bool:
    """issue_cracker 스키마와 핵심 테이블 존재 여부를 확인합니다.

    Returns:
        True: 스키마 정상
        False: 스키마 누락 (DDL 실행 필요)
    """
    try:
        async with pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'issue_cracker'
                      AND table_name = 'issues'
                )
            """)
            if result:
                logger.info("✅ issue_cracker 스키마 확인 완료")
            else:
                logger.warning(
                    "⚠️ issue_cracker 스키마가 없습니다. "
                    "sql/001_create_tables.sql을 실행하세요."
                )
            return result
    except Exception as e:
        logger.error(f"❌ 스키마 확인 실패: {e}")
        return False


# asyncio import는 예외 핸들링에서만 사용
import asyncio  # noqa: E402
