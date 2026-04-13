"""Redis client configuration."""

import redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from .config import get_settings
from .logging_config import setup_logging

logger = setup_logging("redis-client")


def get_redis_client() -> redis.Redis:
    """Get a configured Redis client.

    Returns:
        Configured Redis client instance
    """
    settings = get_settings()

    client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
    )

    return client


def test_redis_connection() -> bool:
    """Test Redis connectivity.

    Returns:
        True if Redis is reachable, False otherwise
    """
    try:
        client = get_redis_client()
        return client.ping()
    except (RedisConnectionError, RedisError) as e:
        logger.error(f"Redis connection failed: {e}")
        return False


def store_task_result(task_id: str, result: dict) -> bool:
    """Store task result in Redis.

    Args:
        task_id: Celery task ID
        result: Task result data

    Returns:
        True if stored successfully
    """
    try:
        import json
        client = get_redis_client()
        client.setex(
            f"task:{task_id}",
            86400,  # 24 hours TTL
            json.dumps(result),
        )
        return True
    except RedisError as e:
        logger.error(f"Failed to store task result: {e}")
        return False


def get_task_result(task_id: str) -> dict | None:
    """Get task result from Redis.

    Args:
        task_id: Celery task ID

    Returns:
        Task result dict or None
    """
    try:
        import json
        client = get_redis_client()
        data = client.get(f"task:{task_id}")
        if data:
            return json.loads(data)
        return None
    except RedisError as e:
        logger.error(f"Failed to get task result: {e}")
        return None
