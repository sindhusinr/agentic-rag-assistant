import os
import redis
from dotenv import load_dotenv

load_dotenv(override=True)

REDIS_URL = os.getenv("REDIS_URL")

_redis_client = None

def get_redis_client():
    global _redis_client

    if not REDIS_URL:
        raise ValueError("REDIS_URL is not configured.")

    # Create the Redis connection only when needed
    if _redis_client is None:
        _redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
        )

    return _redis_client