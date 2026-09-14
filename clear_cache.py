from advanced_rag_agent.cache.redis_client import get_redis_client

redis_client = get_redis_client()

# Delete only this project's RAG cache keys
patterns = [
    "rag_cache:*",
    "rag_cache_meta:*",
    "rag_cache_queries:*",
]

deleted = 0

for pattern in patterns:
    for key in redis_client.scan_iter(match=pattern):
        redis_client.delete(key)
        deleted += 1

print(f"Deleted {deleted} RAG cache keys.")