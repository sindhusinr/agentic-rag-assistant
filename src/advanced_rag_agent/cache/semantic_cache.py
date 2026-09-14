import hashlib
import json
import os

from sklearn.metrics.pairwise import cosine_similarity

from advanced_rag_agent.cache.redis_client import get_redis_client
from advanced_rag_agent.ingestion.embedder import get_embedding_model

CACHE_TTL = int(os.getenv("CACHE_TTL", 86400))


def build_query_hash(query: str) -> str:
    # Create a stable ID for each query
    return hashlib.sha256(query.encode()).hexdigest()


def build_cache_key(
    query: str,
    kb_version: str,
    route: str,
) -> str:
    query_hash = build_query_hash(query)
    # Separate cache entries by route
    return f"rag_cache:{route}:{kb_version}:{query_hash}"


def build_metadata_key(
    query_hash: str,
    kb_version: str,
    route: str,
) -> str:
    return f"rag_cache_meta:{route}:{kb_version}:{query_hash}"


def build_index_key(
    kb_version: str,
    route: str,
) -> str:
    # Each route gets its own semantic-cache index
    return f"rag_cache_queries:{route}:{kb_version}"


def get_exact_cached_result(
    query: str,
    kb_version: str,
    route: str,
):
    redis_client = get_redis_client()

    cache_key = build_cache_key(
        query=query,
        kb_version=kb_version,
        route=route,
    )

    cached_value = redis_client.get(cache_key)

    if not cached_value:
        return None

    return json.loads(cached_value)


def get_semantic_cached_result(
    query: str,
    kb_version: str,
    route: str,
    threshold: float = 0.85,
):
    redis_client = get_redis_client()
    embedding_model = get_embedding_model()

    query_embedding = embedding_model.embed_query(query)

    index_key = build_index_key(
        kb_version=kb_version,
        route=route,
    )

    query_hashes = redis_client.smembers(index_key)

    best_result = None
    best_score = 0

    for query_hash in query_hashes:
        metadata_key = build_metadata_key(
            query_hash=query_hash,
            kb_version=kb_version,
            route=route,
        )

        result_key = (
            f"rag_cache:{route}:"
            f"{kb_version}:{query_hash}"
        )

        metadata_value = redis_client.get(metadata_key)
        result_value = redis_client.get(result_key)

        # Remove expired entries from semantic index
        if not metadata_value or not result_value:
            redis_client.srem(index_key, query_hash)
            continue

        metadata = json.loads(metadata_value)

        score = cosine_similarity(
            [query_embedding],
            [metadata["embedding"]],
        )[0][0]

        if score > best_score:
            best_score = score
            best_result = json.loads(result_value)

    if best_result and best_score >= threshold:
        print(
            f"Semantic cache hit "
            f"[{route}]: {best_score:.2f}"
        )
        return best_result

    print(f"Semantic cache miss [{route}]")
    return None


def get_cached_result(
    query: str,
    kb_version: str,
    route: str,
    threshold: float = 0.85,
):
    # Exact lookup is cheaper, so check it first
    exact_result = get_exact_cached_result(
        query=query,
        kb_version=kb_version,
        route=route,
    )

    if exact_result:
        print(f"Exact cache hit [{route}]")
        return exact_result

    # Semantic lookup stays inside the same route
    return get_semantic_cached_result(
        query=query,
        kb_version=kb_version,
        route=route,
        threshold=threshold,
    )


def save_to_cache(
    query: str,
    result: dict,
    kb_version: str,
    route: str,
):
    redis_client = get_redis_client()
    embedding_model = get_embedding_model()

    query_hash = build_query_hash(query)

    result_key = build_cache_key(
        query=query,
        kb_version=kb_version,
        route=route,
    )

    metadata_key = build_metadata_key(
        query_hash=query_hash,
        kb_version=kb_version,
        route=route,
    )

    index_key = build_index_key(
        kb_version=kb_version,
        route=route,
    )

    # Store query embedding for semantic comparison
    embedding = embedding_model.embed_query(query)

    metadata = {
        "query": query,
        "route": route,
        "embedding": embedding,
    }

    # Store answer/result with TTL
    redis_client.setex(
        result_key,
        CACHE_TTL,
        json.dumps(result),
    )

    # Store semantic metadata with same TTL
    redis_client.setex(
        metadata_key,
        CACHE_TTL,
        json.dumps(metadata),
    )

    # Keep semantic index for this route
    redis_client.sadd(
        index_key,
        query_hash,
    )

    redis_client.expire(
        index_key,
        CACHE_TTL,
    )