from langchain_huggingface import HuggingFaceEmbeddings
from advanced_rag_agent.config.settings import EMBEDDING_MODEL
#lazy initialization
_embedding_model = None

def get_embedding_model():
    global _embedding_model

    if not EMBEDDING_MODEL:
        raise ValueError(
            "EMBEDDING_MODEL is not configured."
        )

    # Load model only when it is actually needed
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    return _embedding_model