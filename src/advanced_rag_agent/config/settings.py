import os

from dotenv import load_dotenv

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

RERANK_MODEL = os.getenv("RERANK_MODEL")

CHROMA_PATH = os.getenv("CHROMA_PATH")

TOP_K = int(os.getenv("TOP_K", 10))
FINAL_K = int(os.getenv("FINAL_K", 3))
BROAD_K = int(os.getenv("BROAD_K", 6))

SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", 0.65)
)