# Advanced_rag_agent
Advanced RAG Agent using LangGraph

uv run python src/app.py
uv run python -m advanced_rag_agent.app.py

to index 
uv run python -m advanced_rag_agent.index_documents

# To Run streamlit
uv run streamlit run src/advanced_rag_agent/app_streamlit.py
streamlit run src/advanced_rag_agent/app_streamlit.py

          ┌──────────────────┐
                         │   PDF Upload UI  │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Document Ingestion Node  │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │ Chunking Node            │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │ Embedding Generation     │
                    │ BGE Small               │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │ Chroma Vector Store      │
                    └──────────────────────────┘


══════════════════════════════════════════════════

                         USER QUESTION
                                │
                                ▼

                  ┌──────────────────────────┐
                  │ Chat History Memory      │
                  │ Conversation Context     │
                  └──────────┬───────────────┘
                             │
                             ▼

                  ┌──────────────────────────┐
                  │ Query Enhancement Node   │
                  │ Multi Query Generation   │
                  └──────────┬───────────────┘
                             │
                             ▼

                  ┌──────────────────────────┐
                  │ Conversational RAG Node  │
                  │ Resolve References       │
                  │ ("it", "that", etc.)     │
                  └──────────┬───────────────┘
                             │
                             ▼

                  ┌──────────────────────────┐
                  │ Hybrid Retrieval Node    │
                  │ Chroma + BM25            │
                  └──────────┬───────────────┘
                             │
                             ▼

                  ┌──────────────────────────┐
                  │ Re-ranker Node           │
                  │ BGE-Reranker             │
                  └──────────┬───────────────┘
                             │
                             ▼

                  ┌──────────────────────────┐
                  │ Relevance Validator      │
                  └───────┬──────────────────┘
                          │
                          ▼

                  ┌──────────────────────────┐
                  │ Context Builder          │
                  │ Compression + Packing    │
                  └──────────┬───────────────┘
                             │
                             ▼

                  ┌──────────────────────────┐
                  │ LLM Generation Node      │
                  │ Llama 3.1                │
                  └──────────┬───────────────┘
                             │
                             ▼

                  ┌──────────────────────────┐
                  │ Citation Generator       │
                  │ Source + Page Numbers    │
                  └──────────┬───────────────┘
                             │
                             ▼

                  ┌──────────────────────────┐
                  │ Streaming Response Node  │
                  │ Token Streaming          │
                  └──────────┬───────────────┘
                             │
                             ▼

                         ANSWER
                             │
                             ▼

                  ┌──────────────────────────┐
                  │ Human Feedback Node      │
                  │ 👍 / 👎 / Comments       │
                  └──────────┬───────────────┘
                             │
                             ▼

                  ┌──────────────────────────┐
                  │ Feedback Memory Store    │
                  └──────────────────────────┘

src/
└── advanced_rag_agent/
    │
    ├── __init__.py
    │
    ├── app.py
    │
    ├── config/
    │   ├── __init__.py
    │   ├── settings.py
    │   └── prompts.py
    │
    ├── ingestion/
    │   ├── __init__.py
    │   ├── loader.py
    │   ├── chunker.py
    │   ├── embedder.py
    │   └── indexer.py
    │
    ├── retrieval/
    │   ├── __init__.py
    │   ├── vector_retriever.py
    │   ├── bm25_retriever.py
    │   ├── hybrid_retriever.py
    │   ├── query_expander.py
    │   └── reranker.py
    │
    ├── generation/
    │   ├── __init__.py
    │   ├── llm.py
    │   ├── context_builder.py
    │   └── citation_generator.py
    │
    ├── memory/
    │   ├── __init__.py
    │   ├── chat_memory.py
    │   └── feedback_store.py
    │
    ├── hitl/
    │   ├── __init__.py
    │   ├── clarification.py
    │   └── confidence_checker.py
    │
    ├── graph/
    │   ├── __init__.py
    │   ├── state.py
    │   ├── nodes.py
    │   └── rag_graph.py
    │
    └── ui/
        ├── __init__.py
        └── streamlit_app.py