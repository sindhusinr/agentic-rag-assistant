# Advanced_rag_agent
Advanced RAG Agent using LangGraph
to activate venv 
.venv\Scripts\Activate.ps1

uv run python src/app.py
uv run python -m advanced_rag_agent.app.py

to index 
uv run python -m advanced_rag_agent.index_documents

# To Run streamlit
uv run streamlit run src/advanced_rag_agent/app_streamlit.py
streamlit run src/advanced_rag_agent/app_streamlit.py

User Query
    ↓
Input Guardrail
    ↓
Semantic Cache
    ↓
Query Router
   /       \
General     KB
  ↓          ↓
Direct     Hybrid Retrieval
Answer      Vector + BM25
               ↓
              RRF
               ↓
          Cross-Encoder
               ↓
            Top 3
               ↓
        Evidence Grader
          /         \
  sufficient      insufficient
      ↓                ↓
 RAG Answer        Rewrite once
                       ↓
                    Retry
                   /     \
              sufficient  insufficient
                   ↓          ↓
               RAG Answer   Safe fallback
                    \        /
                     ↓
              Output Guardrail
                     ↓
                 Redis Cache
                     ↓
                    END



START
  ↓
prepare_query
  ↓
input_guardrail
  ├── blocked ──────────────────────────────┐
  ↓                                        │
cache_lookup                               │
  ├── hit ─────────────────────────────┐    │
  ↓                                    │    │
route_question                         │    │
  ├── general                          │    │
  │      ↓                             │    │
  │   general_answer                   │    │
  │                                    │    │
  └── kb                               │    │
         ↓                             │    │
      retrieve                         │    │
         ↓                             │    │
      rerank                           │    │
         ↓                             │    │
      grade                            │    │
       ├── sufficient                  │    │
       │       ↓                       │    │
       │   KB answer                   │    │
       │                               │    │
       └── insufficient                │    │
               ↓                       │    │
           rewrite once                │    │
               ↓                       │    │
           retrieve again              │    │
               ↓                       │    │
             rerank                    │    │
               ↓                       │    │
             grade                     │    │
               ├── sufficient → answer │    │
               └── insufficient        │    │
                       ↓               │    │
                controlled fallback    │    │
                                       ↓    ↓
                               output_guardrail
                                       ↓
                                   save_cache
                                       ↓
                                  final_message
                                       ↓
                                      END