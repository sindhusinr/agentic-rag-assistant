from advanced_rag_agent.ingestion.loader import load_pdfs
from advanced_rag_agent.ingestion.chunker import chunk_documents

# Load all PDFs from the data folder
documents = load_pdfs("data")
chunks = chunk_documents(documents)

print(f"Total pages loaded: {len(documents)}")
print(f"Total chunks created: {len(chunks)}")

# Inspect every generated chunk
for i, chunk in enumerate(chunks, start=1):
    print("\n" + "=" * 80)
    print(f"CHUNK {i}")
    print(f"Source: {chunk.metadata.get('source')}")
    print(f"Page: {chunk.metadata.get('page')}")
    print(f"Section: {chunk.metadata.get('section')}")
    print(f"Length: {len(chunk.page_content)} characters")
    print("-" * 80)
    print(chunk.page_content)