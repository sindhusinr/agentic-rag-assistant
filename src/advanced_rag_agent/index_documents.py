from advanced_rag_agent.ingestion.ingest import ingest_documents

def main():
    # Run incremental ingestion for all PDFs in the data folder
    ingest_documents("data")
    print("Indexing complete")

if __name__ == "__main__":
    main()