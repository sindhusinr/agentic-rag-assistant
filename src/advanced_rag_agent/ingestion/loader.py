from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    """
    Load a single PDF and return LangChain documents.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = path.name

    return documents


def load_pdfs(data_dir: str):
    """
    Load all PDF files from a directory.
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    documents = []

    for pdf_path in data_path.glob("*.pdf"):
        pdf_documents = load_pdf(str(pdf_path))
        documents.extend(pdf_documents)

    if not documents:
        raise ValueError(
            f"No PDF files found in: {data_dir}"
        )

    return documents