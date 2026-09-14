from langchain_core.documents import Document


def build_context(documents: list[Document]) -> str:
    context_parts = []

    for doc in documents:
        metadata = doc.metadata

        source = metadata.get("source", "Unknown")
        section = metadata.get("section", "Unknown")
        page_start = metadata.get("page_start", 0)
        page_end = metadata.get("page_end", page_start)

        # PDF page metadata is zero-based; show human-readable page numbers
        page_start += 1
        page_end += 1

        if page_start == page_end:
            page_info = f"Page: {page_start}"
        else:
            page_info = f"Pages: {page_start}-{page_end}"

        # Keep citation metadata attached to each retrieved chunk
        context_parts.append(
            f"[Source: {source} | Section: {section} | {page_info}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(context_parts)