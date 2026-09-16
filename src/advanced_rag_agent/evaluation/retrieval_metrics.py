def _unique_sections(documents):
    # A section can produce multiple chunks, so count each section only once
    seen = set()
    sections = []

    for doc in documents:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("section"),
        )

        if key not in seen:
            seen.add(key)
            sections.append(key)

    return sections


def _relevant_sections(relevant_sections):
    # Convert ground-truth dictionaries into comparable (source, section) pairs
    return {
        (item["source"], item["section"])
        for item in relevant_sections
    }


def precision_at_k(documents, relevant_sections, k):
    retrieved = _unique_sections(documents)[:k]
    relevant = _relevant_sections(relevant_sections)

    if not retrieved:
        return 0.0

    relevant_retrieved = sum(
        1 for section in retrieved if section in relevant
    )

    return relevant_retrieved / len(retrieved)


def recall_at_k(documents, relevant_sections, k):
    retrieved = _unique_sections(documents)[:k]
    relevant = _relevant_sections(relevant_sections)

    if not relevant:
        return 0.0

    relevant_retrieved = sum(
        1 for section in relevant if section in retrieved
    )

    return relevant_retrieved / len(relevant)


def mrr(documents, relevant_sections):
    retrieved = _unique_sections(documents)
    relevant = _relevant_sections(relevant_sections)

    # Reciprocal rank of the first relevant retrieved section
    for rank, section in enumerate(retrieved, start=1):
        if section in relevant:
            return 1 / rank

    return 0.0