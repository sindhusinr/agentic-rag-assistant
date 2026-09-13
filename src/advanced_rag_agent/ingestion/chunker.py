import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_SECTION_SIZE = 1200

def get_section_number(line):
    # Extract section numbers like 10, 10.1, 18.6
    match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", line)
    return match.group(1) if match else None

def is_heading(line, current_section, source):
    # Ignore time values like "10.30 a.m."
    if re.match(r"^\d{1,2}\.\d{2}\s*(a\.m\.|p\.m\.)", line, re.I):
        return False

    # Ignore table rows like "5 years INR 10,000"
    if re.match(r"^\d+\s+years?\b", line, re.I):
        return False

    # Leave Policy has a slightly different heading structure
    if source == "Leave policy.pdf":
        if line in {
            "Maternity Leave (ML)",
            "Other Parental Leave",
            "B. General Guidelines",
        }:
            return True

        return bool(
            re.match(r"^[1-5]\.\s+[A-Za-z].+", line)
        )

    # Detect numbered handbook headings
    match = re.match(
        r"^(\d+(?:\.\d+)*)\.?\s+[A-Za-z].+",
        line,
    )

    if not match:
        return False

    new_number = match.group(1)
    current_number = get_section_number(
        current_section or ""
    )

    # Keep nested numbering like "1. Eligibility"
    # inside sections such as "18.6 Mobile Devices"
    if (
        current_number
        and "." in current_number
        and "." not in new_number
    ):
        current_major = int(
            current_number.split(".")[0]
        )

        if int(new_number) < current_major:
            return False

    return True

def process_section(
    section,
    content,
    metadata,
    splitter,
):
    if not section or not content:
        return []

    body = "\n".join(
        content[1:]
    ).strip()

    # Ignore heading-only sections
    if len(body) < 50:
        return []

    section_text = "\n".join(content)

    section_document = Document(
        page_content=section_text,
        metadata={
            **metadata,
            "section": section,
        },
    )

    # Keep small sections as one chunk
    if len(section_text) <= MAX_SECTION_SIZE:
        return [section_document]

    # Split only large sections
    split_texts = splitter.split_text(
        section_text
    )

    chunks = []

    for text in split_texts:
        # Preserve the section heading in every child chunk
        if not text.startswith(section):
            text = f"{section}\n{text}"

        chunks.append(
            Document(
                page_content=text,
                metadata={
                    **metadata,
                    "section": section,
                },
            )
        )

    return chunks

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    all_chunks = []

    current_section = None
    current_content = []
    current_metadata = {}
    current_source = None

    for document in documents:
        source = document.metadata.get(
            "source"
        )

        page = document.metadata.get(
            "page",
            0,
        )

        # Finish previous PDF before processing next PDF
        if (
            current_source
            and source != current_source
        ):
            all_chunks.extend(
                process_section(
                    current_section,
                    current_content,
                    current_metadata,
                    splitter,
                )
            )

            current_section = None
            current_content = []

        current_source = source

        for line in document.page_content.splitlines():
            line = line.strip()

            # Remove empty lines and PDF footer text
            if (
                not line
                or line == "Classified as Internal"
            ):
                continue

            if is_heading(
                line,
                current_section,
                source,
            ):
                # Finish previous section
                all_chunks.extend(
                    process_section(
                        current_section,
                        current_content,
                        current_metadata,
                        splitter,
                    )
                )

                # Start new section
                current_section = line
                current_content = [line]

                current_metadata = {
                    **document.metadata,
                    "page_start": page,
                    "page_end": page,
                }

            elif current_section:
                # Continue current section across PDF pages
                current_content.append(line)
                current_metadata["page_end"] = page

    # Process the final section
    all_chunks.extend(
        process_section(
            current_section,
            current_content,
            current_metadata,
            splitter,
        )
    )

    return all_chunks