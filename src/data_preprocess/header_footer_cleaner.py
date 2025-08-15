from collections import Counter
from langchain.schema import Document
from typing import List


def remove_repeating_headers_footers(
    docs: List[Document], min_repeat: int = 2, max_line_len: int = 100
) -> List[Document]:
    """
    Remove headers/footers that repeat across pages in LangChain Document objects.

    Args:
        docs: List of LangChain Document objects
        min_repeat: Minimum number of times a line must repeat to be considered a header/footer
        max_line_len: Ignore very long lines (likely content, not headers/footers)

    Returns:
        List of cleaned Document objects
    """
    if not docs:
        return docs

    # Extract first and last lines of each document/chunk
    first_lines = []
    last_lines = []

    for doc in docs:
        lines = doc.page_content.strip().splitlines()
        if lines:
            # Check first line
            if len(lines[0]) <= max_line_len:
                first_lines.append(lines[0].strip())
            # Check last line
            if len(lines) > 1 and len(lines[-1]) <= max_line_len:
                last_lines.append(lines[-1].strip())

    # Count frequency of first and last lines
    first_line_counts = Counter(first_lines)
    last_line_counts = Counter(last_lines)

    # Identify headers and footers to remove
    headers_to_remove = {
        line
        for line, count in first_line_counts.items()
        if count >= min_repeat and line
    }
    footers_to_remove = {
        line for line, count in last_line_counts.items() if count >= min_repeat and line
    }

    print(
        f"----IDENTIFIED {len(headers_to_remove)} HEADERS AND {len(footers_to_remove)} FOOTERS TO REMOVE----"
    )
    if headers_to_remove:
        print("Headers to remove:", list(headers_to_remove)[:3])  # Show first 3
    if footers_to_remove:
        print("Footers to remove:", list(footers_to_remove)[:3])  # Show first 3

    # Clean documents
    cleaned_docs = []
    removed_headers = 0
    removed_footers = 0

    for doc in docs:
        lines = doc.page_content.strip().splitlines()
        original_line_count = len(lines)

        # Remove matching header (first line)
        if lines and lines[0].strip() in headers_to_remove:
            lines.pop(0)
            removed_headers += 1

        # Remove matching footer (last line)
        if lines and lines[-1].strip() in footers_to_remove:
            lines.pop(-1)
            removed_footers += 1

        # Only keep document if it still has content
        if lines:
            cleaned_content = "\n".join(lines).strip()
            if cleaned_content and len(cleaned_content) > 10:  # Ensure substantial content remains
                cleaned_docs.append(
                    Document(page_content=cleaned_content, metadata=doc.metadata.copy())
                )
        else:
            # If cleaning removed all content, keep original
            if doc.page_content.strip():
                cleaned_docs.append(Document(page_content=doc.page_content, metadata=doc.metadata.copy()))

    print(
        f"----REMOVED {removed_headers} HEADERS AND {removed_footers} FOOTERS FROM {len(docs)} DOCUMENTS----"
    )
    print(f"----KEPT {len(cleaned_docs)} DOCUMENTS AFTER CLEANING----")

    return cleaned_docs


def clean_chunked_documents(
    doc_splits: List[Document], min_repeat: int = 2, max_line_len: int = 100
) -> List[Document]:
    """
    Clean headers and footers from already chunked/split documents.

    Args:
        doc_splits: List of chunked/split Document objects
        min_repeat: Minimum number of times a line must repeat to be considered a header/footer
        max_line_len: Ignore very long lines (likely content, not headers/footers)

    Returns:
        List of cleaned Document objects
    """
    print(f"----CLEANING {len(doc_splits)} CHUNKED DOCUMENTS----")
    return remove_repeating_headers_footers(doc_splits, min_repeat, max_line_len)
