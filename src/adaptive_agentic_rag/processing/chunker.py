from langchain_text_splitters import RecursiveCharacterTextSplitter

from adaptive_agentic_rag.schemas import (
    Document,
    Chunk
)



def split_paragraphs(text: str) -> list[str]:
    """
    Split document into paragraphs.
    """

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    return paragraphs



def merge_small_paragraphs(
    paragraphs: list[str],
    min_words: int = 20
) -> list[str]:
    """
    Merge very small paragraphs with previous content.
    """

    merged = []

    buffer = ""


    for paragraph in paragraphs:

        words = len(
            paragraph.split()
        )


        if words < min_words:

            buffer += " " + paragraph

        else:

            if buffer:

                paragraph = (
                    buffer.strip()
                    + " "
                    + paragraph
                )

                buffer = ""


            merged.append(
                paragraph.strip()
            )


    if buffer:

        merged.append(
            buffer.strip()
        )


    return merged



def chunk_document(
    document: Document,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    min_chunk_words: int = 20
) -> list[Chunk]:


    paragraphs = split_paragraphs(
        document.text
    )


    paragraphs = merge_small_paragraphs(
        paragraphs,
        min_chunk_words
    )


    text = "\n\n".join(
        paragraphs
    )


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


    texts = splitter.split_text(
        text
    )


    chunks = []


    for idx, chunk_text in enumerate(texts):


        word_count = len(
            chunk_text.split()
        )


        if word_count < min_chunk_words:
            continue



        chunks.append(

            Chunk(
                id=f"{document.id}_chunk_{idx}",

                document_id=document.id,

                text=chunk_text,

                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "word_count": word_count
                }
            )

        )


    return chunks