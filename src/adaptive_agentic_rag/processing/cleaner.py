import re

from adaptive_agentic_rag.schemas import Document



def clean_text(text: str) -> str:
    """
    Basic text normalization.
    """

    # حذف فاصله‌های اضافی
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )


    # محدود کردن newline های پشت سر هم
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    # حذف فاصله ابتدا و انتها
    text = text.strip()


    return text



def clean_document(
    document: Document
) -> Document:

    cleaned_text = clean_text(
        document.text
    )


    return Document(
        id=document.id,

        text=cleaned_text,

        metadata=document.metadata
    )