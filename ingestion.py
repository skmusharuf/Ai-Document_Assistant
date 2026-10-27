import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR
)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingest_pdf(uploaded_file):
    """Save an uploaded PDF, load pages, split, and upsert chunks."""
    filename = Path(uploaded_file.name).name
    destination = UPLOAD_DIR / filename
    uploaded_file.seek(0)
    destination.write_bytes(uploaded_file.getvalue())

    pages = PyPDFLoader(str(destination)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)

    if not chunks or not any(chunk.page_content.strip() for chunk in chunks):
        raise ValueError(
            "No extractable text found. This may be a scanned PDF; OCR is not included."
        )

    ids = []
    for chunk in chunks:
        page = chunk.metadata.get("page", 0)
        start_index = chunk.metadata.get("start_index", 0)
        digest = hashlib.sha256(
            f"{filename}|{page}|{start_index}|{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        ids.append(digest)
        chunk.metadata["source"] = filename
        # PyPDFLoader page metadata is zero-based; show page numbers as one-based in UI.
        chunk.metadata["page_number"] = int(page) + 1

    store = get_vector_store()
    store.add_documents(documents=chunks, ids=ids)
    return len(chunks)


def indexed_chunk_count():
    return get_vector_store()._collection.count()
