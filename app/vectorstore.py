"Vector store local con Chroma"
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings


def create_vectorstore(settings: Settings, embeddings: Embeddings) -> Chroma:
    "Abre o crea la colección de Chroma usando similitud coseno."
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.vectorstore_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def count_chunks(vectorstore: Chroma) -> int:
    "Número de chunks indexados."
    return len(vectorstore.get(include=[])["ids"])


def replace_chunks(vectorstore: Chroma, chunks: list[Document]) -> None:
    "Borra el índice actual y guarda los nuevos chunks"
    existing_ids = vectorstore.get(include=[])["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)
    if chunks:
        vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])