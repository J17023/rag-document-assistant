"Pruebas de la división en fragmentos (app/chunking.py)."
from langchain_core.documents import Document

from app.chunking import split_documents


def make_docs() -> list[Document]:
    "Un documento largo (PDF) y uno corto (Markdown) para dividir."
    long_text = "Párrafo de prueba con varias palabras. " * 40  # ~1600 caracteres
    return [
        Document(page_content=long_text, metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Texto corto", metadata={"source": "b.md"}),
    ]


def test_chunks_respect_max_size():
    "Ningún fragmento supera el tamaño máximo."
    chunks = split_documents(make_docs(), chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 2
    assert all(len(c.page_content) <= 300 for c in chunks)


def test_chunks_keep_original_metadata():
    "Los fragmentos conservan el documento y la página de origen."
    chunks = split_documents(make_docs(), chunk_size=300, chunk_overlap=50)

    pdf_chunks = [c for c in chunks if c.metadata["source"] == "a.pdf"]
    assert all(c.metadata["page"] == 1 for c in pdf_chunks)


def test_chunk_ids_are_unique_and_readable():
    "Cada fragmento tiene un ID único con el formato 'archivo#n'."
    chunks = split_documents(make_docs(), chunk_size=300, chunk_overlap=50)
    ids = [c.metadata["chunk_id"] for c in chunks]

    assert len(ids) == len(set(ids))
    assert ids[0] == "a.pdf#1"
    assert "b.md#1" in ids