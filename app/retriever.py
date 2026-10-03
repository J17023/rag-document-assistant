"Recuperación de los fragmentos más relevantes para una pregunta."
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    "Un fragmento recuperado con su score de similitud"

    document: Document
    score: float

    @property
    def label(self) -> str:
        "Nombre de la fuente para mostrar en las citas."
        source = self.document.metadata["source"]
        page = self.document.metadata.get("page")
        return f"{source}, p.{page}" if page is not None else source


def retrieve(
    vectorstore: VectorStore, question: str, top_k: int, score_threshold: float
) -> list[RetrievedChunk]:
    "Búsqueda por similitud. Descarta los fragmentos por debajo del umbral."
    results = vectorstore.similarity_search_with_relevance_scores(question, k=top_k)
    return [
        RetrievedChunk(document=doc, score=score)
        for doc, score in results
        if score >= score_threshold
    ]