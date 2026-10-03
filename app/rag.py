"Servicio RAG: une carga, indexación, recuperación y generación"
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.chunking import split_documents
from app.config import Settings, get_settings
from app.generator import generate_answer
from app.loaders import SUPPORTED_EXTENSIONS, load_directory
from app.prompts import NOT_FOUND_MESSAGE
from app.retriever import RetrievedChunk, retrieve
from app.vectorstore import count_chunks, create_vectorstore, replace_chunks

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    documents: int
    chunks: int


@dataclass
class RAGAnswer:
    question: str
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)


class RAGService:
    def __init__(self, settings: Settings, embeddings: Embeddings, llm: BaseChatModel):
        self.settings = settings
        self.llm = llm
        self.vectorstore = create_vectorstore(settings, embeddings)

    def ingest(self) -> IngestionStats:
        "Carga todos los documentos, los divide en chunks y reconstruye el índice."
        documents = load_directory(self.settings.docs_dir)
        chunks = split_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        replace_chunks(self.vectorstore, chunks)

        n_files = len({doc.metadata["source"] for doc in documents})
        logger.info("Indexados %d documentos en %d chunks", n_files, len(chunks))
        return IngestionStats(documents=n_files, chunks=len(chunks))

    def ensure_index(self) -> None:
        "Indexa automáticamente si el vector store está vacío."
        if count_chunks(self.vectorstore) == 0:
            self.ingest()

    def ask(self, question: str) -> RAGAnswer:
        "Recupera el contexto relevante y genera una respuesta con sus fuentes."
        self.ensure_index()
        chunks = retrieve(
            self.vectorstore, question, self.settings.top_k, self.settings.score_threshold
        )
        if not chunks:
            return RAGAnswer(question=question, answer=NOT_FOUND_MESSAGE)

        answer = generate_answer(self.llm, question, chunks)
        return RAGAnswer(question=question, answer=answer, chunks=chunks)

    def list_documents(self) -> list[str]:
        "Nombres de los documentos disponibles para indexar."
        docs_dir = self.settings.docs_dir
        if not docs_dir.is_dir():
            return []
        return sorted(
            p.name
            for p in docs_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def save_document(self, filename: str, content: bytes) -> Path:
        "Guarda un documento nuevo en la carpeta de documentos."
        safe_name = Path(filename).name  # evita rutas como ../../archivo
        if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Formato no soportado. Formatos válidos: {supported}")

        self.settings.docs_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.docs_dir / safe_name
        path.write_bytes(content)
        return path


@lru_cache
def get_rag_service() -> RAGService:
    """Crea el servicio una sola vez (cargar los modelos es costoso)."""
    from app.providers import build_embeddings, build_llm

    settings = get_settings()
    return RAGService(settings, build_embeddings(settings), build_llm(settings))