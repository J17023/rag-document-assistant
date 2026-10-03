"API REST del asistente."
import logging

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.rag import RAGService, get_rag_service
from app.retriever import RetrievedChunk
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    Source,
    UploadResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="RAG Document Assistant",
    description="Asistente RAG para consultar documentos internos",
    version="0.1.0",
)


def to_source(chunk: RetrievedChunk) -> Source:
    "Convierte un fragmento recuperado al formato de fuente de la API."
    metadata = chunk.document.metadata
    return Source(
        document=metadata["source"],
        page=metadata.get("page"),
        chunk_id=metadata.get("chunk_id", ""),
        score=round(chunk.score, 3),
        excerpt=chunk.document.page_content[:200],
    )


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


@app.get("/documents", response_model=DocumentsResponse)
def list_documents(service: RAGService = Depends(get_rag_service)) -> DocumentsResponse:
    return DocumentsResponse(documents=service.list_documents())


@app.post("/ingest", response_model=IngestResponse)
def ingest(service: RAGService = Depends(get_rag_service)) -> IngestResponse:
    stats = service.ingest()
    return IngestResponse(documents=stats.documents, chunks=stats.chunks)


@app.post("/documents/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...), service: RAGService = Depends(get_rag_service)
) -> UploadResponse:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera el límite de 10 MB.")

    try:
        path = service.save_document(file.filename or "", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stats = service.ingest()
    return UploadResponse(filename=path.name, documents=stats.documents, chunks=stats.chunks)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, service: RAGService = Depends(get_rag_service)) -> AskResponse:
    try:
        result = service.ask(request.question)
    except Exception as exc:
        logger.exception("Error al responder la pregunta")
        raise HTTPException(
            status_code=502,
            detail="No fue posible generar la respuesta. Revisa la configuración del LLM.",
        ) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[to_source(chunk) for chunk in result.chunks],
    )