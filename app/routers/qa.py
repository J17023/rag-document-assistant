"Preguntas y respuestas sobre los documentos."
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.rag import RAGService, get_rag_service
from app.retriever import RetrievedChunk
from app.schemas import AskRequest, AskResponse, Source

logger = logging.getLogger(__name__)

router = APIRouter(tags=["qa"])


def to_source(chunk: RetrievedChunk) -> Source:
    """Convierte un fragmento recuperado al formato de fuente de la API."""
    metadata = chunk.document.metadata
    return Source(
        document=metadata["source"],
        page=metadata.get("page"),
        chunk_id=metadata.get("chunk_id", ""),
        score=round(chunk.score, 3),
        excerpt=chunk.document.page_content[:200],
    )


@router.post("/ask", response_model=AskResponse)
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