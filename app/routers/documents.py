"Gestión de documentos: listar, subir e indexar."
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.rag import RAGService, get_rag_service
from app.schemas import DocumentsResponse, IngestResponse, UploadResponse

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentsResponse)
def list_documents(service: RAGService = Depends(get_rag_service)) -> DocumentsResponse:
    return DocumentsResponse(documents=service.list_documents())


@router.post("/ingest", response_model=IngestResponse)
def ingest(service: RAGService = Depends(get_rag_service)) -> IngestResponse:
    stats = service.ingest()
    return IngestResponse(documents=stats.documents, chunks=stats.chunks)


@router.post("/upload", response_model=UploadResponse)
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