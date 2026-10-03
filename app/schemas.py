"Modelos de entrada y salida de la API."
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["¿Cuántos días de vacaciones tengo al año?"],
    )


class Source(BaseModel):
    document: str
    page: int | None = None
    chunk_id: str
    score: float
    excerpt: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class UploadResponse(IngestResponse):
    filename: str


class DocumentsResponse(BaseModel):
    documents: list[str]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    embedding_model: str