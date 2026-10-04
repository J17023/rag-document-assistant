"Pruebas de los endpoints de la API con un servicio falso."
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app
from app.rag import IngestionStats, RAGAnswer, get_rag_service
from app.retriever import RetrievedChunk


class FakeRAGService:
    "Servicio falso con respuestas fijas, para probar solo la API."

    def __init__(self, fail: bool = False):
        self.fail = fail

    def ask(self, question: str) -> RAGAnswer:
        if self.fail:
            raise RuntimeError("LLM caído")
        chunk = RetrievedChunk(
            Document(
                page_content="Todo colaborador tiene 15 días hábiles.",
                metadata={"source": "politica.md", "chunk_id": "politica.md#1"},
            ),
            0.87,
        )
        return RAGAnswer(question=question, answer="15 días hábiles [1].", chunks=[chunk])

    def ingest(self) -> IngestionStats:
        return IngestionStats(documents=2, chunks=21)

    def list_documents(self) -> list[str]:
        return ["manual.pdf", "politica.md"]

    def save_document(self, filename: str, content: bytes) -> Path:
        if not filename.endswith((".pdf", ".md", ".txt")):
            raise ValueError("Formato no soportado.")
        return Path("data/docs") / filename


@pytest.fixture
def client():
    "Cliente de prueba de la API usando el servicio falso."
    app.dependency_overrides[get_rag_service] = lambda: FakeRAGService()
    yield TestClient(app)
    app.dependency_overrides.clear()

"El endpoint de estado responde ok."
def test_health(client):
    
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

"La respuesta incluye el texto generado y sus fuentes."
def test_ask_returns_answer_and_sources(client):
    response = client.post("/ask", json={"question": "¿Cuántos días de vacaciones tengo?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "15 días hábiles [1]."
    assert body["sources"][0]["document"] == "politica.md"
    assert body["sources"][0]["page"] is None

"Una pregunta demasiado corta se rechaza con 422."
def test_ask_rejects_too_short_question(client):
    
    response = client.post("/ask", json={"question": "a"})

    assert response.status_code == 422

"Si el LLM falla, la API responde 502 en lugar de caerse."
def test_ask_returns_502_when_llm_fails():

    app.dependency_overrides[get_rag_service] = lambda: FakeRAGService(fail=True)
    response = TestClient(app).post("/ask", json={"question": "¿Pregunta válida?"})
    app.dependency_overrides.clear()

    assert response.status_code == 502

"Lista los documentos disponibles."
def test_list_documents(client):
    
    response = client.get("/documents")

    assert response.json() == {"documents": ["manual.pdf", "politica.md"]}

"Reindexa y devuelve el conteo de documentos y fragmentos."
def test_ingest(client):
    
    response = client.post("/documents/ingest")

    assert response.json() == {"documents": 2, "chunks": 21}

"Acepta un documento con formato soportado."
def test_upload_valid_document(client):
    response = client.post(
        "/documents/upload", files={"file": ("nuevo.md", b"# Nuevo", "text/markdown")}
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "nuevo.md"

"Rechaza un formato no soportado con 400."
def test_upload_rejects_unsupported_format(client):

    response = client.post(
        "/documents/upload", files={"file": ("virus.exe", b"x", "application/octet-stream")}
    )

    assert response.status_code == 400

"Rechaza un archivo vacío con 400."
def test_upload_rejects_empty_file(client):
    
    response = client.post("/documents/upload", files={"file": ("vacio.md", b"", "text/markdown")})

    assert response.status_code == 400