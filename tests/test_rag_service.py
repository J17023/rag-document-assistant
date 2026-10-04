"Pruebas de integración del servicio RAG (app/rag.py) con Chroma real y modelos falsos."
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.prompts import NOT_FOUND_MESSAGE
from app.rag import RAGService
from tests.conftest import KeywordEmbeddings

"Crea el servicio con embeddings falsos y un LLM que devuelve respuestas fijas."
def make_service(settings, responses=None) -> RAGService:
    llm = FakeListChatModel(responses=responses or ["Respuesta generada [1]."])
    return RAGService(settings, KeywordEmbeddings(), llm)

"Indexa los dos documentos de prueba."
def test_ingest_indexes_all_documents(settings):

    stats = make_service(settings).ingest()

    assert stats.documents == 2
    assert stats.chunks >= 2

"La pregunta sobre el auxilio recupera primero el documento del auxilio."
def test_ask_retrieves_the_relevant_document_first(settings):
    """La pregunta sobre el auxilio recupera primero el documento del auxilio."""
    result = make_service(settings).ask("¿Cuánto es el auxilio de conectividad?")

    assert result.answer == "Respuesta generada [1]."
    assert result.chunks[0].document.metadata["source"] == "remoto.txt"

def test_ask_indexes_automatically_when_empty(settings):
    "Si el índice está vacío, se construye con la primera pregunta."
    result = make_service(settings).ask("¿Cuántos días de vacaciones tengo?")

    assert result.chunks

"Sin fragmentos relevantes responde 'no encuentro' sin llamar al LLM."
def test_ask_returns_not_found_without_calling_llm(settings):
    
    settings.score_threshold = 1.01  # ningún fragmento puede superarlo
    result = make_service(settings).ask("¿Cuál es el salario?")

    assert result.answer == NOT_FOUND_MESSAGE
    assert result.chunks == []

"Reindexar no duplica fragmentos en Chroma."
def test_reingest_does_not_duplicate_chunks(settings):
    
    service = make_service(settings)

    first = service.ingest()
    second = service.ingest()

    assert first.chunks == second.chunks
    assert len(service.vectorstore.get(include=[])["ids"]) == second.chunks

"No permite guardar formatos no soportados."
def test_save_document_rejects_unsupported_format(settings):
    
    with pytest.raises(ValueError):
        make_service(settings).save_document("malware.exe", b"x")


"Un nombre como '../../archivo' se guarda dentro de la carpeta de documentos."
def test_save_document_ignores_path_traversal(settings):
    path = make_service(settings).save_document("../../fuera.md", b"contenido")
    assert path.parent == settings.docs_dir
    assert path.name == "fuera.md"