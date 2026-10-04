"Pruebas del armado del prompt y de la normalización de la respuesta."
from langchain_core.documents import Document

from app.generator import content_to_text
from app.prompts import build_messages, format_context
from app.retriever import RetrievedChunk


def make_chunks() -> list[RetrievedChunk]:
    "Un fragmento de PDF con  y uno de Markdown"
    return [
        RetrievedChunk(Document(page_content="Texto PDF", metadata={"source": "m.pdf", "page": 2}), 0.9),
        RetrievedChunk(Document(page_content="Texto MD", metadata={"source": "p.md"}), 0.8),
    ]


def test_context_numbers_fragments_with_their_source():
    "Los fragmentos se numeran con su fuente, para que el LLM pueda citarlos."
    context = format_context(make_chunks())

    assert "[1] (m.pdf, p.2)\nTexto PDF" in context
    assert "[2] (p.md)\nTexto MD" in context


def test_messages_separate_rules_from_data():
    "Las reglas van en el mensaje de sistema y los datos en el de usuario."
    messages = build_messages("¿Pregunta?", make_chunks())

    assert messages[0][0] == "system"
    assert messages[1][0] == "human"
    assert "Pregunta: ¿Pregunta?" in messages[1][1]


def test_content_to_text_handles_string_and_parts():
    "Extrae solo el texto de la respuesta e ignora las partes que no son texto."
    assert content_to_text("  hola  ") == "hola"

    parts = [
        {"type": "text", "text": "ho"},
        {"type": "thinking", "thinking": "no debe aparecer"},
        {"type": "text", "text": "la"},
    ]
    assert content_to_text(parts) == "hola"