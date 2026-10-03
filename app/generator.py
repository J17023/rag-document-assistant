"Generación de la respuesta con el LLM."
from langchain_core.language_models import BaseChatModel

from app.prompts import build_messages
from app.retriever import RetrievedChunk


def content_to_text(content: str | list) -> str:
    "Normaliza el contenido de una respuesta del LLM a un string"
    if isinstance(content, str):
        return content.strip()
    return "".join(
        part if isinstance(part, str) else part.get("text", "")
        for part in content
        if isinstance(part, str) or part.get("type") == "text"
    ).strip()


def generate_answer(llm: BaseChatModel, question: str, chunks: list[RetrievedChunk]) -> str:
    "Pide al LLM una respuesta basada solo en los fragmentos recuperados"
    response = llm.invoke(build_messages(question, chunks))
    return content_to_text(response.content)