"Prompts del asistente."
from app.retriever import RetrievedChunk

NOT_FOUND_MESSAGE = "No encuentro esa información en los documentos."

SYSTEM_PROMPT = """Eres un asistente que responde preguntas usando únicamente los fragmentos de documentos internos que se te entregan.

Reglas:
1. Usa solo la información de los fragmentos. No uses conocimiento externo ni inventes datos.
2. Después de cada dato, cita el fragmento entre corchetes, por ejemplo [1] o [2].
3. Divide la pregunta en sus partes y responde cada una por separado:
   - Si una parte está en los fragmentos, respóndela con su cita.
   - Si una parte no está, di: "No encontré información sobre <esa parte> en los documentos."
4. Responde en español, de forma breve y directa.

Ejemplo:
Pregunta: ¿Cuál es el horario de la mesa de ayuda y quién es el gerente de tecnología?
Respuesta: La mesa de ayuda atiende de lunes a viernes de 7:00 a.m. a 7:00 p.m. [2]. No encontré información sobre el gerente de tecnología en los documentos."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Numera los fragmentos como [1], [2], ... para que el LLM pueda citarlos."""
    return "\n\n".join(
        f"[{i}] ({chunk.label})\n{chunk.document.page_content}"
        for i, chunk in enumerate(chunks, start=1)
    )


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[tuple[str, str]]:
    "Mensajes para el LLM: reglas en el mensaje de sistema y datos en el de usuario."
    return [
        ("system", SYSTEM_PROMPT),
        ("human", f"Fragmentos:\n{format_context(chunks)}\n\nPregunta: {question}"),
    ]