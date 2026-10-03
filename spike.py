"""Spike: RAG de punta a punta en un solo script para validar el flujo."""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = Path("data/docs")
DB_DIR = "vectorstore"
TOP_K = 5
SCORE_THRESHOLD = 0.2  # red de seguridad; con e5 los scores son altos y comprimidos

# 1. Cargar documentos
docs = []
for path in DOCS_DIR.iterdir():
    if path.suffix == ".pdf":
        docs += PyPDFLoader(str(path)).load()
    elif path.suffix in {".md", ".txt"}:
        docs += TextLoader(str(path), encoding="utf-8").load()

# 2. Chunking (500 para que cada sección quede más aislada)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(docs)
print(f"{len(docs)} páginas/documentos -> {len(chunks)} chunks")

# 3-4. Embeddings + vector store (se reconstruye en cada ejecución)
# e5 necesita los prefijos "query: " (preguntas) y "passage: " (documentos)
shutil.rmtree(DB_DIR, ignore_errors=True)
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
    encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
    query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
)
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=DB_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)

# 5. LLM configurable desde .env
llm = init_chat_model(
    model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
    model_provider=os.getenv("LLM_PROVIDER", "ollama"),
    temperature=0,
)

SYSTEM_PROMPT = """Eres un asistente que responde preguntas usando únicamente los fragmentos de documentos internos que se te entregan.

Reglas:
1. Usa solo la información de los fragmentos. No uses conocimiento externo ni inventes datos.
2. Después de cada dato, cita el fragmento entre corchetes, por ejemplo [1] o [2].
3. Si la pregunta tiene varias partes, responde cada parte por separado:
   - Si una parte está en los fragmentos, respóndela.
   - Si una parte no está, di: "No encontré información sobre <esa parte> en los documentos."
4. Solo si NINGUNA parte de la pregunta está en los fragmentos, responde: "No encuentro esa información en los documentos."
5. Responde en español, de forma breve y directa."""

NOT_FOUND = "No encuentro esa información en los documentos."


def source_label(doc) -> str:
    """Nombre del archivo + página legible (pypdf numera desde 0)."""
    name = Path(doc.metadata["source"]).name
    page = doc.metadata.get("page")
    return f"{name}, p.{page + 1}" if page is not None else name


def ask(question: str, k: int = TOP_K) -> None:
    # 6. Recuperar contexto relevante
    results = db.similarity_search_with_relevance_scores(question, k=k)
    relevant = [(d, s) for d, s in results if s >= SCORE_THRESHOLD]

    print(f"\n{'=' * 80}\nQ: {question}")
    for d, s in results:
        preview = d.page_content[:70].replace("\n", " ")
        print(f"   score={s:.2f} | {source_label(d)} | {preview}...")

    if not relevant:
        print(f"A: {NOT_FOUND} (ningún fragmento superó el umbral)")
        return

    # 7. Generar respuesta con el contexto recuperado
    context = "\n\n".join(
        f"[{i}] ({source_label(d)})\n{d.page_content}"
        for i, (d, _) in enumerate(relevant, 1)
    )
    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", f"Fragmentos:\n{context}\n\nPregunta: {question}"),
    ])
    print(f"A: {response.content}")


if __name__ == "__main__":
    ask("¿Cuántos días de vacaciones tengo al año?")
    ask("¿Cuánto es el auxilio de conectividad para un trabajador 100 % remoto?")
    ask("¿Cuántos días de vacaciones tengo y cuánto cubre el seguro de salud?")
    ask("¿Cuál es el salario de un analista junior?")
    ask("Si soy remoto y quiero hacer un diplomado, ¿qué auxilios recibo?")