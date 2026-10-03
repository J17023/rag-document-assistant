"Creación de los modelos de embeddings y del LLM a partir de la configuración."
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    "Modelo de embeddings local con sentence-transformers"
    from langchain_huggingface import HuggingFaceEmbeddings

    encode_kwargs = {"normalize_embeddings": True}
    query_encode_kwargs = {"normalize_embeddings": True}
    if "e5" in settings.embedding_model.lower():
        encode_kwargs["prompt"] = "passage: "
        query_encode_kwargs["prompt"] = "query: "

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs=encode_kwargs,
        query_encode_kwargs=query_encode_kwargs,
    )


def build_llm(settings: Settings) -> BaseChatModel:
    "LLM configurable desde el .env (google_genai, ollama, openai, ...)."
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=settings.llm_temperature,
    )