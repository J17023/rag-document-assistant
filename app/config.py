"Configuración de la aplicación, leída desde variables de entorno y el archivo .env."
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Carga el .env en os.environ para que los proveedores de LLM
load_dotenv()


class Settings(BaseSettings):
    "Parámetros configurables del asistente"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: str = "google_genai"
    llm_model: str = "gemini-3.8-flash"
    llm_temperature: float = 0.0

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-base"

    # Rutas
    docs_dir: Path = Path("data/docs")
    vectorstore_dir: Path = Path("vectorstore")
    collection_name: str = "documents"

    # RAG
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 5
    score_threshold: float = 0.2


@lru_cache
def get_settings() -> Settings:
    "Devuelve una única instancia de la configuración."
    return Settings()