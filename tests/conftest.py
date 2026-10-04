"Fixtures compartidas por los tests: modelos falsos y configuración aislada."
import math
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from app.config import Settings

DATA_DIR = Path(__file__).parent.parent / "data" / "docs"

VOCABULARY = ["vacaciones", "días", "auxilio", "conectividad", "remoto", "salario", "equipos"]


class KeywordEmbeddings(Embeddings):
    """Embeddings falsos: cuentan palabras clave en vez de usar un modelo real."""

    def _embed(self, text: str) -> list[float]:
        text = text.lower()
        vector = [float(text.count(word)) for word in VOCABULARY] + [0.1]
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Configuración con documentos y vector store en una carpeta temporal."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "vacaciones.md").write_text(
        "# Vacaciones\nCada colaborador tiene 15 días de vacaciones al año.", encoding="utf-8"
    )
    (docs_dir / "remoto.txt").write_text(
        "El auxilio de conectividad para trabajo remoto es de 180 000 COP.", encoding="utf-8"
    )
    return Settings(
        docs_dir=docs_dir,
        vectorstore_dir=tmp_path / "vectorstore",
        collection_name="test",
        chunk_size=200,
        chunk_overlap=20,
        top_k=2,
        score_threshold=0.2,
    )