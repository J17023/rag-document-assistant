"Carga de documentos PDF, Markdown y texto plano."
import logging
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def load_pdf(path: Path) -> list[Document]:
    "Crea un Document por cada página con texto. Las páginas se numeran desde 1."
    reader = PdfReader(path)
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                Document(page_content=text, metadata={"source": path.name, "page": page_number})
            )
    return documents


def load_text(path: Path) -> list[Document]:
    "Carga un archivo .md o .txt completo como un solo Document."
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": path.name})]


def load_file(path: Path) -> list[Document]:
    "Carga un archivo según su extensión."
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".md", ".txt"}:
        return load_text(path)
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Formato no soportado: '{suffix}'. Formatos válidos: {supported}")


def load_directory(directory: Path) -> list[Document]:
    "Carga todos los archivos soportados de una carpeta."
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de documentos: {directory}")

    documents = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            loaded = load_file(path)
            logger.info("Cargado %s (%d secciones)", path.name, len(loaded))
            documents.extend(loaded)
    return documents