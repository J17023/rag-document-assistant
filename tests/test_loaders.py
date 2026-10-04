from pathlib import Path

import pytest

from app.loaders import load_directory, load_file
from tests.conftest import DATA_DIR


"Un Markdown se carga con su nombre en 'source' y sin página."
def test_load_markdown_keeps_source(tmp_path: Path):
    path = tmp_path / "nota.md"
    path.write_text("# Título\nContenido", encoding="utf-8")

    docs = load_file(path)

    assert len(docs) == 1
    assert docs[0].metadata == {"source": "nota.md"}
    assert "Contenido" in docs[0].page_content

"El PDF se carga por páginas numeradas desde 1."
def test_load_pdf_numbers_pages_from_one():
    docs = load_file(DATA_DIR / "manual_trabajo_remoto.pdf")

    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "conectividad" in docs[0].page_content

"Un archivo vacío no genera documentos."
def test_empty_file_returns_no_documents(tmp_path: Path):
    path = tmp_path / "vacio.txt"
    path.write_text("   ", encoding="utf-8")

    assert load_file(path) == []

"Un formato no soportado lanza un error claro."
def test_unsupported_format_raises(tmp_path: Path):
    path = tmp_path / "datos.docx"
    path.write_bytes(b"x")

    with pytest.raises(ValueError, match="Formato no soportado"):
        load_file(path)

"Al cargar una carpeta se ignoran los formatos no soportados."
def test_load_directory_ignores_unsupported_files(tmp_path: Path):
    (tmp_path / "a.md").write_text("texto a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("texto b", encoding="utf-8")
    (tmp_path / "imagen.png").write_bytes(b"x")

    docs = load_directory(tmp_path)

    assert sorted(d.metadata["source"] for d in docs) == ["a.md", "b.txt"]

"Una carpeta inexistente lanza un error."
def test_load_directory_missing_folder_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "no_existe")