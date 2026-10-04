"Ejecuta las preguntas de prueba y guarda los resultados."
import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.rag import get_rag_service


logging.basicConfig(level=logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)

EVAL_DIR = Path(__file__).parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"


def to_cell(text: str) -> str:
    "Adapta un texto para que quepa en una celda de tabla Markdown."
    return text.replace("|", "\\|").replace("\n", "<br>")


def main() -> None:
    settings = get_settings()
    service = get_rag_service()
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    stats = service.ingest()
    results = []
    for item in questions:
        start = time.perf_counter()
        result = service.ask(item["question"])
        elapsed = time.perf_counter() - start

        results.append(
            {
                **item,
                "answer": result.answer,
                "sources": [
                    {
                        "label": c.label,
                        "chunk_id": c.document.metadata.get("chunk_id"),
                        "score": round(c.score, 3),
                    }
                    for c in result.chunks
                ],
                "seconds": round(elapsed, 1),
            }
        )
        print(f"[{item['id']}] {item['question']}\n    -> {result.answer}\n")

    model_slug = re.sub(r"[^a-zA-Z0-9.]+", "-", f"{settings.llm_provider}_{settings.llm_model}")
    json_path = EVAL_DIR / f"results_{model_slug}.json"
    md_path = EVAL_DIR / f"results_{model_slug}.md"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Resultados de evaluación — {settings.llm_provider} / {settings.llm_model}",
        "",
        f"- Fecha: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Embeddings: `{settings.embedding_model}`",
        f"- Chunking: {settings.chunk_size} caracteres, solapamiento {settings.chunk_overlap}",
        f"- Top K: {settings.top_k} · Umbral: {settings.score_threshold}",
        f"- Documentos indexados: {stats.documents} · Chunks: {stats.chunks}",
        "",
        "| # | Tipo | Pregunta | Respuesta esperada | Respuesta generada | Fuentes | Tiempo | ¿Correcta? | Observación |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        sources = "<br>".join(f"{s['label']} ({s['score']})" for s in r["sources"]) or "—"
        lines.append(
            f"| {r['id']} | {r['type']} | {to_cell(r['question'])} | {to_cell(r['expected'])} "
            f"| {to_cell(r['answer'])} | {sources} | {r['seconds']} s | _pendiente_ | _pendiente_ |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Resultados guardados en:\n  {md_path}\n  {json_path}")


if __name__ == "__main__":
    main()