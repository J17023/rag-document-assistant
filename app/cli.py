"""Interfaz por consola.

Uso:
    python -m app.cli ingest                 # indexa los documentos de data/docs
    python -m app.cli ask "¿Tu pregunta?"    # hace una pregunta
    python -m app.cli chat                   # modo interactivo
"""
import argparse
import logging

from app.rag import RAGAnswer, get_rag_service


def print_answer(result: RAGAnswer) -> None:
    print(f"\nRespuesta:\n{result.answer}\n")
    if result.chunks:
        print("Fuentes:")
        for i, chunk in enumerate(result.chunks, start=1):
            print(f"  [{i}] {chunk.label} (score={chunk.score:.2f})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Asistente RAG para documentos internos")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Indexa los documentos de la carpeta de documentos")
    ask_parser = subparsers.add_parser("ask", help="Hace una pregunta")
    ask_parser.add_argument("question", help="Pregunta entre comillas")
    subparsers.add_parser("chat", help="Modo interactivo (escribe 'salir' para terminar)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    service = get_rag_service()

    if args.command == "ingest":
        stats = service.ingest()
        print(f"Indexados {stats.documents} documentos en {stats.chunks} chunks.")
    elif args.command == "ask":
        print_answer(service.ask(args.question))
    elif args.command == "chat":
        print("Asistente de documentos. Escribe 'salir' para terminar.")
        while (question := input("\nPregunta: ").strip()).lower() not in {"salir", "exit"}:
            if question:
                print_answer(service.ask(question))


if __name__ == "__main__":
    main()