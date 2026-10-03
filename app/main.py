"API REST del asistente."
import logging

from fastapi import FastAPI

from app.routers import documents, health, qa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

app = FastAPI(
    title="RAG Document Assistant",
    description="Asistente RAG para consultar documentos internos.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(qa.router)