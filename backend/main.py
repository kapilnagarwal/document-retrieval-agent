import logging
from contextlib import asynccontextmanager, nullcontext

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.api import chat, documents, search
from backend.config import Settings
from backend.rag.generator import GenerationError
from backend.rag.query_plan import ENGINE_VERSION
from backend.services import Services
from backend.state import ProcessLease


def create_app(settings=None, services=None):
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    # HTTP client INFO logs may contain request URLs. Keep SDK/transport logs quiet.
    for logger in ("httpx", "httpcore", "google", "chromadb"):
        logging.getLogger(logger).setLevel(logging.WARNING)

    @asynccontextmanager
    async def lifespan(app):
        lease = (
            ProcessLease(settings.upload_directory / ".owner.lock")
            if services is None
            else nullcontext()
        )
        with lease:
            app.state.services = services or Services(settings)
            app.state.services.jobs.start()
            try:
                yield
            finally:
                app.state.services.jobs.close()
                for thread in app.state.services.stream_threads:
                    thread.join()

    app = FastAPI(title="Document Retrieval AI Agent", lifespan=lifespan)
    for router in (documents.router, search.router, chat.router):
        app.include_router(router)

    @app.exception_handler(GenerationError)
    async def generation_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logging.getLogger(__name__).error("Request failed (%s)", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Operation failed. Check local model availability, disk space, and backend logs."
            },
        )

    @app.get("/health")
    def health(request: Request):
        services = request.app.state.services
        with services.ingestion.lock:
            return dict(
                status="degraded" if services.ingestion.recovery_issues else "ok",
                recovery_issues=services.ingestion.recovery_issues,
                retrieval_engine_version=ENGINE_VERSION,
                vector_database="ready",
                chunks=services.store.count(),
                documents=len(services.ingestion.manifest.all()),
            )

    @app.get("/config")
    def config():
        return settings.public()

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = Settings()
    uvicorn.run("backend.main:app", host=config.backend_host, port=config.backend_port, workers=1)
