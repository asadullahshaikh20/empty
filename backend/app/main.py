"""FastAPI application entry point.

Run (from backend/):  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import anyio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .db import SessionLocal
from .routers import assessments, auth, images, system
from .services import storage
from .services.model_service import ModelService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("iqa")

DISCLAIMER = "Academic prototype (FYP 23F-34). Not FDA-approved or CE-marked; not for clinical use."


def create_app(model_service: ModelService | None = None, load_model: bool | None = None) -> FastAPI:
    settings = get_settings()
    service = model_service or ModelService()
    should_load = settings.load_model_on_startup if load_model is None else load_model

    def _warm_up() -> None:
        with SessionLocal() as db:
            service.warm_up(db)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        storage.ensure_dirs()
        if should_load:
            await anyio.to_thread.run_sync(_warm_up)  # model loading blocks; keep the event loop free
        yield

    app = FastAPI(
        title="AI-Based Medical Image Quality and Reliability Assessment API (FYP 23F-34)",
        version="0.3.0",
        description=DISCLAIMER,
        lifespan=lifespan,
    )
    app.state.model_service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # never leak stack traces to clients
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    for r in (auth.router, images.router, assessments.router, system.router):
        app.include_router(r)
    return app


app = create_app()
