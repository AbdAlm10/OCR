from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.ocr import get_registry, image_from_bytes

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("katib")

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/tiff",
    "image/tif",
    "image/bmp",
}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    registry = get_registry()
    preload = os.getenv("OCR_PRELOAD", "false").lower() in {"1", "true", "yes"}
    if preload:
        try:
            registry.ensure_loaded(registry.default_model_id)
        except Exception:
            logger.exception(
                "Model preload failed — first request will retry. "
                "Install torch/transformers/peft and ensure Hugging Face access."
            )
    else:
        logger.info(
            "API ready (lazy load). Default model: %s · device: %s",
            registry.default_model_id,
            registry.device,
        )
    yield


app = FastAPI(
    title="Katib OCR",
    description="Arabic–English handwritten OCR API (multi-model)",
    version="1.1.0",
    lifespan=lifespan,
)

origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    registry = get_registry()
    return {
        "status": "ok",
        "device": registry.device,
        "default_model": registry.default_model_id,
        "models": registry.list_models(),
    }


@app.get("/api/models")
def list_models() -> dict:
    registry = get_registry()
    return {
        "default": registry.default_model_id,
        "device": registry.device,
        "models": registry.list_models(),
    }


@app.post("/api/ocr")
async def ocr(
    file: UploadFile = File(...),
    language: str = Form(default="ar"),
    model: str = Form(default=""),
) -> dict:
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        name = (file.filename or "").lower()
        if not any(
            name.endswith(ext)
            for ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")
        ):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Use PNG, JPEG, WEBP, or TIFF.",
            )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail=f"File too large (max {MAX_UPLOAD_MB} MB)."
        )

    try:
        image = image_from_bytes(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    registry = get_registry()
    model_id = (model or registry.default_model_id).strip()

    try:
        engine = registry.ensure_loaded(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Model load failed (%s)", model_id)
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

    try:
        text = engine.extract(image, language=language)
    except Exception as exc:
        logger.exception("OCR failed (%s)", model_id)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

    return {
        "text": text,
        "filename": file.filename,
        "characters": len(text),
        "model": engine.info.hf_id,
        "model_id": engine.info.id,
        "model_label": engine.info.label,
        "device": engine.device,
    }


@app.post("/api/ocr.txt", response_class=PlainTextResponse)
async def ocr_plain(
    file: UploadFile = File(...),
    language: str = Form(default="ar"),
    model: str = Form(default=""),
) -> str:
    result = await ocr(file=file, language=language, model=model)
    return result["text"]
