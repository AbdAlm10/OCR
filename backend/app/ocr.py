from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

HF_ID = "microsoft/trocr-base-handwritten"


def resolve_device() -> str:
    preference = os.getenv("OCR_DEVICE", "auto").lower()
    if preference == "cpu":
        return "cpu"
    if preference == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preference == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return "cpu"


def enhance_image(image: Image.Image, max_side: int = 1200) -> Image.Image:
    img = image.convert("RGB")
    img = ImageOps.exif_transpose(img)

    side = max(img.size)
    if side < 800:
        scale = 800 / side
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    elif side > max_side:
        scale = max_side / side
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=3))
    return img


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    description: str
    hf_id: str
    languages: tuple[str, ...]
    note: str = ""


class BaseEngine(Protocol):
    info: ModelInfo
    device: str

    @property
    def ready(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def extract(self, image: Image.Image, language: str = "en") -> str: ...


class TrocrHandwrittenEngine:
    """Microsoft TrOCR base, fine-tuned on IAM handwritten English lines."""

    info = ModelInfo(
        id="trocr-handwritten",
        label="TrOCR Handwritten",
        description="Microsoft TrOCR base — handwritten English (IAM)",
        hf_id=HF_ID,
        languages=("en",),
        note="Best on single-line handwriting crops.",
    )

    def __init__(self) -> None:
        self.device = resolve_device()
        self.max_new_tokens = int(os.getenv("OCR_MAX_NEW_TOKENS", "64"))
        self.model: Any = None
        self.processor: Any = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        if self._ready:
            return

        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        logger.info("Loading %s on %s …", self.info.hf_id, self.device)
        self.processor = TrOCRProcessor.from_pretrained(self.info.hf_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.info.hf_id)
        self.model.to(self.device)
        self.model.eval()
        self._ready = True
        logger.info("%s ready.", self.info.id)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self._ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded %s", self.info.id)

    @torch.inference_mode()
    def extract(self, image: Image.Image, language: str = "en") -> str:
        del language
        if not self._ready:
            self.load()
        assert self.model is not None and self.processor is not None

        img = enhance_image(image, max_side=1024)
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        generated_ids = self.model.generate(
            pixel_values,
            max_new_tokens=self.max_new_tokens,
        )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()


class EngineRegistry:
    """Lazy model registry — keeps at most one model loaded."""

    def __init__(self) -> None:
        self.device = resolve_device()
        self._engines: dict[str, Any] = {
            "trocr-handwritten": TrocrHandwrittenEngine(),
        }
        self._active: str | None = None
        default = os.getenv("OCR_DEFAULT_MODEL", "trocr-handwritten")
        self.default_model_id = (
            default if default in self._engines else "trocr-handwritten"
        )

    def list_models(self) -> list[dict[str, Any]]:
        items = []
        for mid, engine in self._engines.items():
            info = engine.info
            items.append(
                {
                    "id": info.id,
                    "label": info.label,
                    "description": info.description,
                    "hf_id": info.hf_id,
                    "languages": list(info.languages),
                    "note": info.note,
                    "ready": engine.ready,
                    "active": mid == self._active,
                }
            )
        return items

    def get(self, model_id: str | None = None) -> Any:
        mid = model_id or self.default_model_id
        if mid not in self._engines:
            raise KeyError(f"Unknown model '{mid}'. Available: {', '.join(self._engines)}")
        return self._engines[mid]

    def ensure_loaded(self, model_id: str | None = None) -> Any:
        mid = model_id or self.default_model_id
        engine = self.get(mid)

        if self._active and self._active != mid:
            other = self._engines[self._active]
            if other.ready:
                other.unload()

        if not engine.ready:
            engine.load()
        self._active = mid
        return engine


_registry: EngineRegistry | None = None


def get_registry() -> EngineRegistry:
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry


def image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))
