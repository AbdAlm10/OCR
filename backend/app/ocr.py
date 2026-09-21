from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

PROMPT_V3_AR = (
    "ارجو استخراج النص العربي كاملاً من هذه الصورة من البداية الى النهاية "
    "بدون اي اختصار ودون ذيادة او حذف. اقرأ كل المحتوى النصي الموجود في الصورة:"
)
PROMPT_V3_EN = (
    "Extract all text from this image completely, from beginning to end, "
    "without shortening, adding, or deleting anything. Read every textual content:"
)
PROMPT_WARRAQ = "اكتب النص المكتوب بخط اليد في هذه الصورة كما هو تماما."
PROMPT_FANAR_HTR = (
    "Transcribe the handwritten Arabic text in this image exactly as written, "
    "including any diacritics. Output only the transcription, nothing else."
)


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

    def extract(self, image: Image.Image, language: str = "ar") -> str: ...


class QwenHandwrittenV3Engine:
    info = ModelInfo(
        id="ocr-v3",
        label="Arabic-English OCR v3",
        description="Qwen2.5-VL-3B fine-tune — Arabic + English pages",
        hf_id="sherif1313/Arabic-English-handwritten-OCR-v3",
        languages=("ar", "en"),
        note="Good for full pages and bilingual handwriting.",
    )

    def __init__(self) -> None:
        self.device = resolve_device()
        self.max_new_tokens = int(os.getenv("OCR_MAX_NEW_TOKENS", "512"))
        self.model: Any = None
        self.processor: Any = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        if self._ready:
            return

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading %s on %s …", self.info.hf_id, self.device)
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        load_kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": dtype}
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
        elif self.device == "mps":
            load_kwargs["device_map"] = {"": "mps"}
        else:
            load_kwargs["device_map"] = {"": "cpu"}

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.info.hf_id,
            **load_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(self.info.hf_id, trust_remote_code=True)
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

    def _vision_images(self, messages: list[dict]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                    images.append(enhance_image(item["image"]))
        return images

    @torch.inference_mode()
    def extract(self, image: Image.Image, language: str = "ar") -> str:
        if not self._ready:
            self.load()
        assert self.model is not None and self.processor is not None

        prompt = PROMPT_V3_EN if language.lower().startswith("en") else PROMPT_V3_AR
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=self._vision_images(messages),
            padding=True,
            return_tensors="pt",
        )
        target = next(self.model.parameters()).device
        inputs = inputs.to(target)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=10,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        input_len = inputs.input_ids.shape[1]
        out = self.processor.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return out.strip()


class FanarPeftEngine:
    """Shared loader for Fanar-2-Oryx-IVU + QLoRA handwriting adapters."""

    def __init__(
        self,
        info: ModelInfo,
        prompt: str,
        max_new_tokens: int = 256,
        processor_kwargs: dict[str, Any] | None = None,
        use_chat_template_tokenize: bool = True,
    ) -> None:
        self.info = info
        self.prompt = prompt
        self.device = resolve_device()
        self.max_new_tokens = max_new_tokens
        self.base_id = os.getenv("WARRAQ_BASE", "QCRI/Fanar-2-Oryx-IVU")
        self.adapter_id = info.hf_id
        self.processor_kwargs = processor_kwargs or {}
        self.use_chat_template_tokenize = use_chat_template_tokenize
        self.model: Any = None
        self.processor: Any = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        if self._ready:
            return

        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(
            "Loading Fanar base %s + adapter %s on %s …",
            self.base_id,
            self.adapter_id,
            self.device,
        )

        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        load_kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
        elif self.device == "mps":
            load_kwargs["device_map"] = {"": "mps"}
        else:
            load_kwargs["device_map"] = {"": "cpu"}

        self.processor = AutoProcessor.from_pretrained(
            self.base_id,
            trust_remote_code=True,
            **self.processor_kwargs,
        )
        base = AutoModelForImageTextToText.from_pretrained(self.base_id, **load_kwargs)
        self.model = PeftModel.from_pretrained(base, self.adapter_id)
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
    def extract(self, image: Image.Image, language: str = "ar") -> str:
        del language
        if not self._ready:
            self.load()
        assert self.model is not None and self.processor is not None

        img = enhance_image(image, max_side=1024)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        if self.use_chat_template_tokenize:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            target = next(self.model.parameters()).device
            inputs = {k: v.to(target) if hasattr(v, "to") else v for k, v in inputs.items()}
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            input_len = inputs["input_ids"].shape[1]
            text = self.processor.decode(
                generated[0][input_len:],
                skip_special_tokens=True,
            )
            return text.strip()

        # Official Fanar-HTR path: template → processor(text, images)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        target = next(self.model.parameters()).device
        inputs = inputs.to(target)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        input_len = inputs.input_ids.shape[1]
        out = self.processor.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
        )[0]
        return out.strip()


def _make_warraq() -> FanarPeftEngine:
    return FanarPeftEngine(
        info=ModelInfo(
            id="warraq",
            label="Warraq Arabic HTR 7B",
            description="QLoRA on Fanar — strong on real phone photos & forms",
            hf_id="mabdulaziz499/Warraq-Arabic-HTR-7B",
            languages=("ar",),
            note="Best on single-line crops; full pages still work but may be slower/noisier.",
        ),
        prompt=PROMPT_WARRAQ,
        max_new_tokens=int(os.getenv("WARRAQ_MAX_NEW_TOKENS", "256")),
        use_chat_template_tokenize=True,
    )


def _make_fanar_htr() -> FanarPeftEngine:
    # Highest measured CER in the author's open KHATT comparison (~3.77%).
    return FanarPeftEngine(
        info=ModelInfo(
            id="fanar-htr",
            label="Fanar HTR 7B (best CER)",
            description="Top open KHATT score ~3.77% CER — generalist Arabic handwriting",
            hf_id="mabdulaziz499/arabic-htr-fanar-7b-lora",
            languages=("ar",),
            note="Best verified line-level accuracy. Prefer single-line crops.",
        ),
        prompt=PROMPT_FANAR_HTR,
        max_new_tokens=int(os.getenv("FANAR_HTR_MAX_NEW_TOKENS", "160")),
        processor_kwargs={"min_pixels": 200704, "max_pixels": 802816},
        use_chat_template_tokenize=False,
    )


class EngineRegistry:
    """Lazy multi-model registry — keeps at most one heavy model loaded."""

    def __init__(self) -> None:
        self.device = resolve_device()
        self._engines: dict[str, Any] = {
            "ocr-v3": QwenHandwrittenV3Engine(),
            "warraq": _make_warraq(),
            "fanar-htr": _make_fanar_htr(),
        }
        self._active: str | None = None
        default = os.getenv("OCR_DEFAULT_MODEL", "ocr-v3")
        self.default_model_id = default if default in self._engines else "ocr-v3"

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

        # Unload other models to free RAM/VRAM before loading
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


# Back-compat aliases used by older imports
DEFAULT_PROMPT_AR = PROMPT_V3_AR
DEFAULT_PROMPT_EN = PROMPT_V3_EN
