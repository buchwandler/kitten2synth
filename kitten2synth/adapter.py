from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from onnxvoice.errors import CapabilityError, RuntimeContractError
from onnxvoice.systems import register_adapter
from onnxvoice.systems.base import SystemAdapter
from onnxvoice.types import InferenceResult

from .audio import validate_audio
from .errors import EngineUnavailableError


def _provider_names(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    raw = [value] if isinstance(value, str) else list(value)
    result: list[str] = []
    for provider in raw:
        lowered = provider.lower()
        if lowered in {"cpu", "cpuexecutionprovider"}:
            normalized = "CPUExecutionProvider"
        elif lowered in {"cuda", "cudaexecutionprovider", "gpu"}:
            normalized = "CUDAExecutionProvider"
        else:
            normalized = provider
        if normalized not in result:
            result.append(normalized)
    if "CUDAExecutionProvider" in result and "CPUExecutionProvider" not in result:
        result.append("CPUExecutionProvider")
    return result


class Kitten2Adapter(SystemAdapter):
    """ONNXVoice adapter bridge for the existing kitten2-onnx engine."""

    system = "kitten2"

    def __init__(self, installation, **kwargs: Any) -> None:
        super().__init__(installation, **kwargs)
        self._engine_instance: Any | None = None

    def _create_engine(self) -> Any:
        try:
            from kitten2_onnx.tts import Kitten2OnnxTTS
        except ImportError as exc:
            raise EngineUnavailableError(
                "kitten2-onnx is required by the MVP bridge. Install your local "
                "kitten2-onnx checkout with `python -m pip install -e ../kitten2-onnx`."
            ) from exc
        if self.provider_options is not None:
            raise CapabilityError("the MVP Kitten2 bridge does not yet forward provider_options")
        if self.session_options is not None:
            raise CapabilityError("the MVP Kitten2 bridge does not yet forward session_options")
        try:
            return Kitten2OnnxTTS(
                self.installation.path,
                providers=_provider_names(self.providers),
            )
        except Exception as exc:
            raise RuntimeContractError(f"failed to open Kitten2 bundle: {exc}") from exc

    @property
    def engine(self) -> Any:
        if self._engine_instance is None:
            self._engine_instance = self._create_engine()
        return self._engine_instance

    def infer(self, token_ids: Sequence[int], **kwargs: Any) -> InferenceResult:
        raise CapabilityError(
            "Kitten2 is a text-level multi-stage runtime; use Kitten2Adapter.infer_text()"
        )

    def infer_text(
        self,
        text: str,
        *,
        voice: str | None = None,
        preset: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 512,
    ) -> InferenceResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        try:
            audio = self.engine.generate(
                text,
                voice=voice,
                preset=preset,
                seed=seed,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            raise RuntimeContractError(f"Kitten2 generation failed: {exc}") from exc
        waveform = validate_audio(np.asarray(audio, dtype=np.float32))
        sample_rate = int(
            getattr(self.engine, "sample_rate", self.installation.sample_rate or 24000)
        )
        return InferenceResult(
            audio=waveform,
            sample_rate=sample_rate,
            metadata={
                "system": self.system,
                "voice": voice or self.installation.default_voice,
                "preset": preset,
                "seed": seed,
                "max_new_tokens": int(max_new_tokens),
                "layout": "kitten2-onnx-bundle-v1",
            },
        )

    def close(self) -> None:
        engine = self._engine_instance
        self._engine_instance = None
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        super().close()


def register() -> None:
    register_adapter(Kitten2Adapter.system, Kitten2Adapter)
