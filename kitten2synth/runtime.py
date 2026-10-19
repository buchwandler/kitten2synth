from __future__ import annotations

from pathlib import Path
from typing import Any

from onnxvoice import OnnxVoice

from .types import SynthesisResult


class Kitten2Runtime:
    """High-level KittenTTS 2 frontend backed by ONNXVoice's native adapter."""

    def __init__(self, adapter: Any, *, manager: OnnxVoice | None = None) -> None:
        self.adapter = adapter
        self._manager = manager

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        provider: str = "cpu",
    ) -> Kitten2Runtime:
        directory = Path(directory).expanduser().resolve()
        adapter = OnnxVoice.open_local(
            system="kitten2",
            files={"bundle": directory / "bundle.json"},
            providers=provider,
        )
        return cls(adapter)

    @classmethod
    def from_pretrained(
        cls,
        bundle: str = "kitten-tts-v2",
        *,
        catalog: str | Path | None = None,
        cache_dir: str | Path | None = None,
        provider: str = "cpu",
        force: bool = False,
    ) -> Kitten2Runtime:
        sources = None
        if catalog is not None:
            sources = {"kitten2": str(Path(catalog).expanduser())}

        manager = OnnxVoice(cache_dir=cache_dir, catalog_sources=sources)
        installation = manager.install(f"kitten2:{bundle}", force=force)
        adapter = manager.open(installation, providers=provider)
        return cls(adapter, manager=manager)

    @classmethod
    def from_installed(
        cls,
        bundle: str = "kitten-tts-v2",
        *,
        catalog: str | Path | None = None,
        cache_dir: str | Path | None = None,
        provider: str = "cpu",
    ) -> Kitten2Runtime:
        sources = None
        if catalog is not None:
            sources = {"kitten2": str(Path(catalog).expanduser())}

        manager = OnnxVoice(cache_dir=cache_dir, catalog_sources=sources, offline=True)
        adapter = manager.open(f"kitten2:{bundle}", providers=provider)
        return cls(adapter, manager=manager)

    @property
    def model_id(self) -> str:
        return self.adapter.installation.id

    @property
    def voices(self) -> tuple[str, ...]:
        return tuple(self.adapter.voices)

    @property
    def default_voice(self) -> str:
        return self.adapter.default_voice

    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        preset: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 512,
        use_emotion: bool | None = None,
        use_reference_prompt: bool | None = None,
    ) -> SynthesisResult:
        inference = self.adapter.infer_text(
            text,
            voice=voice,
            preset=preset,
            seed=seed,
            max_new_tokens=max_new_tokens,
            use_emotion=use_emotion,
            use_reference_prompt=use_reference_prompt,
        )
        return SynthesisResult(
            audio=inference.audio,
            sample_rate=inference.sample_rate,
            metadata={"model_id": self.model_id, **dict(inference.metadata)},
        )

    def close(self) -> None:
        self.adapter.close()

    def __enter__(self) -> Kitten2Runtime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
