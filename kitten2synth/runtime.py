from __future__ import annotations

from pathlib import Path
from typing import Any

from onnxvoice import OnnxVoice
from onnxvoice.errors import AssetNotFoundError, NotInstalledError

from .adapter import register
from .catalog import install_bundle, load_catalog, bundle_record, catalog_item
from .errors import CatalogError, InvalidBundleError
from .types import SynthesisResult

register()


def _local_files(directory: Path) -> dict[str, Path]:
    import json

    manifest_path = directory / "bundle.json"
    if not manifest_path.is_file():
        raise InvalidBundleError(f"missing bundle.json: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidBundleError(f"invalid bundle.json: {exc}") from exc
    if manifest.get("format") != "kitten2-onnx-bundle-v1":
        raise InvalidBundleError(f"unsupported bundle format: {manifest.get('format')!r}")
    files: dict[str, Path] = {"bundle": manifest_path}
    for index, relative in enumerate(sorted(manifest.get("files", {}))):
        path = directory / relative
        if not path.is_file():
            raise InvalidBundleError(f"bundle is missing {relative}")
        files[f"asset:{index:03d}"] = path
    return files


class Kitten2Runtime:
    """High-level KittenTTS 2 frontend backed by an ONNXVoice adapter."""

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
            files=_local_files(directory),
            sample_rate=24000,
            providers=provider,
            metadata={"runtime": {"layout": "kitten2-onnx-bundle-v1"}},
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
        manager, installation = install_bundle(
            bundle,
            catalog=catalog,
            cache_dir=cache_dir,
            force=force,
        )
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
        data = load_catalog(catalog)
        record = bundle_record(bundle, data)
        manager = OnnxVoice(cache_dir=cache_dir, offline=True)
        item = catalog_item(record)
        try:
            installation = manager.store.get(item.system, item.id)
            manager.store.verify(installation)
        except (AssetNotFoundError, NotInstalledError) as exc:
            raise CatalogError(f"bundle is not installed: {item.ref}") from exc
        adapter = manager.open(installation, providers=provider)
        return cls(adapter, manager=manager)

    @property
    def model_id(self) -> str:
        return self.adapter.installation.id

    @property
    def voices(self) -> tuple[str, ...]:
        return tuple(self.adapter.installation.voices)

    @property
    def default_voice(self) -> str | None:
        return self.adapter.installation.default_voice

    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        preset: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 512,
    ) -> SynthesisResult:
        inference = self.adapter.infer_text(
            text,
            voice=voice,
            preset=preset,
            seed=seed,
            max_new_tokens=max_new_tokens,
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
