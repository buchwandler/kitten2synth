from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from onnxvoice import OnnxVoice
from onnxvoice.types import Artifact, CatalogItem, Installation

from .errors import CatalogError

DEFAULT_CATALOG = (
    "https://raw.githubusercontent.com/buchwandler/kitten2-onnx-bundles/main/catalog/bundles.json"
)


def _read_json(source: str | Path) -> dict[str, Any]:
    value = str(source)
    try:
        if value.startswith(("http://", "https://")):
            request = urllib.request.Request(value, headers={"User-Agent": "kitten2synth/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
        else:
            raw = Path(value).expanduser().read_bytes()
        data = json.loads(raw)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise CatalogError(f"unable to read catalog {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError("catalog must contain a JSON object")
    return data


def load_catalog(source: str | Path | None = None) -> dict[str, Any]:
    source = source or os.environ.get("KITTEN2SYNTH_CATALOG") or DEFAULT_CATALOG
    data = _read_json(source)
    if data.get("schema") != 1 or data.get("kind") != "kitten2-onnx-bundle-catalog":
        raise CatalogError("unsupported Kitten2 catalog schema")
    bundles = data.get("bundles")
    if not isinstance(bundles, dict) or not bundles:
        raise CatalogError("catalog bundles must be a non-empty object")
    return data


def bundle_record(bundle_id: str, catalog: dict[str, Any]) -> dict[str, Any]:
    bundles = catalog["bundles"]
    match = bundle_id
    if match not in bundles:
        for key, value in bundles.items():
            if bundle_id in value.get("aliases", []):
                match = key
                break
    if match not in bundles:
        raise CatalogError(f"unknown Kitten2 bundle: {bundle_id}")
    record = bundles[match]
    if record.get("layout") != "kitten2-onnx-bundle-v1":
        raise CatalogError(f"unsupported bundle layout: {record.get('layout')!r}")
    return record


def _hf_url(repository: str, revision: str, path: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    return (
        f"https://huggingface.co/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{encoded}"
    )


def catalog_item(record: dict[str, Any]) -> CatalogItem:
    source = record["source"]
    repository = source["repository"]
    revision = source["revision"]
    artifacts = tuple(
        Artifact(
            role=artifact["role"],
            filename=artifact["path"],
            url=_hf_url(repository, revision, artifact["path"]),
            sha256=artifact["sha256"],
            format=Path(artifact["path"]).suffix.lstrip(".") or None,
            metadata={
                "source": {
                    "provider": "huggingface",
                    "repository": repository,
                    "revision": revision,
                    "path": artifact["path"],
                }
            },
        )
        for artifact in record["artifacts"]
    )
    return CatalogItem(
        system="kitten2",
        id=record["id"],
        kind="model",
        artifacts=artifacts,
        aliases=tuple(record.get("aliases", ())),
        sample_rate=int(record["sample_rate"]),
        voices=tuple(record.get("voices", ())),
        default_voice=record.get("default_voice"),
        metadata={
            "runtime": dict(record.get("runtime", {})),
            "layout": record["layout"],
            "source_repository": repository,
            "source_revision": revision,
            "license": source.get("license"),
            **dict(record.get("metadata", {})),
        },
    )


def install_bundle(
    bundle_id: str,
    *,
    catalog: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[OnnxVoice, Installation]:
    data = load_catalog(catalog)
    item = catalog_item(bundle_record(bundle_id, data))
    manager = OnnxVoice(cache_dir=cache_dir)
    installation = manager.store.install(item, force=force)
    return manager, installation
