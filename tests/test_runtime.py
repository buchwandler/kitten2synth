import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import kitten2synth.runtime as runtime_module
from kitten2synth.runtime import Kitten2Runtime


def make_adapter():
    return SimpleNamespace(
        installation=SimpleNamespace(id="kitten-tts-v2"),
        voices=["Bruno", "Kiki"],
        default_voice="Bruno",
        infer_text=Mock(),
        close=Mock(),
    )


def test_import_does_not_mutate_native_adapter_registry():
    source = """
from onnxvoice.systems import get_adapter
from onnxvoice.systems.kitten2 import Kitten2Adapter
before = get_adapter("kitten2")
assert before is Kitten2Adapter
import kitten2synth
assert get_adapter("kitten2") is before
"""
    subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_load_delegates_only_bundle_manifest(tmp_path, monkeypatch):
    adapter = make_adapter()
    open_local = Mock(return_value=adapter)
    monkeypatch.setattr(runtime_module.OnnxVoice, "open_local", open_local)
    bundle_dir = tmp_path / "bundle"

    runtime = Kitten2Runtime.load(bundle_dir, provider="cuda")

    open_local.assert_called_once_with(
        system="kitten2",
        files={"bundle": (bundle_dir / "bundle.json").resolve()},
        providers="cuda",
    )
    assert runtime.adapter is adapter


def test_from_pretrained_uses_native_install_and_catalog_override(tmp_path, monkeypatch):
    adapter = make_adapter()
    installation = object()
    manager = Mock()
    manager.install.return_value = installation
    manager.open.return_value = adapter
    manager_factory = Mock(return_value=manager)
    monkeypatch.setattr(runtime_module, "OnnxVoice", manager_factory)
    catalog = tmp_path / "catalog.json"
    cache_dir = tmp_path / "cache"

    runtime = Kitten2Runtime.from_pretrained(
        "kitten-tts-v2",
        catalog=catalog,
        cache_dir=cache_dir,
        provider="cuda",
        force=True,
    )

    manager_factory.assert_called_once_with(
        cache_dir=cache_dir,
        catalog_sources={"kitten2": str(catalog)},
    )
    manager.install.assert_called_once_with("kitten2:kitten-tts-v2", force=True)
    manager.open.assert_called_once_with(installation, providers="cuda")
    assert runtime.adapter is adapter


def test_from_installed_uses_native_offline_resolution(tmp_path, monkeypatch):
    adapter = make_adapter()
    manager = Mock()
    manager.open.return_value = adapter
    manager_factory = Mock(return_value=manager)
    monkeypatch.setattr(runtime_module, "OnnxVoice", manager_factory)
    catalog = tmp_path / "catalog.json"
    cache_dir = tmp_path / "cache"

    runtime = Kitten2Runtime.from_installed(
        catalog=catalog,
        cache_dir=cache_dir,
        provider="cpu",
    )

    manager_factory.assert_called_once_with(
        cache_dir=cache_dir,
        catalog_sources={"kitten2": str(catalog)},
        offline=True,
    )
    manager.open.assert_called_once_with("kitten2:kitten-tts-v2", providers="cpu")
    assert runtime.adapter is adapter


def test_generate_forwards_expression_and_wraps_native_result(tmp_path):
    audio = np.array([0.25, -0.25], dtype=np.float32)
    inference = SimpleNamespace(
        audio=audio,
        sample_rate=24000,
        metadata={"voice": "Kiki", "preset": "expressive"},
    )
    adapter = make_adapter()
    adapter.infer_text.return_value = inference
    runtime = Kitten2Runtime(adapter)
    text = "[joyful] We won the grant <laugh> I can (((hardly))) believe it!"

    result = runtime.generate(
        text,
        voice="Kiki",
        preset="expressive",
        seed=7,
        max_new_tokens=64,
        use_emotion=True,
        use_reference_prompt=False,
    )

    adapter.infer_text.assert_called_once_with(
        text,
        voice="Kiki",
        preset="expressive",
        seed=7,
        max_new_tokens=64,
        use_emotion=True,
        use_reference_prompt=False,
    )
    assert result.audio is audio
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 24000
    assert result.metadata == {
        "model_id": "kitten-tts-v2",
        "voice": "Kiki",
        "preset": "expressive",
    }
    assert runtime.voices == ("Bruno", "Kiki")
    assert runtime.default_voice == "Bruno"

    output = result.save_wav(tmp_path / "expressive.wav")
    with wave.open(str(output), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getframerate() == 24000
        assert stream.getnframes() == 2
