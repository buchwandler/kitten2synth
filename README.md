# kitten2synth

A small Python and CLI facade for KittenTTS 2. ONNXVoice owns the Kitten2 runtime, catalog, bundle validation, installation, and synthesis. `kitten2synth` provides convenient runtime construction, result wrapping, and WAV output.

## Architecture

```text
kitten2synth
    | high-level Python and CLI API
    v
ONNXVoice
    | native Kitten2 catalog and Kitten2Adapter
    v
KittenTTS 2 ONNX bundle
```

Importing this package does not modify ONNXVoice's adapter registry. The package does not parse catalogs, validate bundle layouts, or preprocess expression markup.

## Install

The normal dependency includes ONNXVoice's Kitten2 feature and tokenizer support. Install an execution provider with one of these extras:

```bash
python -m pip install "kitten2synth[cpu]"
# or
python -m pip install "kitten2synth[gpu]"
```

## Local bundle

```python
from kitten2synth import Kitten2Runtime

with Kitten2Runtime.load("/path/to/kitten-tts-v2-onnx") as runtime:
    result = runtime.generate(
        "Hello from KittenTTS 2.",
        voice="Bruno",
        preset="stable",
        seed=7,
    )
    result.save_wav("hello-local.wav")
```

The local bundle must contain `bundle.json`. ONNXVoice validates and opens the bundle.

## Managed bundle

```python
from kitten2synth import Kitten2Runtime

with Kitten2Runtime.from_pretrained("kitten-tts-v2") as runtime:
    result = runtime.generate(
        "Hello from KittenTTS 2.",
        voice="Bruno",
        preset="stable",
    )
    result.save_wav("hello-managed.wav")
```

ONNXVoice installs and caches the bundle using its configured Kitten2 catalog. To use a catalog file or a custom cache, pass `catalog=` or `cache_dir=`. `Kitten2Runtime.from_installed()` opens an already installed bundle without downloading it.

## Expression markup

Pass expression markup directly to ONNXVoice. `kitten2synth` does not inspect or rewrite it.

```python
from kitten2synth import Kitten2Runtime

with Kitten2Runtime.from_pretrained("kitten-tts-v2") as runtime:
    result = runtime.generate(
        "[joyful] We won the grant <laugh> I can (((hardly))) believe it!",
        voice="Kiki",
        preset="expressive",
    )
    result.save_wav("kiki-expressive.wav")
```

The request combines the `[joyful]` emotion, `<laugh>` event, `(((hardly)))` emphasis, `Kiki` voice, and `expressive` preset.

The equivalent CLI request accepts the markup literally:

```bash
kitten2synth \
  --bundle kitten-tts-v2 \
  --voice Kiki \
  --preset expressive \
  --output kiki-expressive.wav \
  '[joyful] We won the grant <laugh> I can (((hardly))) believe it!'
```

For a local bundle, use `--bundle-dir` instead of `--bundle`. The CLI also supports `--catalog`, `--cache-dir`, `--provider`, `--seed`, and `--max-new-tokens`.

See `examples/` for complete local, managed, and expressive scripts. The earlier migration brief is retained as a historical note in `docs/onnxvoice-native-next.md`.
