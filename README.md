# kitten2synth MVP

A starter package for consuming the new KittenTTS 2 ONNX bundle through **ONNXVoice**.

## MVP architecture

```text
kitten2synth
    |
    | custom Kitten2 catalog -> ONNXVoice CatalogItem/Artifact
    v
ONNXVoice AssetStore
    |
    | verified managed installation
    v
ONNXVoice registered `kitten2` adapter
    |
    | MVP bridge
    v
kitten2_onnx.tts.Kitten2OnnxTTS
    |
    v
lm + speaker + flow + mel2wav -> 24 kHz audio
```

This intentionally mirrors the separation used by `qwen3synth`: the application package owns
text-to-speech ergonomics while ONNXVoice owns model installations and the runtime adapter
boundary.

The current ONNXVoice release already supports runtime adapter registration, but its built-in
`kitten` adapter is the old Kitten v0.8 single-graph contract. Therefore this MVP registers a
new `kitten2` adapter at import time and delegates the actual v2 generation pipeline to your
existing, validated `kitten2-onnx` implementation.

## Install for development

Assuming sibling checkouts:

```text
src/
├── kitten2-onnx/
├── kitten2-onnx-bundles/
└── kitten2synth/
```

Install the existing engine and this package:

```bash
cd ../kitten2-onnx
python -m pip install -e .

cd ../kitten2synth
python -m pip install -e ".[cpu]"
```

## 1. Test a local bundle first

```bash
kitten2synth \
  --bundle-dir /path/to/kitten-tts-v2-onnx \
  --voice Bruno \
  --preset stable \
  --seed 7 \
  --max-new-tokens 64 \
  --output hello.wav \
  "Hello from KittenTTS 2."
```

Python:

```python
from kitten2synth import Kitten2Runtime

with Kitten2Runtime.load("/path/to/kitten-tts-v2-onnx") as runtime:
    result = runtime.generate(
        "Hello from KittenTTS 2.",
        voice="Bruno",
        preset="stable",
        seed=7,
    )
    result.save_wav("hello.wav")
```

## 2. Test managed install from the companion catalog

Before `kitten2-onnx-bundles` is published on GitHub, point directly at its local JSON:

```bash
kitten2synth \
  --catalog ../kitten2-onnx-bundles/catalog/bundles.json \
  --bundle kitten-tts-v2 \
  --voice Bruno \
  --preset stable \
  --output hello-managed.wav \
  "Hello from the managed model."
```

That path downloads every catalog artifact into the ONNXVoice content-addressed cache,
checks the bundle SHA-256 values, reconstructs the nested bundle layout, then opens it through
the registered `kitten2` adapter.

## Important MVP limitation

`kitten2synth` still needs the local `kitten2-onnx` Python package because the full v2
generation/decoder implementation has not yet been moved into ONNXVoice. This is deliberate:
it avoids rewriting a pipeline you have already validated while you establish the new package
and catalog boundaries.

The next milestone is to upstream the bridge implementation into ONNXVoice as a native
`onnxvoice.systems.kitten2.Kitten2Adapter`. At that point:

1. remove the `kitten2-onnx` runtime dependency from this package;
2. teach ONNXVoice's catalog client/CLI the `kitten2-onnx-bundle-catalog` schema;
3. replace `manager.store.install(item)` with normal `manager.install("kitten2:kitten-tts-v2")`;
4. keep the public `Kitten2Runtime` API unchanged.

See `docs/onnxvoice-native-next.md`.
