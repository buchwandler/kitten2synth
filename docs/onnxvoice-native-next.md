# Native ONNXVoice Kitten2 follow-up

The MVP deliberately bridges to `kitten2_onnx.tts.Kitten2OnnxTTS`. Once the two starter
repositories are working end-to-end, move that implementation into ONNXVoice.

Recommended order:

1. Add `onnxvoice/systems/kitten2.py` with `system = "kitten2"`.
2. Port the tested LM KV-cache generation, speaker projection, S3 token cleanup, flow decoder,
   mel2wav decoder, decode presets, voice loading and low-memory exact-ternary session options.
3. Add a Kitten2 catalog parser accepting nested multi-file artifacts and immutable HF revisions.
4. Add `onnxvoice catalog kitten2 build|verify` commands, following Qwen3/Pocket conventions.
5. Point ONNXVoice's default `kitten2` catalog source at `kitten2-onnx-bundles`.
6. Change kitten2synth `from_pretrained()` to `manager.install("kitten2:<id>")`.
7. Keep local `OnnxVoice.open_local(system="kitten2", files=...)` parity tests.
8. Add a real-bundle smoke test using `Bruno`, `stable`, seed 7, and a short token limit.

Do not mutate the existing ONNXVoice `kitten` v0.8 contract. Treat KittenTTS 2 as a separate
system until compatibility semantics are explicitly designed.
