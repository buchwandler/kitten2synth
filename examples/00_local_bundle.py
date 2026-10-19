from kitten2synth import Kitten2Runtime

with Kitten2Runtime.load("/path/to/kitten-tts-v2-onnx") as runtime:
    result = runtime.generate(
        "Hello from KittenTTS 2.",
        voice="Bruno",
        preset="stable",
        seed=7,
    )
    result.save_wav("hello-local.wav")
