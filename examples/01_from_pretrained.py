from kitten2synth import Kitten2Runtime

with Kitten2Runtime.from_pretrained("kitten-tts-v2") as runtime:
    result = runtime.generate(
        "Hello from KittenTTS 2.",
        voice="Bruno",
        preset="stable",
    )
    result.save_wav("hello-managed.wav")
