from kitten2synth import Kitten2Runtime

with Kitten2Runtime.from_pretrained("kitten-tts-v2") as runtime:
    result = runtime.generate(
        "[joyful] We won the grant <laugh> I can (((hardly))) believe it!",
        voice="Kiki",
        preset="expressive",
    )
    result.save_wav("kiki-expressive.wav")
