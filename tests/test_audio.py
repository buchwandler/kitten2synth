import wave

import numpy as np

from kitten2synth.audio import write_wav


def test_write_wav(tmp_path):
    path = write_wav(tmp_path / "a.wav", np.array([0.0, 0.5, -0.5], dtype=np.float32), 24000)
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getframerate() == 24000
        assert stream.getnframes() == 3
