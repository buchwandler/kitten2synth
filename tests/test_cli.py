import sys
from unittest.mock import MagicMock, Mock

import kitten2synth.__main__ as cli


def test_cli_forwards_literal_expression_to_native_runtime(monkeypatch, tmp_path):
    runtime = MagicMock()
    runtime.generate.return_value.duration_seconds = 0.5
    runtime.generate.return_value.sample_rate = 24000
    from_pretrained = Mock(return_value=runtime)
    monkeypatch.setattr(cli.Kitten2Runtime, "from_pretrained", from_pretrained)
    output = tmp_path / "kiki-expressive.wav"
    text = "[joyful] We won the grant <laugh> I can (((hardly))) believe it!"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kitten2synth",
            "--bundle",
            "kitten-tts-v2",
            "--voice",
            "Kiki",
            "--preset",
            "expressive",
            "--output",
            str(output),
            text,
        ],
    )

    cli.main()

    from_pretrained.assert_called_once_with(
        "kitten-tts-v2",
        catalog=None,
        cache_dir=None,
        provider="cpu",
    )
    runtime.generate.assert_called_once_with(
        text,
        voice="Kiki",
        preset="expressive",
        seed=None,
        max_new_tokens=512,
    )
    runtime.generate.return_value.save_wav.assert_called_once_with(output)
