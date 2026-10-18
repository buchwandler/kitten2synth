from __future__ import annotations

import argparse
from pathlib import Path

from .runtime import Kitten2Runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize speech with KittenTTS 2 ONNX")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--bundle-dir", type=Path, help="complete local Kitten2 ONNX bundle")
    source.add_argument("--bundle", default="kitten-tts-v2", help="catalog bundle id")
    parser.add_argument(
        "--catalog", type=Path, help="local kitten2-onnx-bundles catalog/bundles.json"
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--voice")
    parser.add_argument("--preset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("text")
    args = parser.parse_args()

    if args.bundle_dir:
        runtime = Kitten2Runtime.load(args.bundle_dir, provider=args.provider)
    else:
        runtime = Kitten2Runtime.from_pretrained(
            args.bundle,
            catalog=args.catalog,
            cache_dir=args.cache_dir,
            provider=args.provider,
        )
    with runtime:
        result = runtime.generate(
            args.text,
            voice=args.voice,
            preset=args.preset,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
        result.save_wav(args.output)
    print(f"wrote {args.output} ({result.duration_seconds:.2f}s, {result.sample_rate} Hz)")


if __name__ == "__main__":
    main()
