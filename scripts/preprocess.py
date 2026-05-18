from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anime_style.preprocess import preprocess_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resize and normalize image folders for CycleGAN.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--no-crop", action="store_true", help="Resize without square center crop.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    count = preprocess_images(
        args.input_dir,
        args.output_dir,
        size=args.size,
        crop_square=not args.no_crop,
        overwrite=args.overwrite,
    )
    print(f"processed {count} images")


if __name__ == "__main__":
    main()

