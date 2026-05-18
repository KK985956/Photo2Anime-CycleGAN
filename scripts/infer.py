from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anime_style.infer import AnimeStylizer
from anime_style.utils import IMAGE_EXTENSIONS, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stylize a photo or a folder of photos.")
    parser.add_argument("input", help="Image file or directory.")
    parser.add_argument("--output", default="outputs/inference", help="Output file or directory.")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint. Uses fallback if omitted.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    return parser


def iter_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    stylizer = AnimeStylizer(
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        device=args.device,
        use_fallback=True,
    )

    inputs = iter_inputs(input_path)
    if not inputs:
        raise FileNotFoundError(f"No images found: {input_path}")

    if input_path.is_file() and output_path.suffix:
        outputs = [output_path]
    else:
        ensure_dir(output_path)
        outputs = [output_path / f"{item.stem}_anime.png" for item in inputs]

    for source, target in zip(inputs, outputs):
        target.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(source).convert("RGB")
        result = stylizer.stylize(image)
        result.save(target)
        print(f"saved {target}")


if __name__ == "__main__":
    main()

