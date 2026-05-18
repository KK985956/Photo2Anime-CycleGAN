from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from .utils import IMAGE_EXTENSIONS, ensure_dir


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def preprocess_images(
    input_dir: str | Path,
    output_dir: str | Path,
    size: int = 256,
    crop_square: bool = True,
    overwrite: bool = False,
) -> int:
    input_root = Path(input_dir)
    output_root = ensure_dir(output_dir)
    count = 0

    for path in sorted(input_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(input_root)
        output_path = output_root / relative.with_suffix(".jpg")
        if output_path.exists() and not overwrite:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        if crop_square:
            image = center_crop_square(image)
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        image.save(output_path, quality=95, optimize=True)
        count += 1
    return count

