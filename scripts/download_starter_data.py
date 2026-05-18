from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]

OPEN_IMAGES_URLS = {
    "validation_boxes": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "validation_metadata": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}
HUMAN_FACE_LABEL = "/m/0dzct"
ANIME_DATASETS = {
    "small": {
        "repo_id": "huggan/few-shot-anime-face",
        "files": ["data/train-00000-of-00001.parquet"],
    },
    "large": {
        "repo_id": "minoruskore/anime-faces-256",
        "files": [
            "data/train-00000-of-00004.parquet",
            "data/train-00001-of-00004.parquet",
            "data/train-00002-of-00004.parquet",
            "data/train-00003-of-00004.parquet",
        ],
    },
}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def download_file(url: str, target: Path, timeout: int = 60) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with target.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return target


def sanitize_url_name(url: str, fallback: str) -> str:
    path = Path(urlparse(url).path)
    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    return f"{fallback}{suffix}"


def load_open_images_metadata(path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            metadata[row["ImageID"]] = row
    return metadata


def crop_box(image: Image.Image, xmin: float, xmax: float, ymin: float, ymax: float, padding: float = 0.35) -> Image.Image:
    width, height = image.size
    left = xmin * width
    right = xmax * width
    top = ymin * height
    bottom = ymax * height
    box_w = right - left
    box_h = bottom - top
    side = max(box_w, box_h) * (1 + padding)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left = max(0, int(center_x - side / 2))
    crop_top = max(0, int(center_y - side / 2))
    crop_right = min(width, int(center_x + side / 2))
    crop_bottom = min(height, int(center_y + side / 2))
    return image.crop((crop_left, crop_top, crop_right, crop_bottom))


def fetch_open_images_face(candidate: dict, image_size: int) -> tuple[Image.Image, dict] | None:
    try:
        response = requests.get(candidate["url"], timeout=20)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image = ImageOps.exif_transpose(image).convert("RGB")
        face = crop_box(
            image,
            float(candidate["XMin"]),
            float(candidate["XMax"]),
            float(candidate["YMin"]),
            float(candidate["YMax"]),
        )
        if min(face.size) < 64:
            return None
        face = face.resize((image_size, image_size), Image.Resampling.LANCZOS)
        return face, candidate["manifest"]
    except Exception:
        return None


def handle_completed_face_futures(
    pending: set,
    target_dir: Path,
    limit: int,
    saved: int,
    manifest: list,
    block: bool = True,
) -> tuple[set, int]:
    if not pending:
        return pending, saved
    return_when = concurrent.futures.FIRST_COMPLETED if block else concurrent.futures.ALL_COMPLETED
    done, pending = concurrent.futures.wait(pending, return_when=return_when)
    for future in done:
        result = future.result()
        if result is None or saved >= limit:
            continue
        image, item = result
        output = target_dir / f"openimages_face_{saved:05d}.jpg"
        image.save(output, quality=94, optimize=True)
        item["file"] = str(output.relative_to(ROOT))
        manifest.append(item)
        saved += 1
    return pending, saved


def download_open_images_faces(
    target_dir: Path,
    raw_dir: Path,
    limit: int,
    image_size: int,
    num_workers: int,
) -> int:
    ensure_dir(target_dir)
    metadata_path = download_file(OPEN_IMAGES_URLS["validation_metadata"], raw_dir / "validation-images-with-rotation.csv")
    boxes_path = download_file(OPEN_IMAGES_URLS["validation_boxes"], raw_dir / "oidv6-validation-annotations-bbox.csv")
    metadata = load_open_images_metadata(metadata_path)

    saved = 0
    manifest = []
    pending: set = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        with boxes_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if saved >= limit:
                    break
                if row.get("LabelName") != HUMAN_FACE_LABEL:
                    continue
                if row.get("IsDepiction") == "1" or row.get("IsGroupOf") == "1":
                    continue
                image_id = row["ImageID"]
                item = metadata.get(image_id)
                if not item:
                    continue
                url = item.get("Thumbnail300KURL") or item.get("OriginalURL")
                if not url:
                    continue
                pending.add(
                    executor.submit(
                        fetch_open_images_face,
                        {
                            "url": url,
                            "XMin": row["XMin"],
                            "XMax": row["XMax"],
                            "YMin": row["YMin"],
                            "YMax": row["YMax"],
                            "manifest": {
                                "source": "Open Images validation",
                                "image_id": image_id,
                                "landing_url": item.get("OriginalLandingURL", ""),
                                "license": item.get("License", ""),
                                "author": item.get("Author", ""),
                            },
                        },
                        image_size,
                    )
                )
                if len(pending) >= max(2, num_workers * 4):
                    pending, saved = handle_completed_face_futures(
                        pending, target_dir, limit, saved, manifest, block=True
                    )

        while pending and saved < limit:
            pending, saved = handle_completed_face_futures(
                pending, target_dir, limit, saved, manifest, block=True
            )
        for future in pending:
            future.cancel()

    (target_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return saved


def image_from_parquet_value(image_value) -> Image.Image:
    if isinstance(image_value, dict):
        if image_value.get("bytes"):
            return Image.open(io.BytesIO(image_value["bytes"]))
        return Image.open(image_value["path"])
    return Image.open(io.BytesIO(image_value))


def extract_hf_anime_faces(target_dir: Path, cache_dir: Path, limit: int, image_size: int) -> int:
    ensure_dir(target_dir)
    ensure_dir(cache_dir)
    source = ANIME_DATASETS["large"] if limit > 120 else ANIME_DATASETS["small"]
    saved = 0
    for filename in source["files"]:
        if saved >= limit:
            break
        parquet_path = hf_hub_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            filename=filename,
            cache_dir=cache_dir,
        )
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=128, columns=["image"]):
            if saved >= limit:
                break
            for image_value in batch.to_pydict()["image"]:
                if saved >= limit:
                    break
                try:
                    image = image_from_parquet_value(image_value)
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
                    output = target_dir / f"hf_anime_face_{saved:05d}.jpg"
                    image.save(output, quality=94, optimize=True)
                    saved += 1
                except Exception:
                    continue
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small starter dataset for photo-to-anime training.")
    parser.add_argument("--real-limit", type=int, default=120)
    parser.add_argument("--anime-limit", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--raw-dir", default="raw/starter")
    parser.add_argument("--train-a", default="data/trainA")
    parser.add_argument("--train-b", default="data/trainB")
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    raw_dir = ensure_dir(ROOT / args.raw_dir)
    started = time.time()
    real_count = download_open_images_faces(
        ROOT / args.train_a,
        raw_dir / "open_images",
        args.real_limit,
        args.image_size,
        args.num_workers,
    )
    anime_count = extract_hf_anime_faces(
        ROOT / args.train_b,
        raw_dir / "huggingface",
        args.anime_limit,
        args.image_size,
    )
    elapsed = time.time() - started
    print(f"saved real faces: {real_count} -> {args.train_a}")
    print(f"saved anime faces: {anime_count} -> {args.train_b}")
    print(f"elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
