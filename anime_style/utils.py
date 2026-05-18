from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def list_image_paths(root: str | Path, max_images: int | None = None) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    paths = [
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    if max_images and max_images > 0:
        return paths[:max_images]
    return paths


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def tensor_to_pil(tensor):
    import torch
    from PIL import Image

    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) * 127.5
    array = tensor.permute(1, 2, 0).numpy().astype("uint8")
    return Image.fromarray(array)


class ImagePool:
    """History buffer from the original CycleGAN paper."""

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size <= 0:
            return images

        import torch

        output = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image.detach().clone())
                output.append(image)
            elif random.random() > 0.5:
                index = random.randrange(len(self.images))
                old = self.images[index].clone()
                self.images[index] = image.detach().clone()
                output.append(old)
            else:
                output.append(image)
        return torch.cat(output, dim=0)


def set_requires_grad(models: Iterable, requires_grad: bool) -> None:
    if not isinstance(models, Sequence):
        models = [models]
    for model in models:
        if model is None:
            continue
        for parameter in model.parameters():
            parameter.requires_grad = requires_grad

