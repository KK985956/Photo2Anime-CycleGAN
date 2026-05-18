from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import list_image_paths


def build_transform(image_size: int, crop_size: int | None = None, augment: bool = True):
    operations = [transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC)]
    if crop_size:
        if augment:
            operations.append(transforms.RandomCrop(crop_size))
        else:
            operations.append(transforms.CenterCrop(crop_size))
    if augment:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.01),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(operations)


class UnpairedImageDataset(Dataset):
    """Loads unpaired A/B domain images for CycleGAN."""

    def __init__(
        self,
        dataroot: str | Path,
        phase: str = "train",
        image_size: int = 286,
        crop_size: int | None = 256,
        augment: bool = True,
        max_images: int | None = None,
    ):
        self.root = Path(dataroot)
        self.paths_a = list_image_paths(self.root / f"{phase}A", max_images=max_images)
        self.paths_b = list_image_paths(self.root / f"{phase}B", max_images=max_images)
        if not self.paths_a:
            raise FileNotFoundError(f"No images found in {self.root / f'{phase}A'}")
        if not self.paths_b:
            raise FileNotFoundError(f"No images found in {self.root / f'{phase}B'}")

        self.transform = build_transform(image_size=image_size, crop_size=crop_size, augment=augment)

    def __len__(self) -> int:
        return max(len(self.paths_a), len(self.paths_b))

    def __getitem__(self, index: int):
        path_a = self.paths_a[index % len(self.paths_a)]
        path_b = random.choice(self.paths_b)
        image_a = Image.open(path_a).convert("RGB")
        image_b = Image.open(path_b).convert("RGB")
        return {
            "A": self.transform(image_a),
            "B": self.transform(image_b),
            "A_path": str(path_a),
            "B_path": str(path_b),
        }

