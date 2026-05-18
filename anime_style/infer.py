from __future__ import annotations

from pathlib import Path

from PIL import Image

from .classical import anime_filter


class AnimeStylizer:
    """Inference wrapper with a trained CycleGAN model and a classical fallback."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        image_size: int = 256,
        num_blocks: int = 9,
        ngf: int = 64,
        use_fallback: bool = True,
    ):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.device_name = device
        self.image_size = image_size
        self.num_blocks = num_blocks
        self.ngf = ngf
        self.use_fallback = use_fallback
        self.model = None
        self.device = None

        if self.checkpoint and self.checkpoint.exists():
            self._load_model()
        elif self.checkpoint and not use_fallback:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

    @property
    def using_model(self) -> bool:
        return self.model is not None

    def _load_model(self) -> None:
        import torch

        from .models import ResnetGenerator

        if self.device_name:
            self.device = torch.device(self.device_name)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self.checkpoint, map_location=self.device)
        checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        model_ngf = int(checkpoint_args.get("ngf", self.ngf))
        model_blocks = int(checkpoint_args.get("res_blocks", self.num_blocks))
        state = checkpoint.get("G_A") or checkpoint.get("model") or checkpoint
        self.model = ResnetGenerator(3, 3, ngf=model_ngf, num_blocks=model_blocks).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _model_predict(self, image: Image.Image) -> Image.Image:
        import torch
        from torchvision import transforms

        from .utils import tensor_to_pil

        original_size = image.size
        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        result = tensor_to_pil(output)
        return result.resize(original_size, Image.Resampling.BICUBIC)

    def stylize(self, image: Image.Image) -> Image.Image:
        if self.model is not None:
            return self._model_predict(image)
        if not self.use_fallback:
            raise RuntimeError("No model is loaded and fallback is disabled.")
        return anime_filter(image)


def stylize_file(
    input_path: str | Path,
    output_path: str | Path,
    checkpoint: str | Path | None = None,
    image_size: int = 256,
    device: str | None = None,
) -> Path:
    stylizer = AnimeStylizer(checkpoint=checkpoint, image_size=image_size, device=device)
    image = Image.open(input_path).convert("RGB")
    result = stylizer.stylize(image)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    return output
