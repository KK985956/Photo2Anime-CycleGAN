from __future__ import annotations

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


def _pil_cartoon_filter(image: Image.Image, color_bits: int = 4, edge_strength: float = 0.7) -> Image.Image:
    image = image.convert("RGB")
    smooth = image.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.SMOOTH_MORE)
    poster = ImageOps.posterize(smooth, bits=max(2, min(color_bits, 8)))
    edges = ImageOps.grayscale(image).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.invert(edges).point(lambda pixel: 255 if pixel > 120 else int(255 * (1 - edge_strength)))
    edges = ImageOps.colorize(edges, black=(28, 25, 32), white=(255, 255, 255))
    blended = ImageChops.multiply(poster, edges)
    return ImageEnhance.Color(blended).enhance(1.12)


def anime_filter(
    image: Image.Image,
    color_levels: int = 12,
    edge_strength: float = 0.72,
    smoothing_passes: int = 2,
) -> Image.Image:
    """Fast non-neural fallback that produces a usable anime-like preview."""

    image = image.convert("RGB")
    try:
        import cv2
        import numpy as np
    except Exception:
        return _pil_cartoon_filter(image, color_bits=4, edge_strength=edge_strength)

    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    smooth = bgr
    for _ in range(max(1, smoothing_passes)):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=80, sigmaSpace=80)

    levels = max(4, min(int(color_levels), 32))
    step = max(1, 256 // levels)
    quantized = (smooth // step) * step + step // 2
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=6,
    )
    edge_alpha = np.clip(edge_strength, 0.0, 1.0)
    softened_edges = cv2.addWeighted(edges, edge_alpha, np.full_like(edges, 255), 1 - edge_alpha, 0)
    cartoon = cv2.bitwise_and(quantized, quantized, mask=softened_edges)
    cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(cartoon_rgb)
    return ImageEnhance.Color(result).enhance(1.14)

