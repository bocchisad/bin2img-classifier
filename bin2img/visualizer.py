"""Byte-array → 2D image conversion and thermal entropy heatmaps.

Stage 2 render engine for bin2img-classifier. Maps raw executable bytes
into grayscale byteplots and blue→red thermal overlays where colour tracks
local Shannon entropy (low = code / structured, high = packed / encrypted).
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Adaptive width breakpoints (bytes → image width in pixels).
_WIDTH_RULES: tuple[tuple[int, int], ...] = (
    (10 * 1024, 32),
    (30 * 1024, 64),
    (100 * 1024, 128),
    (500 * 1024, 256),
    (1024 * 1024, 384),
)
_WIDTH_LARGE: int = 512

# Local-entropy window for thermal colouring (bytes).
DEFAULT_ENTROPY_WINDOW: int = 256

# OpenCV Jet: blue (low) → red (high) — matches Stage 2 thermal semantics.
_THERMAL_COLORMAP: int = cv2.COLORMAP_JET


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    """Rendered grayscale + thermal views for a binary payload."""

    width: int
    height: int
    grayscale: np.ndarray  # HxW uint8
    heatmap: np.ndarray  # HxW x 3 BGR uint8
    entropy_map: np.ndarray  # HxW float64, bits/byte in [0, 8]
    file_size: int

    def grayscale_png_bytes(self) -> bytes:
        """Encode grayscale view as PNG bytes."""
        return _encode_png(self.grayscale)

    def heatmap_png_bytes(self) -> bytes:
        """Encode thermal heatmap as PNG bytes."""
        return _encode_png(self.heatmap)

    def grayscale_base64(self) -> str:
        """Base64-encoded grayscale PNG (no data-URI prefix)."""
        return base64.b64encode(self.grayscale_png_bytes()).decode("ascii")

    def heatmap_base64(self) -> str:
        """Base64-encoded thermal PNG (no data-URI prefix)."""
        return base64.b64encode(self.heatmap_png_bytes()).decode("ascii")


def resolve_image_width(file_size: int) -> int:
    """Map binary size to a standard byteplot width.

    Rules:
        < 10 KB          → 32
        10–30 KB         → 64
        30–100 KB        → 128
        100–500 KB       → 256
        500 KB–1 MB      → 384
        > 1 MB           → 512
    """
    if file_size < 0:
        raise ValueError(f"file_size must be >= 0, got {file_size}")
    for limit, width in _WIDTH_RULES:
        if file_size < limit:
            return width
    return _WIDTH_LARGE


def bytes_to_matrix(data: bytes | bytearray | memoryview, width: int) -> np.ndarray:
    """Convert raw bytes into a 2D ``uint8`` matrix (row-major byteplot).

    The last row is zero-padded when ``len(data)`` is not divisible by ``width``.

    Args:
        data: Binary payload.
        width: Target image width in pixels (must be > 0).

    Returns:
        Array of shape ``(height, width)`` with dtype ``uint8``.
        Empty input yields shape ``(0, width)``.
    """
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")

    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    if arr.size == 0:
        return np.zeros((0, width), dtype=np.uint8)

    remainder = arr.size % width
    if remainder:
        arr = np.pad(arr, (0, width - remainder), mode="constant", constant_values=0)

    height = arr.size // width
    return arr.reshape((height, width))


def local_entropy_map(
    data: bytes | bytearray | memoryview,
    width: int,
    *,
    window: int = DEFAULT_ENTROPY_WINDOW,
) -> np.ndarray:
    """Compute per-byte local Shannon entropy and reshape to image geometry.

    Uses a rolling histogram over a sliding window of ``window`` bytes so cost
    stays O(n) rather than O(n · window).

    Args:
        data: Binary payload.
        width: Image width used for reshape / padding (same as grayscale).
        window: Sliding window size in bytes (clamped to ``[1, len(data)]``).

    Returns:
        ``float64`` array shaped like the grayscale matrix; values in ``[0, 8]``.
    """
    raw = bytes(data)
    n = len(raw)
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")

    if n == 0:
        return np.zeros((0, width), dtype=np.float64)

    win = max(1, min(int(window), n))
    ent = _rolling_entropy(raw, win)

    height = (n + width - 1) // width
    pad = height * width - n
    if pad > 0:
        ent = np.pad(ent, (0, pad), mode="constant", constant_values=0.0)
    return ent.reshape((height, width))


def _rolling_entropy(raw: bytes, window: int) -> np.ndarray:
    """Per-offset Shannon entropy of the forward window ``raw[i : i+window]``."""
    n = len(raw)
    ent = np.empty(n, dtype=np.float64)
    hist = np.zeros(256, dtype=np.int32)

    for b in raw[:window]:
        hist[b] += 1
    length = window
    ent[0] = _entropy_from_hist(hist, length)

    for i in range(1, n):
        hist[raw[i - 1]] -= 1
        length -= 1
        add = i + window - 1
        if add < n:
            hist[raw[add]] += 1
            length += 1
        ent[i] = _entropy_from_hist(hist, length) if length > 0 else 0.0

    return ent

def _entropy_from_hist(hist: np.ndarray, length: int) -> float:
    """Shannon entropy from a 256-bin count histogram of ``length`` samples."""
    if length <= 0:
        return 0.0
    # Only non-zero bins contribute.
    nz = hist[hist > 0].astype(np.float64)
    probs = nz / float(length)
    return float(-np.sum(probs * np.log2(probs)))


def generate_colored_heatmap(
    data: bytes | bytearray | memoryview,
    *,
    width: int | None = None,
    window: int = DEFAULT_ENTROPY_WINDOW,
) -> np.ndarray:
    """Render a thermal (blue→red) entropy heatmap for a binary payload.

    Blue ≈ low entropy (code / padding); red ≈ high entropy (packed / encrypted).

    Args:
        data: Binary payload.
        width: Optional override; defaults to :func:`resolve_image_width`.
        window: Local-entropy window in bytes.

    Returns:
        BGR ``uint8`` image of shape ``(H, W, 3)`` suitable for OpenCV / PNG.
        Empty input yields shape ``(0, W, 3)``.
    """
    raw = bytes(data)
    w = width if width is not None else resolve_image_width(len(raw))
    entropy = local_entropy_map(raw, w, window=window)
    return _entropy_to_bgr(entropy)


def _entropy_to_bgr(entropy: np.ndarray) -> np.ndarray:
    """Map entropy matrix ``[0, 8]`` → Jet colormap BGR image."""
    if entropy.size == 0:
        h, w = entropy.shape if entropy.ndim == 2 else (0, 0)
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Scale 0..8 → 0..255 for cv2.applyColorMap.
    scaled = np.clip(entropy / 8.0 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, _THERMAL_COLORMAP)


def _encode_png(image: np.ndarray) -> bytes:
    """Encode a grayscale or BGR image as PNG bytes."""
    if image.size == 0:
        # OpenCV rejects empty mats — emit a 1x1 transparent-ish placeholder.
        placeholder = np.zeros((1, 1), dtype=np.uint8) if image.ndim == 2 else np.zeros(
            (1, 1, 3), dtype=np.uint8
        )
        ok, buf = cv2.imencode(".png", placeholder)
    else:
        ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cv2.imencode failed for PNG")
    return bytes(buf)


def _load_bytes(source: str | Path | bytes | bytearray | memoryview | BinaryIO) -> bytes:
    """Normalize path / bytes / stream into raw bytes."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    data = source.read()
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("file-like object must yield bytes")
    return bytes(data)


class BinaryVisualizer:
    """Convert binaries into grayscale byteplots and thermal entropy heatmaps.

    Usage::

        viz = BinaryVisualizer()
        result = viz.render("sample.exe")
        viz.save_png(result.heatmap, "heatmap.png")
        b64 = result.heatmap_base64()
    """

    def __init__(self, entropy_window: int = DEFAULT_ENTROPY_WINDOW) -> None:
        """
        Args:
            entropy_window: Sliding window (bytes) for local entropy colouring.
        """
        if entropy_window < 1:
            raise ValueError(f"entropy_window must be >= 1, got {entropy_window}")
        self.entropy_window = entropy_window

    def resolve_width(self, file_size: int) -> int:
        """Public wrapper around :func:`resolve_image_width`."""
        return resolve_image_width(file_size)

    def to_grayscale(
        self,
        source: str | Path | bytes | bytearray | memoryview | BinaryIO,
        *,
        width: int | None = None,
    ) -> np.ndarray:
        """Build a 2D ``uint8`` grayscale byteplot from a binary source."""
        raw = _load_bytes(source)
        w = width if width is not None else self.resolve_width(len(raw))
        return bytes_to_matrix(raw, w)

    def generate_colored_heatmap(
        self,
        source: str | Path | bytes | bytearray | memoryview | BinaryIO,
        *,
        width: int | None = None,
    ) -> np.ndarray:
        """Build a BGR thermal heatmap (blue=low entropy, red=high)."""
        raw = _load_bytes(source)
        w = width if width is not None else self.resolve_width(len(raw))
        return generate_colored_heatmap(raw, width=w, window=self.entropy_window)

    def render(
        self,
        source: str | Path | bytes | bytearray | memoryview | BinaryIO,
        *,
        width: int | None = None,
    ) -> VisualizationResult:
        """Produce grayscale, entropy map, and thermal heatmap together."""
        raw = _load_bytes(source)
        w = width if width is not None else self.resolve_width(len(raw))
        gray = bytes_to_matrix(raw, w)
        entropy = local_entropy_map(raw, w, window=self.entropy_window)
        heatmap = _entropy_to_bgr(entropy)
        return VisualizationResult(
            width=w,
            height=int(gray.shape[0]),
            grayscale=gray,
            heatmap=heatmap,
            entropy_map=entropy,
            file_size=len(raw),
        )

    def save_png(
        self,
        image: np.ndarray,
        path: str | Path,
    ) -> Path:
        """Write a grayscale or BGR image to ``path`` as PNG."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        png = _encode_png(image)
        out.write_bytes(png)
        return out

    def to_base64_png(self, image: np.ndarray) -> str:
        """Return Base64-encoded PNG string for Web UI payloads."""
        return base64.b64encode(_encode_png(image)).decode("ascii")

    def save_render(
        self,
        source: str | Path | bytes | bytearray | memoryview | BinaryIO,
        output_dir: str | Path,
        *,
        stem: str = "binary",
        width: int | None = None,
    ) -> tuple[Path, Path]:
        """Render and save ``{stem}_gray.png`` + ``{stem}_heatmap.png``.

        Returns:
            ``(grayscale_path, heatmap_path)``.
        """
        result = self.render(source, width=width)
        out_dir = Path(output_dir)
        gray_path = self.save_png(result.grayscale, out_dir / f"{stem}_gray.png")
        heat_path = self.save_png(result.heatmap, out_dir / f"{stem}_heatmap.png")
        return gray_path, heat_path
