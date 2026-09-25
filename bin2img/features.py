"""CV & structural feature extraction for binary byteplots.

Stage 3 engine: GLCM texture descriptors, Local Binary Pattern histograms,
and PE/ELF section-entropy metadata fused into a fixed-length feature vector
for downstream CatBoost classification.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import numpy as np

from bin2img.byte_stats import extract_byte_string_stats, extract_pe_structural_features
from bin2img.constants import SUSPICIOUS_SECTION_HINTS
from bin2img.parser import BinaryMetadata, BinaryParser
from bin2img.visualizer import BinaryVisualizer, bytes_to_matrix

logger = logging.getLogger(__name__)

# GLCM configuration — mean over angles yields stable 5-D texture block.
_GLCM_DISTANCES: tuple[int, ...] = (1,)
_GLCM_ANGLES: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
_GLCM_PROPS: tuple[str, ...] = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
)
_GLCM_LEVELS: int = 32  # Quantize gray levels for speed / stability.

# Uniform LBP (P=8, R=1) → P+2 = 10 histogram bins.
_LBP_P: int = 8
_LBP_R: int = 1
_LBP_METHOD: str = "uniform"
_LBP_BINS: int = _LBP_P + 2

_STRUCT_NAMES: tuple[str, ...] = (
    "virt_raw_ratio_max",
    "high_entropy_section_ratio",
    "wx_section_ratio",
    "mean_section_raw_size_log10",
    "entry_point_present",
    "overlay_ratio",
    "import_count_log1p",
    "url_hit_ratio",
    "ip_hit_ratio",
    "suspicious_api_hits",
    "avg_string_len",
    "string_density",
)

# Ordered feature names — must stay stable across train / serve.
FEATURE_NAMES: tuple[str, ...] = (
    "mean_section_entropy",
    "max_section_entropy",
    "overall_entropy",
    "section_count",
    "executable_section_ratio",
    "file_size_log10",
    "high_entropy_alert",
    "is_valid_executable",
    "printable_ratio",
    "null_byte_ratio",
    "has_mz_header",
    "entropy_spread",
    "suspicious_section_hint",
    *_STRUCT_NAMES,
    *(f"glcm_{prop}" for prop in _GLCM_PROPS),
    *(f"lbp_{i}" for i in range(_LBP_BINS)),
)

_SUSPICIOUS_NAME_HINTS = SUSPICIOUS_SECTION_HINTS


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Named + dense feature representation for a single binary."""

    names: tuple[str, ...]
    values: np.ndarray  # shape (n_features,), float64
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("values must be a 1-D array")
        if len(self.names) != self.values.shape[0]:
            raise ValueError(
                f"names/values length mismatch: {len(self.names)} vs {self.values.shape[0]}"
            )

    def to_dict(self) -> dict[str, float]:
        """Map feature name → value."""
        return {name: float(val) for name, val in zip(self.names, self.values, strict=True)}

    def as_row(self) -> np.ndarray:
        """Return a copy suitable for stacking into a design matrix."""
        return np.asarray(self.values, dtype=np.float64).copy()


class FeatureExtractor:
    """Extract a unified feature vector from a binary file / byte payload.

    Pipeline:
        1. Parse headers & section entropy (:class:`BinaryParser`).
        2. Render grayscale byteplot (:class:`BinaryVisualizer`).
        3. Compute GLCM + LBP on the byteplot.
        4. Concatenate metadata + texture descriptors in :data:`FEATURE_NAMES` order.
    """

    def __init__(
        self,
        *,
        parser: BinaryParser | None = None,
        visualizer: BinaryVisualizer | None = None,
        glcm_levels: int = _GLCM_LEVELS,
        lbp_p: int = _LBP_P,
        lbp_r: int = _LBP_R,
    ) -> None:
        self.parser = parser or BinaryParser()
        self.visualizer = visualizer or BinaryVisualizer()
        if glcm_levels < 2 or glcm_levels > 256:
            raise ValueError(f"glcm_levels must be in [2, 256], got {glcm_levels}")
        self.glcm_levels = glcm_levels
        self.lbp_p = lbp_p
        self.lbp_r = lbp_r

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Stable feature name ordering used by this extractor."""
        lbp_bins = self.lbp_p + 2
        return (
            "mean_section_entropy",
            "max_section_entropy",
            "overall_entropy",
            "section_count",
            "executable_section_ratio",
            "file_size_log10",
            "high_entropy_alert",
            "is_valid_executable",
            "printable_ratio",
            "null_byte_ratio",
            "has_mz_header",
            "entropy_spread",
            "suspicious_section_hint",
            *_STRUCT_NAMES,
            *(f"glcm_{prop}" for prop in _GLCM_PROPS),
            *(f"lbp_{i}" for i in range(lbp_bins)),
        )

    def extract(
        self,
        source: str | Path | bytes | bytearray | memoryview | BinaryIO,
    ) -> FeatureVector:
        """Parse + visualize + featurize a binary source end-to-end.

        Bytes are loaded once so file-like streams are not double-consumed.
        """
        raw, display_path = _load_raw_once(source)
        meta = self.parser.parse(raw)
        if display_path:
            meta.file_path = display_path
        width = self.visualizer.resolve_width(len(raw))
        gray = bytes_to_matrix(raw, width)
        return self.extract_from_parts(meta, gray, raw=raw)

    def extract_from_parts(
        self,
        metadata: BinaryMetadata,
        grayscale: np.ndarray,
        *,
        raw: bytes | None = None,
    ) -> FeatureVector:
        """Build features from already-parsed metadata and a grayscale matrix."""
        meta_feats = self._metadata_features(metadata, raw=raw)
        glcm_feats = extract_glcm_features(grayscale, levels=self.glcm_levels)
        lbp_feats = extract_lbp_histogram(
            grayscale, p=self.lbp_p, r=self.lbp_r
        )

        names = self.feature_names
        values = np.concatenate(
            [meta_feats, glcm_feats, lbp_feats],
            dtype=np.float64,
        )
        if values.shape[0] != len(names):
            raise RuntimeError(
                f"feature length mismatch: got {values.shape[0]}, expected {len(names)}"
            )

        return FeatureVector(
            names=names,
            values=values,
            metadata={
                "format": metadata.format.value,
                "architecture": metadata.architecture,
                "file_size": metadata.file_size,
                "high_entropy_alert": metadata.high_entropy_alert,
                "section_count": len(metadata.sections),
            },
        )

    def extract_batch(
        self,
        sources: Sequence[str | Path | bytes],
    ) -> tuple[np.ndarray, list[FeatureVector]]:
        """Extract a design matrix ``(n_samples, n_features)`` plus per-sample vectors."""
        vectors = [self.extract(src) for src in sources]
        if not vectors:
            return np.zeros((0, len(self.feature_names)), dtype=np.float64), []
        matrix = np.vstack([v.as_row() for v in vectors])
        return matrix, vectors

    # ------------------------------------------------------------------
    # Metadata block
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_features(
        metadata: BinaryMetadata,
        *,
        raw: bytes | None = None,
    ) -> np.ndarray:
        sections = metadata.sections
        section_count = float(len(sections))
        if sections:
            exec_ratio = sum(1 for s in sections if s.is_executable) / len(sections)
            mean_ent = float(metadata.mean_section_entropy)
            max_ent = float(metadata.max_section_entropy)
            min_ent = float(min(s.entropy for s in sections))
            entropy_spread = max_ent - min_ent
            suspicious = 0.0
            for section in sections:
                name_u = section.name.upper()
                if any(hint in name_u for hint in _SUSPICIOUS_NAME_HINTS):
                    suspicious = 1.0
                    break
        else:
            exec_ratio = 0.0
            mean_ent = float(metadata.overall_entropy)
            max_ent = float(metadata.overall_entropy)
            entropy_spread = 0.0
            suspicious = 0.0

        size = max(metadata.file_size, 0)
        size_log = math.log10(size) if size > 0 else 0.0

        if raw is None:
            printable_ratio = 0.0
            null_ratio = 0.0
            has_mz = 0.0
            string_stats = extract_byte_string_stats(b"")
            pe_struct = extract_pe_structural_features(metadata, None)
        else:
            if raw:
                printable = sum(
                    1 for b in raw if 32 <= b < 127 or b in (9, 10, 13)
                )
                printable_ratio = printable / len(raw)
                null_ratio = raw.count(0) / len(raw)
            else:
                printable_ratio = 0.0
                null_ratio = 0.0
            has_mz = 1.0 if raw[:2] == b"MZ" else 0.0
            string_stats = extract_byte_string_stats(raw)
            pe_struct = extract_pe_structural_features(metadata, raw)

        struct_vals = [float(pe_struct[name]) for name in _STRUCT_NAMES[:7]]
        struct_vals.extend(float(string_stats[name]) for name in _STRUCT_NAMES[7:])

        return np.asarray(
            [
                mean_ent,
                max_ent,
                float(metadata.overall_entropy),
                section_count,
                float(exec_ratio),
                size_log,
                1.0 if metadata.high_entropy_alert else 0.0,
                1.0 if metadata.is_valid_executable else 0.0,
                float(printable_ratio),
                float(null_ratio),
                has_mz,
                float(entropy_spread),
                suspicious,
                *struct_vals,
            ],
            dtype=np.float64,
        )


def extract_glcm_features(
    image: np.ndarray,
    *,
    levels: int = _GLCM_LEVELS,
    distances: Sequence[int] = _GLCM_DISTANCES,
    angles: Sequence[float] = _GLCM_ANGLES,
) -> np.ndarray:
    """Compute mean GLCM properties over configured angles/distances.

    Returns:
        ``float64`` vector of length 5:
        contrast, dissimilarity, homogeneity, energy, correlation.
        Degenerate images yield zeros.
    """
    from skimage.feature import graycomatrix, graycoprops

    prepared = _prepare_gray_image(image, levels=levels)
    if prepared is None:
        return np.zeros(len(_GLCM_PROPS), dtype=np.float64)

    try:
        glcm = graycomatrix(
            prepared,
            distances=list(distances),
            angles=list(angles),
            levels=levels,
            symmetric=True,
            normed=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("GLCM failed: %s", exc)
        return np.zeros(len(_GLCM_PROPS), dtype=np.float64)

    feats: list[float] = []
    for prop in _GLCM_PROPS:
        try:
            props = graycoprops(glcm, prop)  # shape (n_dist, n_angles)
            value = float(np.nanmean(props))
            if not math.isfinite(value):
                value = 0.0
        except Exception:  # noqa: BLE001
            value = 0.0
        feats.append(value)
    return np.asarray(feats, dtype=np.float64)


def extract_lbp_histogram(
    image: np.ndarray,
    *,
    p: int = _LBP_P,
    r: float = _LBP_R,
    method: str = _LBP_METHOD,
) -> np.ndarray:
    """Compute a normalized uniform-LBP histogram.

    Returns:
        ``float64`` vector of length ``p + 2`` (uniform LBP), summing to 1
        when the image is non-empty; zeros otherwise.
    """
    from skimage.feature import local_binary_pattern

    bins = p + 2
    if image.size == 0 or min(image.shape[:2]) < 3:
        return np.zeros(bins, dtype=np.float64)

    gray = _as_uint8_gray(image)
    try:
        lbp = local_binary_pattern(gray, P=p, R=r, method=method)
    except Exception as exc:  # noqa: BLE001
        logger.debug("LBP failed: %s", exc)
        return np.zeros(bins, dtype=np.float64)

    # Uniform LBP codes are in [0, p+1].
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=False)
    total = hist.sum()
    if total <= 0:
        return np.zeros(bins, dtype=np.float64)
    return (hist / total).astype(np.float64)


def _prepare_gray_image(image: np.ndarray, *, levels: int) -> np.ndarray | None:
    """Quantize a grayscale image for GLCM; return None if too small."""
    if image.size == 0 or min(image.shape[:2]) < 2:
        return None
    gray = _as_uint8_gray(image)
    # Map 0..255 → 0..levels-1
    quantized = (gray.astype(np.float64) * (levels - 1) / 255.0).astype(np.uint8)
    return np.clip(quantized, 0, levels - 1)


def _as_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Ensure a 2-D uint8 grayscale view."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _load_raw_once(
    source: str | Path | bytes | bytearray | memoryview | BinaryIO,
) -> tuple[bytes, str | None]:
    """Load bytes exactly once. Returns ``(raw, display_path_or_None)``."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), None
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            return path.read_bytes(), str(path.resolve())
        except OSError as exc:
            logger.debug("Could not read %s: %s", path, exc)
            return b"", str(path)
    # File-like — read once; rewind when possible so callers can reuse the stream.
    name = str(getattr(source, "name", "<stream>"))
    try:
        data = source.read()  # type: ignore[union-attr]
    except OSError as exc:
        logger.debug("Stream read failed for %s: %s", name, exc)
        return b"", name
    if not isinstance(data, (bytes, bytearray)):
        return b"", name
    raw = bytes(data)
    seek = getattr(source, "seek", None)
    if callable(seek):
        try:
            seek(0)
        except OSError:
            pass
    return raw, name


def feature_matrix(
    sources: Sequence[str | Path | bytes],
    extractor: FeatureExtractor | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convenience: ``(X, feature_names)`` for a list of binaries."""
    ext = extractor or FeatureExtractor()
    matrix, _ = ext.extract_batch(sources)
    return matrix, ext.feature_names
