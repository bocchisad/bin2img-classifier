"""Unit tests for Stage 2 — BinaryVisualizer & image width mapping."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import numpy as np
import pytest

from bin2img.parser import calculate_entropy
from bin2img.visualizer import (
    BinaryVisualizer,
    bytes_to_matrix,
    generate_colored_heatmap,
    local_entropy_map,
    resolve_image_width,
)


class TestResolveImageWidth:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            (0, 32),
            (1024, 32),
            (10 * 1024 - 1, 32),
            (10 * 1024, 64),
            (30 * 1024 - 1, 64),
            (30 * 1024, 128),
            (100 * 1024 - 1, 128),
            (100 * 1024, 256),
            (500 * 1024 - 1, 256),
            (500 * 1024, 384),
            (1024 * 1024 - 1, 384),
            (1024 * 1024, 512),
            (5 * 1024 * 1024, 512),
        ],
    )
    def test_breakpoints(self, size: int, expected: int) -> None:
        assert resolve_image_width(size) == expected

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_image_width(-1)


class TestBytesToMatrix:
    def test_exact_fit(self) -> None:
        data = bytes(range(64))
        mat = bytes_to_matrix(data, width=8)
        assert mat.shape == (8, 8)
        assert mat.dtype == np.uint8
        assert mat[0, 0] == 0
        assert mat[0, 7] == 7

    def test_padding(self) -> None:
        data = b"\xff" * 10
        mat = bytes_to_matrix(data, width=4)
        assert mat.shape == (3, 4)
        assert mat[-1, -1] == 0  # pad
        assert mat[-1, 1] == 0xFF  # last real byte at index 9 → row2 col1

    def test_empty(self) -> None:
        mat = bytes_to_matrix(b"", width=32)
        assert mat.shape == (0, 32)

    def test_invalid_width(self) -> None:
        with pytest.raises(ValueError):
            bytes_to_matrix(b"abc", width=0)


class TestLocalEntropyMap:
    def test_uniform_noise_high(self) -> None:
        noise = os.urandom(2048)
        ent = local_entropy_map(noise, width=32, window=256)
        assert ent.shape[1] == 32
        # Most interior positions should be near-max entropy.
        assert float(np.median(ent)) > 7.0

    def test_zeros_low(self) -> None:
        zeros = b"\x00" * 1024
        ent = local_entropy_map(zeros, width=32, window=128)
        assert float(np.max(ent)) == pytest.approx(0.0, abs=1e-9)

    def test_matches_calculate_entropy_at_start(self) -> None:
        data = bytes([i % 64 for i in range(512)])
        window = 128
        ent = local_entropy_map(data, width=64, window=window)
        flat = ent.ravel()[: len(data)]
        expected = calculate_entropy(data[:window])
        assert flat[0] == pytest.approx(expected, abs=1e-9)


class TestHeatmapAndRender:
    def test_heatmap_shape_and_dtype(self) -> None:
        data = os.urandom(500)
        heat = generate_colored_heatmap(data)
        gray_w = resolve_image_width(len(data))
        assert heat.ndim == 3
        assert heat.shape[2] == 3
        assert heat.shape[1] == gray_w
        assert heat.dtype == np.uint8

    def test_low_vs_high_entropy_colour_bias(self) -> None:
        """Low-entropy regions lean blue; high-entropy lean red (JET BGR)."""
        low = BinaryVisualizer(entropy_window=64).render(b"\x00" * 512)
        high = BinaryVisualizer(entropy_window=64).render(os.urandom(512))
        # JET: low value → blue channel dominant in BGR; high → red dominant.
        low_b, low_r = float(low.heatmap[:, :, 0].mean()), float(low.heatmap[:, :, 2].mean())
        high_b, high_r = float(high.heatmap[:, :, 0].mean()), float(high.heatmap[:, :, 2].mean())
        assert low_b >= low_r  # blue-ish
        assert high_r >= high_b * 0.5  # red component elevated on noise

    def test_render_and_base64_roundtrip(self, tmp_path: Path) -> None:
        viz = BinaryVisualizer()
        result = viz.render(b"HelloBinary" * 100)
        assert result.width == 32
        assert result.height > 0
        assert result.grayscale.shape == (result.height, result.width)
        assert result.heatmap.shape == (result.height, result.width, 3)

        b64 = result.grayscale_base64()
        raw_png = base64.b64decode(b64)
        assert raw_png[:8] == b"\x89PNG\r\n\x1a\n"

        gray_path = viz.save_png(result.grayscale, tmp_path / "g.png")
        heat_path = viz.save_png(result.heatmap, tmp_path / "h.png")
        assert gray_path.is_file() and gray_path.stat().st_size > 0
        assert heat_path.is_file() and heat_path.stat().st_size > 0

    def test_save_render_pair(self, tmp_path: Path) -> None:
        viz = BinaryVisualizer()
        g, h = viz.save_render(os.urandom(2048), tmp_path, stem="demo")
        assert g.name == "demo_gray.png"
        assert h.name == "demo_heatmap.png"

    def test_from_file_path(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(bytes(range(256)) * 4)
        result = BinaryVisualizer().render(path)
        assert result.file_size == 1024
        assert result.width == 32

    def test_invalid_window(self) -> None:
        with pytest.raises(ValueError):
            BinaryVisualizer(entropy_window=0)

    def test_empty_render(self) -> None:
        result = BinaryVisualizer().render(b"")
        assert result.height == 0
        assert result.width == 32
        # PNG encoders still produce valid bytes via placeholder.
        assert len(result.grayscale_base64()) > 0
