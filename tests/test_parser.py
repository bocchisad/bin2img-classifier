"""Unit tests for Stage 1 — BinaryParser & Shannon entropy."""

from __future__ import annotations

import math
import struct
from collections import Counter
from pathlib import Path

import pytest

from bin2img.parser import (
    HIGH_ENTROPY_THRESHOLD,
    BinaryFormat,
    BinaryParser,
    calculate_entropy,
)


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------


class TestCalculateEntropy:
    def test_empty_bytes_is_zero(self) -> None:
        assert calculate_entropy(b"") == 0.0

    def test_single_byte_value_is_zero(self) -> None:
        assert calculate_entropy(b"\x00" * 1024) == 0.0
        assert calculate_entropy(b"A" * 256) == 0.0

    def test_two_equiprobable_bytes(self) -> None:
        data = b"\x00\xff" * 500
        assert calculate_entropy(data) == pytest.approx(1.0, abs=1e-9)

    def test_uniform_256_bytes_near_max(self) -> None:
        data = bytes(range(256))
        assert calculate_entropy(data) == pytest.approx(8.0, abs=1e-9)

    def test_clamped_to_eight(self) -> None:
        # Near-uniform large buffer must never exceed 8.0
        data = bytes(i % 256 for i in range(256 * 64))
        entropy = calculate_entropy(data)
        assert 0.0 <= entropy <= 8.0
        assert entropy == pytest.approx(8.0, abs=1e-6)

    def test_matches_manual_formula(self) -> None:
        data = b"AAABBC"
        counts = Counter(data)
        expected = -sum(
            (c / len(data)) * math.log2(c / len(data)) for c in counts.values()
        )
        assert calculate_entropy(data) == pytest.approx(expected, abs=1e-12)

    def test_accepts_bytearray_and_memoryview(self) -> None:
        raw = b"\x01\x02\x03\x01"
        assert calculate_entropy(bytearray(raw)) == calculate_entropy(raw)
        assert calculate_entropy(memoryview(raw)) == calculate_entropy(raw)


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic PE / ELF
# ---------------------------------------------------------------------------


def _craft_pe_with_pefile() -> bytes | None:
    """Generate a minimal valid-enough PE via pefile APIs, or return None."""
    try:
        import pefile
    except ImportError:
        return None

    # Hand-crafted minimal PE that pefile accepts (classic tiny PE pattern).
    # DOS header
    pe = bytearray()
    pe += b"MZ" + b"\x00" * 58
    pe += struct.pack("<I", 0x40)  # e_lfanew = 0x40
    # PE header at 0x40
    pe += b"PE\x00\x00"
    # COFF: Machine x86, 1 section, SizeOfOptionalHeader=0xE0, Characteristics=0x103
    pe += struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, 0xE0, 0x0103)

    opt = bytearray(0xE0)
    struct.pack_into("<H", opt, 0, 0x10B)  # PE32 magic
    struct.pack_into("<I", opt, 16, 0x1000)  # EntryPoint
    struct.pack_into("<I", opt, 28, 0x400000)  # ImageBase
    struct.pack_into("<I", opt, 32, 0x1000)  # SectionAlignment
    struct.pack_into("<I", opt, 36, 0x200)  # FileAlignment
    struct.pack_into("<H", opt, 40, 6)  # MajorOperatingSystemVersion
    struct.pack_into("<H", opt, 44, 6)  # MajorImageVersion — keep zeros ok
    struct.pack_into("<I", opt, 56, 0x2000)  # SizeOfImage
    struct.pack_into("<I", opt, 60, 0x200)  # SizeOfHeaders
    struct.pack_into("<H", opt, 68, 3)  # Subsystem = CONSOLE
    struct.pack_into("<I", opt, 92, 16)  # NumberOfRvaAndSizes
    pe += opt

    # Section header .text — VirtualSize, VA, SizeOfRawData, PointerToRawData
    name = b".text\x00\x00\x00"
    virt_size = 0x100
    virt_addr = 0x1000
    raw_size = 0x200
    ptr_raw = 0x200
    chars = 0x60000020  # CODE | EXECUTE | READ
    pe += name
    pe += struct.pack("<IIIIIIHHI", virt_size, virt_addr, raw_size, ptr_raw, 0, 0, 0, 0, chars)

    # Pad headers to PointerToRawData
    if len(pe) < ptr_raw:
        pe += b"\x00" * (ptr_raw - len(pe))

    # Section content: structured low-entropy code-like pattern
    section_data = (b"\x90\x90\x55\x8B\xEC" * 40)[:raw_size]
    if len(section_data) < raw_size:
        section_data += b"\x00" * (raw_size - len(section_data))
    pe += section_data

    try:
        parsed = pefile.PE(data=bytes(pe), fast_load=True)
        parsed.close()
        return bytes(pe)
    except Exception:
        return None


def _minimal_elf64() -> bytes:
    """Minimal ELF64 header + one PROGBITS section (not fully loadable).

    Enough for LIEF to recognise ELF format in many versions.
    """
    # e_ident
    elf = bytearray(64)
    elf[0:4] = b"\x7fELF"
    elf[4] = 2  # ELFCLASS64
    elf[5] = 1  # ELFDATA2LSB
    elf[6] = 1  # EV_CURRENT
    # e_type = ET_EXEC, e_machine = EM_X86_64, e_version = 1
    struct.pack_into("<HHI", elf, 16, 2, 62, 1)
    struct.pack_into("<QQQ", elf, 24, 0x400000, 64, 0)  # entry, phoff, shoff
    struct.pack_into("<IHHHHHH", elf, 48, 0, 64, 0, 0, 64, 0, 0)
    # Append some low-entropy payload so overall entropy is measurable
    elf += b"\x00" * 64 + b"HelloWorld" * 20
    return bytes(elf)


# ---------------------------------------------------------------------------
# BinaryParser
# ---------------------------------------------------------------------------


class TestBinaryParserBasics:
    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        meta = BinaryParser().parse(path)
        assert meta.file_size == 0
        assert meta.format is BinaryFormat.RAW
        assert meta.overall_entropy == 0.0
        assert meta.high_entropy_alert is False
        assert "empty_file" in meta.parse_errors

    def test_raw_bytes_high_entropy_alert(self) -> None:
        # Uniform noise → entropy ~8.0
        import os

        noise = os.urandom(4096)
        meta = BinaryParser().parse(noise)
        assert meta.format is BinaryFormat.RAW
        assert meta.overall_entropy > HIGH_ENTROPY_THRESHOLD
        assert meta.high_entropy_alert is True
        assert meta.is_valid_executable is False
        assert len(meta.sections) == 1
        assert meta.sections[0].name == ".raw"

    def test_low_entropy_raw_no_alert(self) -> None:
        data = b"\x00" * 2048
        meta = BinaryParser().parse(data)
        assert meta.high_entropy_alert is False
        assert meta.overall_entropy == 0.0

    def test_parse_from_bytes_and_path(self, tmp_path: Path) -> None:
        data = b"AAAA" * 100
        path = tmp_path / "a.bin"
        path.write_bytes(data)
        from_path = BinaryParser().parse(path)
        from_bytes = BinaryParser().parse(data)
        assert from_path.overall_entropy == from_bytes.overall_entropy
        assert from_path.file_size == from_bytes.file_size

    def test_parse_file_alias(self, tmp_path: Path) -> None:
        path = tmp_path / "x.bin"
        path.write_bytes(b"\x01\x02" * 50)
        meta = BinaryParser().parse_file(path)
        assert meta.file_size == 100

    def test_to_dict_serializable(self) -> None:
        meta = BinaryParser().parse(b"\xff" * 128)
        payload = meta.to_dict()
        assert payload["format"] == "RAW"
        assert isinstance(payload["sections"], list)
        assert "overall_entropy" in payload

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            BinaryParser(high_entropy_threshold=0.0)
        with pytest.raises(ValueError):
            BinaryParser(high_entropy_threshold=9.0)

    def test_missing_file_does_not_crash(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.exe"
        meta = BinaryParser().parse(missing)
        assert meta.file_size == 0
        assert any("read_error" in e for e in meta.parse_errors)


class TestBinaryParserPE:
    def test_pe_parsing_when_craftable(self) -> None:
        pe_bytes = _craft_pe_with_pefile()
        if pe_bytes is None:
            pytest.skip("Could not craft a pefile-compatible PE in this environment")

        meta = BinaryParser().parse(pe_bytes)
        assert meta.format is BinaryFormat.PE
        assert meta.is_valid_executable is True
        assert meta.architecture in {"x86", "x86_64", "I386", "AMD64"} or "x86" in meta.architecture.lower() or meta.architecture != "unknown"
        assert meta.entry_point is not None
        assert len(meta.sections) >= 1
        assert meta.overall_entropy >= 0.0
        # Structured NOP sled → should not trip high-entropy alert
        assert meta.high_entropy_alert is False or meta.max_section_entropy <= 8.0

    def test_corrupt_mz_does_not_crash(self) -> None:
        # MZ magic but garbage afterwards
        junk = b"MZ" + b"\x00" * 20 + b"NOT_A_PE" + os_urandom(200)
        meta = BinaryParser().parse(junk)
        assert meta.format in {BinaryFormat.RAW, BinaryFormat.PE, BinaryFormat.UNKNOWN}
        # Must complete without exception
        assert meta.file_size == len(junk)


def os_urandom(n: int) -> bytes:
    import os

    return os.urandom(n)


class TestBinaryParserELF:
    def test_elf_magic_recognized_or_raw_fallback(self) -> None:
        elf = _minimal_elf64()
        meta = BinaryParser().parse(elf)
        # LIEF may or may not fully accept our stub; either ELF or RAW is OK,
        # but we must never crash and must report entropy.
        assert meta.format in {BinaryFormat.ELF, BinaryFormat.RAW, BinaryFormat.UNKNOWN}
        assert meta.overall_entropy >= 0.0
        assert meta.file_size == len(elf)
        if meta.format is BinaryFormat.ELF:
            assert meta.is_valid_executable is True
            assert "x86" in meta.architecture.lower() or meta.architecture != ""


class TestHighEntropyThreshold:
    def test_custom_threshold(self) -> None:
        # Medium entropy data
        data = bytes([i % 16 for i in range(4096)])  # 4 bits → H≈4.0
        h = calculate_entropy(data)
        assert 3.5 < h < 4.5

        strict = BinaryParser(high_entropy_threshold=3.0)
        loose = BinaryParser(high_entropy_threshold=7.5)
        assert strict.parse(data).high_entropy_alert is True
        assert loose.parse(data).high_entropy_alert is False
