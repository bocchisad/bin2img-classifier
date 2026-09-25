"""PE/ELF/Mach-O header analysis, section extraction, and Shannon entropy.

Production Stage 1 engine for bin2img-classifier. Prefer ``pefile`` for PE,
fall back to ``lief`` for ELF/Mach-O (and PE when pefile fails). Malformed or
non-executable inputs never raise to callers — errors are captured in metadata.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

logger = logging.getLogger(__name__)

# Entropy above this threshold strongly suggests packing / encryption.
HIGH_ENTROPY_THRESHOLD: float = 7.2

# Shannon entropy for an 8-bit alphabet is bounded to [0.0, 8.0].
_MAX_BYTE_ENTROPY: float = 8.0


class BinaryFormat(str, Enum):
    """Detected container / executable format."""

    PE = "PE"
    ELF = "ELF"
    MACHO = "Mach-O"
    RAW = "RAW"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class SectionInfo:
    """Single binary section with size and entropy metrics."""

    name: str
    virtual_size: int
    raw_size: int
    entropy: float
    is_executable: bool = False
    characteristics: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize section fields for JSON / CLI output."""
        return asdict(self)


@dataclass(slots=True)
class BinaryMetadata:
    """Normalized parse result for any supported (or raw) binary."""

    file_path: str
    file_size: int
    format: BinaryFormat
    architecture: str
    entry_point: int | None
    sections: list[SectionInfo]
    overall_entropy: float
    high_entropy_alert: bool
    is_valid_executable: bool
    parse_errors: list[str] = field(default_factory=list)
    mean_section_entropy: float = 0.0
    max_section_entropy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata (enums as values) for API / CLI consumers."""
        payload = asdict(self)
        payload["format"] = self.format.value
        return payload


def calculate_entropy(data: bytes | bytearray | memoryview) -> float:
    """Compute Shannon entropy of a byte sequence.

    Formula::

        H = -Σ p_i · log₂(p_i)   for all p_i > 0

    where ``p_i`` is the relative frequency of byte value ``i``.

    Args:
        data: Raw bytes to analyse.

    Returns:
        Entropy in bits per byte, clamped to ``[0.0, 8.0]``.
        Empty input yields ``0.0``.
    """
    if not data:
        return 0.0

    length = len(data)
    counts = Counter(bytes(data))
    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    # Numerical noise can nudge past 8.0 for near-uniform data.
    return min(max(entropy, 0.0), _MAX_BYTE_ENTROPY)


def _detect_magic(raw: bytes) -> BinaryFormat:
    """Cheap magic-byte sniff before invoking heavy parsers."""
    if len(raw) >= 2 and raw[:2] == b"MZ":
        return BinaryFormat.PE
    if len(raw) >= 4 and raw[:4] == b"\x7fELF":
        return BinaryFormat.ELF
    # Mach-O thin / fat magics (both endians)
    if len(raw) >= 4 and raw[:4] in {
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
    }:
        return BinaryFormat.MACHO
    return BinaryFormat.RAW


def _safe_section_name(raw_name: bytes | str | None) -> str:
    """Decode a section name, stripping NULs and non-printables."""
    if raw_name is None:
        return ""
    if isinstance(raw_name, bytes):
        decoded = raw_name.split(b"\x00", 1)[0].decode("latin-1", errors="replace")
    else:
        decoded = str(raw_name).split("\x00", 1)[0]
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in decoded)
    return cleaned or "<unnamed>"


def _architecture_from_machine(machine: int | str | None, *, pe: bool = False) -> str:
    """Map COFF / ELF machine identifiers to a human-readable label."""
    if machine is None:
        return "unknown"

    if isinstance(machine, str):
        return machine or "unknown"

    if pe:
        pe_map: Mapping[int, str] = {
            0x014C: "x86",
            0x8664: "x86_64",
            0x01C0: "ARM",
            0xAA64: "ARM64",
            0x01C4: "ARMv7",
            0x0200: "IA64",
        }
        return pe_map.get(machine, f"unknown(0x{machine:04X})")

    # ELF e_machine (common values)
    elf_map: Mapping[int, str] = {
        3: "x86",
        62: "x86_64",
        40: "ARM",
        183: "ARM64",
        8: "MIPS",
        243: "RISC-V",
    }
    return elf_map.get(machine, f"unknown({machine})")


class BinaryParser:
    """Parse executable binaries and compute per-section Shannon entropy.

    Usage::

        parser = BinaryParser()
        meta = parser.parse("sample.exe")
        if meta.high_entropy_alert:
            ...
    """

    def __init__(self, high_entropy_threshold: float = HIGH_ENTROPY_THRESHOLD) -> None:
        """
        Args:
            high_entropy_threshold: Section/file entropy above this value
                sets ``high_entropy_alert`` (default 7.2).
        """
        if not 0.0 < high_entropy_threshold <= _MAX_BYTE_ENTROPY:
            raise ValueError(
                f"high_entropy_threshold must be in (0, {_MAX_BYTE_ENTROPY}], "
                f"got {high_entropy_threshold}"
            )
        self.high_entropy_threshold = high_entropy_threshold

    def parse(self, source: str | Path | bytes | BinaryIO) -> BinaryMetadata:
        """Parse a binary from a path, bytes, or file-like object.

        Never raises for malformed input; failures are recorded in
        ``BinaryMetadata.parse_errors`` and format falls back to RAW.

        Args:
            source: Filesystem path, raw bytes, or readable binary stream.

        Returns:
            Populated :class:`BinaryMetadata` instance.
        """
        file_path, raw, errors = self._load_bytes(source)
        file_size = len(raw)
        overall_entropy = calculate_entropy(raw)

        metadata = BinaryMetadata(
            file_path=file_path,
            file_size=file_size,
            format=BinaryFormat.UNKNOWN,
            architecture="unknown",
            entry_point=None,
            sections=[],
            overall_entropy=overall_entropy,
            high_entropy_alert=False,
            is_valid_executable=False,
            parse_errors=list(errors),
        )

        if file_size == 0:
            metadata.parse_errors.append("empty_file")
            metadata.format = BinaryFormat.RAW
            metadata.high_entropy_alert = False
            return metadata

        parsed = False
        kind = _detect_magic(raw)
        if kind is BinaryFormat.PE:
            parsed = self._try_pefile(raw, metadata) or self._try_lief(raw, metadata)
        elif kind in {BinaryFormat.ELF, BinaryFormat.MACHO}:
            parsed = self._try_lief(raw, metadata)
        # RAW / unknown: skip heavy parsers to avoid noisy failures.

        if not parsed:
            metadata.format = BinaryFormat.RAW
            metadata.sections = [
                SectionInfo(
                    name=".raw",
                    virtual_size=file_size,
                    raw_size=file_size,
                    entropy=overall_entropy,
                    is_executable=False,
                )
            ]
            if not any("parse" in e.lower() or "lief" in e.lower() or "pefile" in e.lower() for e in metadata.parse_errors):
                metadata.parse_errors.append("unrecognized_format_treated_as_raw")

        self._finalize_entropy_flags(metadata)
        return metadata

    def parse_file(self, path: str | Path) -> BinaryMetadata:
        """Convenience wrapper around :meth:`parse` for filesystem paths."""
        return self.parse(path)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_bytes(
        self, source: str | Path | bytes | BinaryIO
    ) -> tuple[str, bytes, list[str]]:
        """Normalize input into ``(display_path, raw_bytes, errors)``."""
        errors: list[str] = []

        if isinstance(source, (bytes, bytearray, memoryview)):
            return "<bytes>", bytes(source), errors

        if isinstance(source, (str, Path)):
            path = Path(source)
            try:
                return str(path.resolve()), path.read_bytes(), errors
            except OSError as exc:
                logger.warning("Failed to read %s: %s", path, exc)
                errors.append(f"read_error:{exc}")
                return str(path), b"", errors

        # File-like
        name = getattr(source, "name", "<stream>")
        try:
            data = source.read()
            if not isinstance(data, (bytes, bytearray)):
                errors.append("stream_did_not_return_bytes")
                return str(name), b"", errors
            return str(name), bytes(data), errors
        except OSError as exc:
            errors.append(f"stream_read_error:{exc}")
            return str(name), b"", errors

    # ------------------------------------------------------------------
    # PE via pefile
    # ------------------------------------------------------------------

    def _try_pefile(self, raw: bytes, metadata: BinaryMetadata) -> bool:
        """Attempt PE parsing with ``pefile``. Returns True on success."""
        if len(raw) < 2 or raw[:2] != b"MZ":
            return False

        try:
            import pefile
        except ImportError:
            metadata.parse_errors.append("pefile_not_installed")
            return False

        pe: Any = None
        try:
            pe = pefile.PE(data=raw, fast_load=True)
            pe.parse_data_directories(
                directories=[
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                ]
            )
        except Exception as exc:  # noqa: BLE001 — must not crash on bad PE
            logger.debug("pefile failed: %s", exc)
            metadata.parse_errors.append(f"pefile_error:{type(exc).__name__}")
            self._safe_close_pe(pe)
            return False

        try:
            metadata.format = BinaryFormat.PE
            metadata.is_valid_executable = True

            machine = int(getattr(pe.FILE_HEADER, "Machine", 0) or 0)
            metadata.architecture = _architecture_from_machine(machine, pe=True)

            try:
                metadata.entry_point = int(pe.OPTIONAL_HEADER.AddressOfEntryPoint)
            except (AttributeError, TypeError, ValueError):
                metadata.entry_point = None
                metadata.parse_errors.append("pe_missing_entry_point")

            sections: list[SectionInfo] = []
            for section in getattr(pe, "sections", []) or []:
                try:
                    name = _safe_section_name(section.Name)
                    virtual_size = int(getattr(section, "Misc_VirtualSize", 0) or 0)
                    raw_size = int(getattr(section, "SizeOfRawData", 0) or 0)
                    chars = int(getattr(section, "Characteristics", 0) or 0)
                    is_exec = bool(chars & 0x20000000)  # IMAGE_SCN_MEM_EXECUTE
                    try:
                        data = section.get_data() or b""
                    except Exception:  # noqa: BLE001
                        data = b""
                        metadata.parse_errors.append(f"section_data_error:{name}")
                    entropy = calculate_entropy(data) if data else 0.0
                    sections.append(
                        SectionInfo(
                            name=name,
                            virtual_size=virtual_size,
                            raw_size=raw_size,
                            entropy=round(entropy, 6),
                            is_executable=is_exec,
                            characteristics=chars,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    metadata.parse_errors.append(
                        f"section_parse_error:{type(exc).__name__}"
                    )
            metadata.sections = sections
            return True
        finally:
            self._safe_close_pe(pe)

    @staticmethod
    def _safe_close_pe(pe: Any) -> None:
        if pe is None:
            return
        try:
            pe.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # LIEF fallback (ELF / Mach-O / PE)
    # ------------------------------------------------------------------

    def _try_lief(self, raw: bytes, metadata: BinaryMetadata) -> bool:
        """Attempt parsing with ``lief``. Returns True on success."""
        try:
            import lief
        except ImportError:
            metadata.parse_errors.append("lief_not_installed")
            return False

        try:
            # Quiet LIEF stderr noise ("Unknown format", etc.) for bad inputs.
            if hasattr(lief, "logging") and hasattr(lief.logging, "disable"):
                lief.logging.disable()
        except Exception:  # noqa: BLE001
            pass

        try:
            binary = lief.parse(raw)
        except Exception as exc:  # noqa: BLE001
            logger.debug("lief.parse raised: %s", exc)
            metadata.parse_errors.append(f"lief_error:{type(exc).__name__}")
            return False

        if binary is None:
            return False

        try:
            fmt = self._classify_lief_format(binary, lief)
            if fmt is BinaryFormat.UNKNOWN:
                return False

            metadata.format = fmt
            metadata.is_valid_executable = True
            metadata.architecture = self._lief_architecture(binary, lief)
            metadata.entry_point = self._lief_entrypoint(binary)
            metadata.sections = self._lief_sections(binary, metadata)
            return True
        except Exception as exc:  # noqa: BLE001
            metadata.parse_errors.append(f"lief_extract_error:{type(exc).__name__}")
            return False

    @staticmethod
    def _classify_lief_format(binary: Any, lief_mod: Any) -> BinaryFormat:
        """Map a LIEF binary object to :class:`BinaryFormat`."""
        try:
            if isinstance(binary, lief_mod.PE.Binary):
                return BinaryFormat.PE
            if isinstance(binary, lief_mod.ELF.Binary):
                return BinaryFormat.ELF
            if isinstance(binary, lief_mod.MachO.Binary) or isinstance(
                binary, lief_mod.MachO.FatBinary
            ):
                return BinaryFormat.MACHO
        except Exception:  # noqa: BLE001
            pass

        # Older / alternate LIEF APIs expose .format
        fmt_attr = getattr(binary, "format", None)
        if fmt_attr is not None:
            name = str(fmt_attr).upper()
            if "PE" in name:
                return BinaryFormat.PE
            if "ELF" in name:
                return BinaryFormat.ELF
            if "MACHO" in name or "MACH-O" in name:
                return BinaryFormat.MACHO
        return BinaryFormat.UNKNOWN

    @staticmethod
    def _lief_architecture(binary: Any, lief_mod: Any) -> str:
        """Best-effort architecture string from a LIEF binary."""
        # PE
        try:
            if isinstance(binary, lief_mod.PE.Binary):
                header = binary.header
                machine = getattr(header, "machine", None)
                if machine is not None:
                    return str(machine).replace("MACHINE_TYPES.", "").replace(
                        "Header.MACHINE_TYPES.", ""
                    )
        except Exception:  # noqa: BLE001
            pass

        # ELF
        try:
            if isinstance(binary, lief_mod.ELF.Binary):
                machine = getattr(binary.header, "machine_type", None)
                if machine is not None:
                    return str(machine).replace("ARCH.", "").replace(
                        "ARCH_", ""
                    )
        except Exception:  # noqa: BLE001
            pass

        # Mach-O
        try:
            header = getattr(binary, "header", None)
            cpu = getattr(header, "cpu_type", None) if header else None
            if cpu is not None:
                return str(cpu).replace("CPU_TYPES.", "")
        except Exception:  # noqa: BLE001
            pass

        return "unknown"

    @staticmethod
    def _lief_entrypoint(binary: Any) -> int | None:
        """Extract entry point RVA/VA when available."""
        for attr in ("entrypoint", "entry_point"):
            value = getattr(binary, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _lief_sections(
        self, binary: Any, metadata: BinaryMetadata
    ) -> list[SectionInfo]:
        """Extract section list from a LIEF binary object."""
        sections: list[SectionInfo] = []
        raw_sections: Sequence[Any] = getattr(binary, "sections", None) or []

        for section in raw_sections:
            try:
                name = _safe_section_name(getattr(section, "name", None))
                virtual_size = int(
                    getattr(section, "virtual_size", None)
                    or getattr(section, "size", 0)
                    or 0
                )
                raw_size = int(
                    getattr(section, "sizeof_raw_data", None)
                    or getattr(section, "size", 0)
                    or 0
                )

                content = b""
                for content_attr in ("content", "data"):
                    raw_content = getattr(section, content_attr, None)
                    if raw_content is None:
                        continue
                    try:
                        content = bytes(raw_content)
                    except Exception:  # noqa: BLE001
                        content = b""
                    break

                entropy = calculate_entropy(content) if content else 0.0
                is_exec = self._lief_section_executable(section)

                chars = getattr(section, "characteristics", None)
                try:
                    chars_int = int(chars) if chars is not None else None
                except (TypeError, ValueError):
                    chars_int = None

                sections.append(
                    SectionInfo(
                        name=name,
                        virtual_size=virtual_size,
                        raw_size=raw_size if raw_size else len(content),
                        entropy=round(entropy, 6),
                        is_executable=is_exec,
                        characteristics=chars_int,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                metadata.parse_errors.append(
                    f"lief_section_error:{type(exc).__name__}"
                )
        return sections

    @staticmethod
    def _lief_section_executable(section: Any) -> bool:
        """Detect executable flag across PE/ELF/Mach-O section APIs."""
        # PE characteristics bit
        chars = getattr(section, "characteristics", None)
        if isinstance(chars, int) and chars & 0x20000000:
            return True

        # ELF flags (SHF_EXECINSTR = 0x4) or enum with EXECUTE
        flags = getattr(section, "flags", None)
        if flags is not None:
            try:
                if int(flags) & 0x4:
                    return True
            except (TypeError, ValueError):
                pass
            if "EXEC" in str(flags).upper():
                return True

        # Segment / section has EXECUTE in flags list
        for attr in ("has", "has_characteristic"):
            method = getattr(section, attr, None)
            if callable(method):
                for token in ("EXECUTE", "EXECINSTR", "MEM_EXECUTE"):
                    try:
                        if method(token):
                            return True
                    except Exception:  # noqa: BLE001
                        continue

        segment_type = str(getattr(section, "type", "")).upper()
        if "EXEC" in segment_type:
            return True

        return False

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _finalize_entropy_flags(self, metadata: BinaryMetadata) -> None:
        """Fill aggregate entropy fields and high-entropy alert."""
        if metadata.sections:
            entropies = [s.entropy for s in metadata.sections]
            metadata.mean_section_entropy = round(
                sum(entropies) / len(entropies), 6
            )
            metadata.max_section_entropy = round(max(entropies), 6)
        else:
            metadata.mean_section_entropy = round(metadata.overall_entropy, 6)
            metadata.max_section_entropy = round(metadata.overall_entropy, 6)

        section_hot = any(
            s.entropy > self.high_entropy_threshold for s in metadata.sections
        )
        file_hot = metadata.overall_entropy > self.high_entropy_threshold
        metadata.high_entropy_alert = bool(section_hot or file_hot)
        metadata.overall_entropy = round(metadata.overall_entropy, 6)
