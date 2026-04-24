from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import BinaryIO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORBIT_MAGIC: bytes = b"ORBT"
_VERSION: int = 1

# File header layout (fixed size):
#   4s  magic
#   H   version
#   I   n_blocks
#   I   block_size
#   I   codec_registry_checksum
#   32s codec_versions  (JSON-encoded, truncated/padded to 32 bytes)
#
# Total: 4 + 2 + 4 + 4 + 4 + 32 = 50 bytes
_HEADER_FORMAT = ">4sHIII32s"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)  # 50

# Block header layout (fixed 16 bytes):
#   I  block_id
#   H  codec_id
#   I  original_size
#   I  compressed_size
#   H  reserved
#
# Total: 4 + 2 + 4 + 4 + 2 = 16 bytes
_BLOCK_HEADER_FORMAT = ">IHIIH"
_BLOCK_HEADER_SIZE = struct.calcsize(_BLOCK_HEADER_FORMAT)  # 16


# ---------------------------------------------------------------------------
# Header dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ORBITHeader:
    magic: bytes
    version: int
    n_blocks: int
    block_size: int
    codec_registry_checksum: int = 0
    codec_versions: str = "{}"


@dataclass
class BlockHeader:
    block_id: int
    codec_id: int
    original_size: int
    compressed_size: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_codec_versions(codec_versions: str | dict) -> bytes:
    """Encode codec_versions to a 32-byte padded/truncated bytes field."""
    if isinstance(codec_versions, dict):
        codec_versions = json.dumps(codec_versions)
    raw = codec_versions.encode("utf-8")
    # Truncate to 32 bytes, then pad with null bytes to exactly 32
    return raw[:32].ljust(32, b"\x00")


def _decode_codec_versions(raw: bytes) -> str:
    """Decode 32-byte field back to a string, stripping null padding."""
    return raw.rstrip(b"\x00").decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# File header read / write
# ---------------------------------------------------------------------------

def write_file_header(
    f: BinaryIO,
    n_blocks: int,
    block_size: int,
    codec_registry_checksum: int = 0,
    codec_versions: str | dict = "{}",
) -> None:
    """Write the ORBIT file header to an open binary file."""
    cv_bytes = _encode_codec_versions(codec_versions)
    header = struct.pack(
        _HEADER_FORMAT,
        ORBIT_MAGIC,
        _VERSION,
        n_blocks,
        block_size,
        codec_registry_checksum,
        cv_bytes,
    )
    f.write(header)


def read_file_header(f: BinaryIO) -> ORBITHeader:
    """Read and parse the ORBIT file header from an open binary file."""
    raw = f.read(_HEADER_SIZE)
    if len(raw) < _HEADER_SIZE:
        raise ValueError(
            f"File too short to contain ORBIT header "
            f"(got {len(raw)} bytes, need {_HEADER_SIZE})"
        )
    magic, version, n_blocks, block_size, checksum, cv_raw = struct.unpack(
        _HEADER_FORMAT, raw
    )
    if magic != ORBIT_MAGIC:
        raise ValueError(
            f"Invalid ORBIT magic bytes: expected {ORBIT_MAGIC!r}, got {magic!r}"
        )
    return ORBITHeader(
        magic=magic,
        version=version,
        n_blocks=n_blocks,
        block_size=block_size,
        codec_registry_checksum=checksum,
        codec_versions=_decode_codec_versions(cv_raw),
    )


# ---------------------------------------------------------------------------
# Block header read / write
# ---------------------------------------------------------------------------

def write_block_header(f: BinaryIO, header: BlockHeader) -> None:
    """Write a 16-byte block header to an open binary file."""
    packed = struct.pack(
        _BLOCK_HEADER_FORMAT,
        header.block_id,
        header.codec_id,
        header.original_size,
        header.compressed_size,
        0,  # reserved
    )
    f.write(packed)


def read_block_header(f: BinaryIO) -> BlockHeader:
    """Read and parse a 16-byte block header from an open binary file."""
    raw = f.read(_BLOCK_HEADER_SIZE)
    if len(raw) < _BLOCK_HEADER_SIZE:
        raise ValueError(
            f"File too short for block header "
            f"(got {len(raw)} bytes, need {_BLOCK_HEADER_SIZE})"
        )
    block_id, codec_id, original_size, compressed_size, _reserved = struct.unpack(
        _BLOCK_HEADER_FORMAT, raw
    )
    return BlockHeader(
        block_id=block_id,
        codec_id=codec_id,
        original_size=original_size,
        compressed_size=compressed_size,
    )