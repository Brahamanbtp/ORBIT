from __future__ import annotations

import os

from orbit_codecs import CODEC_REGISTRY
from orbit_io.format import (
    BlockHeader,
    write_block_header,
    write_file_header,
)


def _build_codec_versions() -> str:
    """Return a short JSON string of codec names for the file header."""
    import json
    versions: dict[str, str] = {}
    for codec_id, codec in CODEC_REGISTRY.items():
        versions[type(codec).__name__] = str(codec_id)
    raw = json.dumps(versions)
    # header field is 32 bytes — truncate/pad
    return raw[:32].ljust(32)


def _build_checksum() -> int:
    """XOR of all registered codec_ids."""
    result = 0
    for codec_id in CODEC_REGISTRY:
        result ^= codec_id
    return result


class BinaryWriter:
    """Writes ORBIT-format compressed output files."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._fh = None

    # ------------------------------------------------------------------ #
    #  File lifecycle                                                      #
    # ------------------------------------------------------------------ #

    def open_file(self, n_blocks: int, block_size: int) -> None:
        self._fh = open(self.filepath, "wb")
        write_file_header(
            self._fh,
            n_blocks,
            block_size,
            codec_registry_checksum=_build_checksum(),
            codec_versions=_build_codec_versions(),
        )

    def close_file(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except OSError:
                pass
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "BinaryWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_file()

    # ------------------------------------------------------------------ #
    #  Block writing                                                       #
    # ------------------------------------------------------------------ #

    def write_block(
        self,
        compressed_data: bytes,
        codec_id: int,
        block_id: int,
        original_size: int = 0,
    ) -> None:
        if self._fh is None:
            raise RuntimeError("File not opened. Call open_file() first.")
        header = BlockHeader(
            block_id=block_id,
            codec_id=codec_id,
            original_size=original_size,
            compressed_size=len(compressed_data),
        )
        write_block_header(self._fh, header)
        self._fh.write(compressed_data)