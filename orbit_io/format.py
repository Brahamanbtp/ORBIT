import struct
from dataclasses import dataclass
from typing import BinaryIO

ORBIT_MAGIC = b"ORBT"



@dataclass
class ORBITHeader:
    magic: bytes
    version: int
    n_blocks: int
    block_size: int
    codec_registry_checksum: int = 0
    codec_versions: bytes = b""  # 32 bytes, JSON-encoded dict, padded/truncated



_HEADER_FORMAT = "<4sIIII32s"  # add 32-byte codec_versions string
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
_VERSION = 1

def write_file_header(f: BinaryIO, n_blocks: int, block_size: int, codec_registry_checksum: int, codec_versions: bytes) -> None:
    # codec_versions must be 32 bytes (padded/truncated)
    if len(codec_versions) > 32:
        codec_versions = codec_versions[:32]
    elif len(codec_versions) < 32:
        codec_versions = codec_versions.ljust(32, b" ")
    header = struct.pack(_HEADER_FORMAT, ORBIT_MAGIC, _VERSION, n_blocks, block_size, codec_registry_checksum, codec_versions)
    f.write(header)

def read_file_header(f: BinaryIO) -> ORBITHeader:
    header_bytes = f.read(_HEADER_SIZE)
    magic, version, n_blocks, block_size, codec_registry_checksum, codec_versions = struct.unpack(_HEADER_FORMAT, header_bytes)
    return ORBITHeader(
        magic=magic,
        version=version,
        n_blocks=n_blocks,
        block_size=block_size,
        codec_registry_checksum=codec_registry_checksum,
        codec_versions=codec_versions,
    )


# 16-byte block header: block_id:uint32, codec_id:uint32, original_size:uint32, compressed_size:uint32
@dataclass
class BlockHeader:
    block_id: int
    codec_id: int
    original_size: int
    compressed_size: int

_BLOCK_HEADER_FORMAT = "<IIII"
_BLOCK_HEADER_SIZE = struct.calcsize(_BLOCK_HEADER_FORMAT)

def write_block_header(f: BinaryIO, header: BlockHeader) -> None:
    packed = struct.pack(
        _BLOCK_HEADER_FORMAT,
        header.block_id,
        header.codec_id,
        header.original_size,
        header.compressed_size,
    )
    f.write(packed)

def read_block_header(f: BinaryIO) -> BlockHeader:
    header_bytes = f.read(_BLOCK_HEADER_SIZE)
    block_id, codec_id, original_size, compressed_size = struct.unpack(_BLOCK_HEADER_FORMAT, header_bytes)
    return BlockHeader(
        block_id=block_id,
        codec_id=codec_id,
        original_size=original_size,
        compressed_size=compressed_size,
    )
