try:
    import zstandard as zstd
except ImportError as exc:
    raise ImportError("zstandard is required for ZstdAdapter. Install the 'zstandard' package to use this codec.") from exc

from orbit_codecs.base import CodecAdapter


class ZstdAdapter(CodecAdapter):
    codec_id = 1

    def compress(self, data: bytes) -> bytes:
        compressor = zstd.ZstdCompressor()
        return compressor.compress(data)

    def decompress(self, data: bytes) -> bytes:
        decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(data)
