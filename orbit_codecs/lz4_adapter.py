try:
    import lz4.frame
except ImportError as exc:
    raise ImportError("lz4.frame is required for LZ4Adapter. Install the 'lz4' package to use this codec.") from exc

from orbit_codecs.base import CodecAdapter


class LZ4Adapter(CodecAdapter):
    codec_id = 0

    def compress(self, data: bytes) -> bytes:
        return lz4.frame.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return lz4.frame.decompress(data)
