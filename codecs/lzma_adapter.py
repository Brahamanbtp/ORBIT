import lzma

from codecs.base import CodecAdapter


class LZMAAdapter(CodecAdapter):
    codec_id = 2

    def compress(self, data: bytes) -> bytes:
        return lzma.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return lzma.decompress(data)
