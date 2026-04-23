from codecs.base import CodecAdapter


class RawAdapter(CodecAdapter):
    codec_id = 3

    def compress(self, data: bytes) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data
