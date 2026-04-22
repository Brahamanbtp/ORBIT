def validate_roundtrip(codec, data: bytes) -> bool:
    """
    Compress data with codec, decompress result, return True if roundtrip matches.
    """
    try:
        compressed = codec.compress(data)
        decompressed = codec.decompress(compressed)
        return data == decompressed
    except Exception:
        return False
