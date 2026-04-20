import struct


class BinaryWriter:
    _HEADER_FORMAT = "<IHIH"
    _RESERVED = 0

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def write_block(self, compressed_data: bytes, codec_id: int, block_id: int) -> None:
        data_length = len(compressed_data)
        header = struct.pack(
            self._HEADER_FORMAT,
            block_id,
            codec_id,
            data_length,
            self._RESERVED,
        )

        with open(self.filepath, "ab") as fh:
            fh.write(header)
            fh.write(compressed_data)
