

import os
from io.format import ORBITHeader, BlockHeader, write_file_header, write_block_header



class BinaryWriter:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._fh = None

    def open_file(self, n_blocks: int, block_size: int) -> None:
        self._fh = open(self.filepath, "wb")
        write_file_header(self._fh, n_blocks, block_size)

    def close_file(self) -> None:
        if self._fh:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except Exception:
                pass
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_file()

    def write_block(self, compressed_data: bytes, codec_id: int, block_id: int) -> None:
        if self._fh is None:
            raise RuntimeError("File not opened. Call open_file() first.")
        header = BlockHeader(
            block_id=block_id,
            codec_id=codec_id,
            original_size=None,  # Placeholder, must be set by caller if needed
            compressed_size=len(compressed_data),
        )
        # If original_size is not set, fallback to compressed_size
        if header.original_size is None:
            header.original_size = header.compressed_size
        write_block_header(self._fh, header)
        self._fh.write(compressed_data)
