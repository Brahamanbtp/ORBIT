from collections.abc import Iterator


class StreamingReader:
    def __init__(self, filepath: str, block_size: int) -> None:
        self.filepath = filepath
        self.block_size = block_size

    def __iter__(self) -> Iterator[bytes]:
        with open(self.filepath, "rb", buffering=self.block_size) as fh:
            while True:
                chunk = fh.read(self.block_size)
                if not chunk:
                    break
                yield chunk
