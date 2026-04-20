from collections.abc import Iterable, Iterator

from core.block import Block


def split_into_blocks(data_iter: Iterable[bytes], block_size: int) -> Iterator[Block]:
    if block_size <= 0:
        raise ValueError("block_size must be greater than 0")

    block_id = 0
    offset = 0
    buffer = bytearray()

    for chunk in data_iter:
        if not chunk:
            continue

        buffer.extend(chunk)

        while len(buffer) >= block_size:
            block_data = bytes(buffer[:block_size])
            del buffer[:block_size]

            yield Block(
                block_id=block_id,
                data=block_data,
                size=len(block_data),
                offset=offset,
            )
            block_id += 1
            offset += len(block_data)

    if buffer:
        block_data = bytes(buffer)
        yield Block(
            block_id=block_id,
            data=block_data,
            size=len(block_data),
            offset=offset,
        )
