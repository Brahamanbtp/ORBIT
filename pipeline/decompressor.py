from codecs import get_codec
from io.format import ORBIT_MAGIC, read_file_header, read_block_header




    @staticmethod
    def _validate_block_order(headers):
        block_ids = [h.block_id for h in headers]
        expected = list(range(len(headers)))
        if block_ids != expected:
            raise ValueError(f"Block IDs are not contiguous from 0: {block_ids}")
        return True

    def decompress_file(self, input_path: str, output_path: str) -> None:
        blocks = {}
        headers = []
        from io.format import BlockHeader  # ensure import for type
        with open(input_path, "rb") as infile:
            file_header = read_file_header(infile)
            if file_header.magic != ORBIT_MAGIC:
                raise ValueError(f"Invalid ORBIT magic bytes: {file_header.magic!r}")
            for _ in range(file_header.n_blocks):
                block_header = read_block_header(infile)
                headers.append(block_header)
                compressed_data = infile.read(block_header.compressed_size)
                codec = get_codec(block_header.codec_id)
                data = codec.decompress(compressed_data)
                blocks[block_header.block_id] = data
        self._validate_block_order(headers)
        with open(output_path, "wb") as outfile:
            for block_id in sorted(blocks):
                outfile.write(blocks[block_id])
