import struct
from codecs import get_codec


class ORBITDecompressor:
    HEADER_FORMAT = "<IHIH"  # block_id:4B, codec_id:2B, length:4B, reserved:2B
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    def decompress_file(self, input_path: str, output_path: str) -> None:
        blocks = {}
        with open(input_path, "rb") as infile:
            while True:
                header = infile.read(self.HEADER_SIZE)
                if not header or len(header) < self.HEADER_SIZE:
                    break
                block_id, codec_id, length, _ = struct.unpack(self.HEADER_FORMAT, header)
                compressed_data = infile.read(length)
                codec = get_codec(codec_id)
                data = codec.decompress(compressed_data)
                blocks[block_id] = data
        with open(output_path, "wb") as outfile:
            for block_id in sorted(blocks):
                outfile.write(blocks[block_id])
