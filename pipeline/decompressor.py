from orbit_codecs import get_codec
from orbit_io.format import ORBIT_MAGIC, read_file_header, read_block_header




    @staticmethod
    def _validate_block_order(headers):
        block_ids = [h.block_id for h in headers]
        expected = list(range(len(headers)))
        if block_ids != expected:
            raise ValueError(f"Block IDs are not contiguous from 0: {block_ids}")
        return True

    def decompress_file(self, input_path: str, output_path: str) -> None:
        headers = []
        block_data_pairs = []
        from io.format import BlockHeader  # ensure import for type
        from orbit_codecs import CODEC_REGISTRY
        with open(input_path, "rb") as infile:
            file_header = read_file_header(infile)
            if file_header.magic != ORBIT_MAGIC:
                raise ValueError(f"Invalid ORBIT magic bytes: {file_header.magic!r}")
            # Validate codec_registry_checksum
            current_checksum = 0
            for k in CODEC_REGISTRY.keys():
                current_checksum ^= k
            if getattr(file_header, "codec_registry_checksum", None) is not None:
                if file_header.codec_registry_checksum != current_checksum:
                    raise ValueError(
                        f"Codec registry checksum mismatch: file={file_header.codec_registry_checksum}, current={current_checksum}. "
                        f"This indicates a codec registry change or incompatibility."
                    )
            # Log codec_versions if present
            import json, warnings
            codec_versions_bytes = getattr(file_header, "codec_versions", None)
            if codec_versions_bytes:
                try:
                    s = codec_versions_bytes.rstrip(b" ")
                    codec_versions = json.loads(s.decode("utf-8"))
                    warnings.warn(f"File codec_versions: {codec_versions}")
                except Exception:
                    warnings.warn(f"Could not decode codec_versions field: {codec_versions_bytes}")
            # Use block_size from header for validation only
            header_block_size = getattr(file_header, "block_size", None)
            for _ in range(file_header.n_blocks):
                block_header = read_block_header(infile)
                if block_header.compressed_size <= 0:
                    raise AssertionError(f"BlockHeader for block_id={block_header.block_id} has non-positive compressed_size: {block_header.compressed_size}")
                headers.append(block_header)
                compressed_data = infile.read(block_header.compressed_size)
                block_data_pairs.append((block_header, compressed_data))
            # Optionally, validate block sizes (internal check, not enforced)
            if header_block_size is not None:
                for bh in headers:
                    if bh.original_size > header_block_size:
                        raise ValueError(f"BlockHeader original_size {bh.original_size} exceeds header block_size {header_block_size} for block_id={bh.block_id}")
        self._validate_block_order(headers)
        block_data_pairs.sort(key=lambda pair: pair[0].block_id)
        with open(output_path, "wb") as outfile:
            for block_header, compressed_data in block_data_pairs:
                codec = get_codec(block_header.codec_id)
                data = codec.decompress(compressed_data)
                outfile.write(data)
