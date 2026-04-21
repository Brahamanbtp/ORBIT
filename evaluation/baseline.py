from core.processor import split_into_blocks
from core.block import Block
def run_baseline_blockwise(input_path: str, codec, block_size: int) -> dict:
    import time
    start = time.time()
    total_original = 0
    total_compressed = 0
    per_block_ratios = []
    codec_name = getattr(codec, "__class__", type(codec)).__name__
    with open(input_path, "rb") as f:
        def data_iter():
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break
                yield chunk
        blocks = list(split_into_blocks(data_iter(), block_size))
    for block in blocks:
        compressed = codec.compress(block.data)
        total_original += len(block.data)
        total_compressed += len(compressed)
        ratio = len(compressed) / len(block.data) if len(block.data) > 0 else 0.0
        per_block_ratios.append(ratio)
    elapsed_ms = (time.time() - start) * 1000
    compression_ratio = total_compressed / total_original if total_original > 0 else 0.0
    return {
        "codec_name": codec_name,
        "total_original": total_original,
        "total_compressed": total_compressed,
        "compression_ratio": compression_ratio,
        "per_block_ratios": per_block_ratios,
        "elapsed_ms": elapsed_ms,
    }
import time

DEFAULT_BLOCK_SIZE = 4096

def run_baseline(input_path: str, codec) -> dict:
    start = time.time()
    with open(input_path, "rb") as f:
        data = f.read()
    total_original = len(data)
    blocks = [data[i:i+DEFAULT_BLOCK_SIZE] for i in range(0, len(data), DEFAULT_BLOCK_SIZE)]
    compressed_blocks = [codec.compress(block) for block in blocks]
    total_compressed = sum(len(cb) for cb in compressed_blocks)
    elapsed_ms = (time.time() - start) * 1000
    codec_name = getattr(codec, "__class__", type(codec)).__name__
    compression_ratio = total_compressed / total_original if total_original > 0 else 0.0
    return {
        "codec_name": codec_name,
        "total_original": total_original,
        "total_compressed": total_compressed,
        "compression_ratio": compression_ratio,
        "elapsed_ms": elapsed_ms,
    }
