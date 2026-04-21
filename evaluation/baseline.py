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
