def compression_ratio(original: int, compressed: int) -> float:
    if original == 0:
        return 0.0
    return compressed / original


def space_saving(original: int, compressed: int) -> float:
    if original == 0:
        return 0.0
    return (original - compressed) / original


def throughput_mbps(n_bytes: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return 0.0
    return (n_bytes * 8) / (elapsed_ms * 1000)


def overhead_ratio(feature_time_ms: float, bandit_time_ms: float, compress_time_ms: float) -> float:
    total = feature_time_ms + bandit_time_ms + compress_time_ms
    if total == 0:
        return 0.0
    return (feature_time_ms + bandit_time_ms) / total

def aggregate_block_results(results: list[dict]) -> dict:
    if not results:
        return {
            "mean_compression_ratio": 0.0,
            "codec_selection_counts": {},
            "mean_reward": 0.0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
        }
    total_original = sum(r["original_size"] for r in results)
    total_compressed = sum(r["compressed_size"] for r in results)
    mean_compression_ratio = (
        sum(r["compressed_size"] / r["original_size"] for r in results if r["original_size"] > 0) / len(results)
    )
    codec_selection_counts = {}
    for r in results:
        cid = r["action_id"]
        codec_selection_counts[cid] = codec_selection_counts.get(cid, 0) + 1
    mean_reward = sum(r["reward"] for r in results) / len(results)
    return {
        "mean_compression_ratio": mean_compression_ratio,
        "codec_selection_counts": codec_selection_counts,
        "mean_reward": mean_reward,
        "total_original_bytes": total_original,
        "total_compressed_bytes": total_compressed,
    }
