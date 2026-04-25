def compute_overhead_breakdown_from_accumulator(timing_records: list[dict]) -> dict:
    """
    Accepts output from TimingAccumulator.to_dataframe().
    Computes per-phase mean, std, percent of total time, and dominant_phase.
    """
    import numpy as np
    if not timing_records:
        return {}
    # Group by label
    phase_times = {}
    for rec in timing_records:
        label = rec["label"]
        phase_times.setdefault(label, []).append(rec["elapsed_ms"])
    # Compute mean, std per phase
    stats = {}
    total_time = sum(sum(times) for times in phase_times.values())
    for label, times in phase_times.items():
        arr = np.array(times, dtype=float)
        stats[label] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "total": float(np.sum(arr)),
            "percent": float(np.sum(arr)) / total_time * 100 if total_time > 0 else 0.0,
        }
    # Find dominant phase
    dominant_phase = max(stats.items(), key=lambda x: x[1]["total"])[0] if stats else None
    stats["dominant_phase"] = dominant_phase
    return stats


def validate_comparison_record(record: dict) -> list[str]:
    """
    Validate required schema fields for a core comparison result record.
    Returns a list of missing key names; empty list means valid.
    """
    required_keys = [
        "run_id",
        "dataset_name",
        "method_name",
        "compression_ratio",
        "throughput_mbps",
        "overhead_ratio",
        "block_size",
        "n_runs",
        "seed",
    ]
    return [key for key in required_keys if key not in record]


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


def plot_codec_distribution(results: list[dict], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install with 'pip install matplotlib'.") from exc

    # Count codec selections
    codec_selection_counts = {}
    for r in results:
        cid = r["action_id"]
        codec_selection_counts[cid] = codec_selection_counts.get(cid, 0) + 1

    if not codec_selection_counts:
        raise ValueError("No codec selection data to plot.")

    labels = list(codec_selection_counts.keys())
    counts = [codec_selection_counts[k] for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color="skyblue")
    plt.xlabel("Codec ID")
    plt.ylabel("Selection Count")
    plt.title("Codec Selection Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compute_overhead_breakdown(timing_log: list[dict]) -> dict:
    import numpy as np
    feature = np.array([d["feature_ms"] for d in timing_log], dtype=float)
    bandit = np.array([d["bandit_ms"] for d in timing_log], dtype=float)
    compress = np.array([d["compress_ms"] for d in timing_log], dtype=float)
    total = feature + bandit + compress
    overhead_ratio = np.where(total > 0, (feature + bandit) / total, 0.0)
    return {
        "feature_mean": float(np.mean(feature)),
        "feature_std": float(np.std(feature)),
        "bandit_mean": float(np.mean(bandit)),
        "bandit_std": float(np.std(bandit)),
        "compress_mean": float(np.mean(compress)),
        "compress_std": float(np.std(compress)),
        "overhead_ratio": overhead_ratio.tolist(),
    }


def estimate_convergence_block(
    regret_curve: list[float],
    window: int = 50,
    threshold: float = 0.001,
    skip_blocks: int = 20,
) -> int:
    """
    Estimate convergence block as the first index where rolling slope drops below threshold.
    Returns -1 if convergence is not detected.
    """
    import numpy as np

    if not regret_curve:
        return -1

    if skip_blocks < 0:
        skip_blocks = 0

    if skip_blocks >= len(regret_curve):
        return -1

    curve = regret_curve[skip_blocks:]

    if window <= 1 or len(regret_curve) < window:
        return -1

    y = np.array(curve, dtype=float)
    x = np.arange(window, dtype=float)

    for start in range(0, len(y) - window + 1):
        segment = y[start : start + window]
        slope = float(np.polyfit(x, segment, 1)[0])
        if slope < threshold:
            return skip_blocks + start + window - 1

    return -1
