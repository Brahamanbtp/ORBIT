COMPRESSION_GAIN_WEIGHT = 0.95
SPEED_PENALTY_WEIGHT = 0.05


def compute_reward(original_size: int, compressed_size: int, compression_time_ms: float) -> float:
    if original_size <= 0:
        return 0.0

    if compressed_size >= original_size:
        # Penalize expansion proportionally; Raw (equal sizes) yields zero ratio gain.
        compression_ratio_gain = (original_size / compressed_size) - 1.0
    else:
        compression_ratio_gain = 1.0 - (compressed_size / original_size)

    # Cap compression time to avoid excessive penalty from very slow codecs
    safe_time_ms = max(0.0, compression_time_ms)
    safe_time_ms = min(safe_time_ms, 100.0)
    speed_score = 1.0 / (1.0 + safe_time_ms)
    reward = (COMPRESSION_GAIN_WEIGHT * compression_ratio_gain) + (SPEED_PENALTY_WEIGHT * speed_score)
    return float(reward)
