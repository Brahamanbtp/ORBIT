COMPRESSION_GAIN_WEIGHT = 0.8
SPEED_PENALTY_WEIGHT = 0.2


def compute_reward(original_size: int, compressed_size: int, compression_time_ms: float) -> float:
    if original_size <= 0:
        return 0.0

    if compressed_size > original_size:
        compression_ratio_gain = 0.0
    elif compressed_size == original_size:
        compression_ratio_gain = 0.1
    else:
        compression_ratio_gain = max(0.0, (original_size - compressed_size) / original_size)
    speed_penalty = max(0.0, compression_time_ms) / (max(1.0, compression_time_ms) + 1.0)

    reward = (COMPRESSION_GAIN_WEIGHT * compression_ratio_gain) - (SPEED_PENALTY_WEIGHT * speed_penalty)
    return float(reward)
