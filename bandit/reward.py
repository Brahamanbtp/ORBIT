COMPRESSION_GAIN_WEIGHT = 0.8
SPEED_PENALTY_WEIGHT = 0.2


def compute_reward(original_size: int, compressed_size: int, compression_time_ms: float) -> float:
    if original_size <= 0:
        return 0.0

    compression_ratio_gain = 1.0 - (compressed_size / original_size)
    if compression_ratio_gain <= 0.0:
        return -0.1

    safe_time_ms = max(0.0, compression_time_ms)
    speed_score = 1.0 / (1.0 + safe_time_ms)
    reward = (COMPRESSION_GAIN_WEIGHT * compression_ratio_gain) + (SPEED_PENALTY_WEIGHT * speed_score)
    return float(reward)
