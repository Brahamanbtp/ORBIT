from core.block import Block
from orbit_codecs import CODEC_REGISTRY
from bandit.reward import compute_reward  


def compute_oracle_stats(blocks: list[Block], codec_registry: dict) -> dict:
    """
    Compute oracle codec selection statistics for a list of blocks.

    Returns:
        dict with keys:
            - oracle_actions: list[int]
            - per_block_best_ratio: list[float]
            - mean_oracle_ratio: float
            - codec_selection_counts: dict[str, int]
            - oracle_rewards: list[float]  
    """
    oracle_actions = compute_oracle_actions(blocks, codec_registry)

    per_block_best_ratio = []
    codec_selection_counts = {}
    oracle_rewards = []  

    for block, codec_id in zip(blocks, oracle_actions):
        original_size = len(block.data)

        # compress using best codec (oracle choice)
        compressed = codec_registry[codec_id].compress(block.data)
        compressed_size = len(compressed)

        # ratio
        ratio = compressed_size / original_size if original_size > 0 else 0.0
        per_block_best_ratio.append(ratio)

        # codec usage stats
        name = type(codec_registry[codec_id]).__name__
        codec_selection_counts[name] = codec_selection_counts.get(name, 0) + 1

        # ✅ compute reward (elapsed_ms unknown → assume 0.0 or handled inside function)
        reward = compute_reward(original_size, compressed_size, 0.0)
        oracle_rewards.append(reward)

    mean_oracle_ratio = (
        sum(per_block_best_ratio) / len(per_block_best_ratio)
        if per_block_best_ratio else 0.0
    )

    return {
        "oracle_actions": oracle_actions,
        "per_block_best_ratio": per_block_best_ratio,
        "mean_oracle_ratio": mean_oracle_ratio,
        "codec_selection_counts": codec_selection_counts,
        "oracle_rewards": oracle_rewards,  
    }


def compute_oracle_rewards(blocks: list[Block], codec_registry: dict) -> list[float]:
    """
    Compute only oracle rewards per block.

    For each block:
        - Try all codecs
        - Pick codec with minimum compressed size
        - Compute reward using compute_reward

    Returns:
        List of rewards (float), one per block
    """
    rewards = []

    for block in blocks:
        original_size = len(block.data)

        best_size = None

        for codec in codec_registry.values():
            compressed = codec.compress(block.data)
            size = len(compressed)

            if best_size is None or size < best_size:
                best_size = size

        # compute reward for best compression
        reward = compute_reward(original_size, best_size, 0.0)
        rewards.append(reward)

    return rewards


from core.block import Block


def compute_oracle_actions(blocks: list[Block], codec_registry: dict) -> list[int]:
    """
    For each block, compress with every codec in registry, record compressed size,
    and return the codec_id that achieved minimum compressed size.

    Args:
        blocks: List of Block objects (must have .data attribute)
        codec_registry: Dict mapping codec_id (int) to codec instance (must have .compress method)

    Returns:
        List of codec_id (int) for each block, corresponding to the best codec.
    """
    assert all(b.block_id == i for i, b in enumerate(blocks)), \
        "Blocks must be in order and contiguous by block_id."

    oracle_actions = []

    for block in blocks:
        best_codec_id = None
        best_size = None

        for codec_id, codec in codec_registry.items():
            compressed = codec.compress(block.data)
            size = len(compressed)

            if best_size is None or size < best_size:
                best_size = size
                best_codec_id = codec_id

        oracle_actions.append(best_codec_id)

    return oracle_actions