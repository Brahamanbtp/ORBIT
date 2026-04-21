from codecs import get_codec
from utils.timing import measure_time_ms
from bandit.reward import compute_reward

from configs.schema import ORBITConfig
from features.extractor import FeatureExtractor
from bandit.action_space import ActionSpace
from core.interfaces import BanditPolicy
from io.reader import StreamingReader
from core.processor import split_into_blocks
from pipeline.router import BlockRouter
from io.writer import BinaryWriter


def compress_block(block, codec_id: int) -> tuple[bytes, float]:
    codec = get_codec(codec_id)
    def _compress():
        return codec.compress(block.data)
    compressed_bytes, elapsed_ms = measure_time_ms(_compress)
    return compressed_bytes, elapsed_ms



def process_block(block, router, policy, writer) -> dict:
    from utils.timing import TimingContext

    # Feature extraction timing
    with TimingContext("feature_extraction") as t_feat:
        # router.route calls extractor.extract and policy.select_action
        # We'll split timing for feature extraction and bandit decision
        features = router.extractor.extract(block)
    feature_extraction_ms = t_feat.elapsed_ms

    # Bandit decision timing
    with TimingContext("bandit_decision") as t_bandit:
        action_id = policy.select_action(features)
    bandit_decision_ms = t_bandit.elapsed_ms

    # Compression timing
    with TimingContext("compression") as t_comp:
        compressed_bytes, _ = compress_block(block, action_id)
    compression_ms = t_comp.elapsed_ms

    reward = compute_reward(len(block.data), len(compressed_bytes), compression_ms)
    policy.update(features, action_id, reward)
    writer.write_block(compressed_bytes, action_id, block.block_id)
    return {
        "block_id": block.block_id,
        "action_id": action_id,
        "reward": reward,
        "original_size": len(block.data),
        "compressed_size": len(compressed_bytes),
        "feature_extraction_ms": feature_extraction_ms,
        "bandit_decision_ms": bandit_decision_ms,
        "compression_ms": compression_ms,
    }


class ORBITCompressor:
    def __init__(self, config: ORBITConfig, extractor: FeatureExtractor, policy: BanditPolicy, action_space: ActionSpace):
        self.config = config
        self.extractor = extractor
        self.policy = policy
        self.action_space = action_space

    def compress_file(self, input_path: str, output_path: str) -> list[dict]:
        reader = StreamingReader(input_path, self.config.block_size)
        router = BlockRouter(self.extractor, self.policy, self.action_space)
        writer = BinaryWriter(output_path)
        results = []
        for block in split_into_blocks(reader, self.config.block_size):
            result = process_block(block, router, self.policy, writer)
            results.append(result)
        return results
