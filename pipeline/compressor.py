from orbit_codecs import get_codec
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




def process_block(block, router, policy, writer, timing_acc=None) -> dict:
    from utils.timing import TimingContext

    # Feature extraction timing
    with TimingContext("feature_extraction") as t_feat:
        features = router.extractor.extract(block)
    feature_extraction_ms = t_feat.elapsed_ms
    if timing_acc is not None:
        timing_acc.record(block.block_id, "feature_extraction", feature_extraction_ms)

    # Bandit decision timing
    with TimingContext("bandit_decision") as t_bandit:
        action_id = policy.select_action(features)
    bandit_decision_ms = t_bandit.elapsed_ms
    if timing_acc is not None:
        timing_acc.record(block.block_id, "bandit_decision", bandit_decision_ms)

    # Compression timing
    with TimingContext("compression") as t_comp:
        compressed_bytes, _ = compress_block(block, action_id)
    compression_ms = t_comp.elapsed_ms
    if timing_acc is not None:
        timing_acc.record(block.block_id, "compression", compression_ms)

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
        import json
        from io.format import write_file_header
        from orbit_codecs import available_codecs
        import importlib

        reader = StreamingReader(input_path, self.config.block_size)
        router = BlockRouter(self.extractor, self.policy, self.action_space)
        writer = BinaryWriter(output_path)

        # Prepare codec_versions dict
        codec_versions = {}
        for name in available_codecs():
            try:
                if name.lower().startswith("lz4"):
                    mod = importlib.import_module("lz4")
                elif name.lower().startswith("zstd"):
                    mod = importlib.import_module("zstandard")
                elif name.lower().startswith("lzma"):
                    import sys
                    mod = None
                else:
                    mod = None
                if mod is not None:
                    version = getattr(mod, "__version__", "unknown")
                elif name.lower().startswith("lzma"):
                    import sys
                    version = sys.version.split()[0]
                else:
                    version = "unknown"
            except Exception:
                version = "unknown"
            codec_versions[name] = version
        codec_versions_json = json.dumps(codec_versions).encode("utf-8")
        if len(codec_versions_json) > 32:
            codec_versions_json = codec_versions_json[:32]
        elif len(codec_versions_json) < 32:
            codec_versions_json = codec_versions_json.ljust(32, b" ")

        # Write file header (assume writer has a file handle f)
        # You may need to adapt this if header writing is elsewhere
        n_blocks = 0
        block_size = self.config.block_size
        codec_registry_checksum = 0
        from orbit_codecs import CODEC_REGISTRY
        for k in CODEC_REGISTRY.keys():
            codec_registry_checksum ^= k
        # Write header at start
        with open(output_path, "wb") as f:
            write_file_header(f, n_blocks, block_size, codec_registry_checksum, codec_versions_json)
            # Now write blocks
            results = []
            for block in split_into_blocks(reader, self.config.block_size):
                result = process_block(block, router, self.policy, writer)
                results.append(result)
            # Optionally, update n_blocks in header if needed (not shown)
        return results
