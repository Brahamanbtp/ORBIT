from __future__ import annotations

import importlib
import json

from orbit_codecs import get_codec, available_codecs, CODEC_REGISTRY
from utils.timing import measure_time_ms, TimingContext
from bandit.reward import compute_reward
from configs.schema import ORBITConfig
from core.interfaces import BanditPolicy
from bandit.action_space import ActionSpace
from orbit_io.reader import StreamingReader
from orbit_io.writer import BinaryWriter
from core.processor import split_into_blocks
from pipeline.router import BlockRouter


# ------------------------------------------------------------------ #
#  Block-level compression                                             #
# ------------------------------------------------------------------ #

def compress_block(block, codec_id: int) -> tuple[bytes, float]:
    codec = get_codec(codec_id)
    compressed_bytes, elapsed_ms = measure_time_ms(codec.compress, block.data)
    return compressed_bytes, elapsed_ms


# ------------------------------------------------------------------ #
#  Single block processing (route → compress → reward → log)          #
# ------------------------------------------------------------------ #

def process_block(block, router, policy, writer, timing_acc=None) -> dict:
    # ------------------------------------------------------------------ #
    # 1. Feature extraction (router.route equivalent split in your design)
    # ------------------------------------------------------------------ #
    with TimingContext("feature_extraction") as t_feat:
        features = router.extractor.extract(block)

    if timing_acc is not None:
        timing_acc.record(block.block_id, "feature_extraction", t_feat.elapsed_ms)

    # ------------------------------------------------------------------ #
    # 2. Bandit decision (select_action with block_id)
    # ------------------------------------------------------------------ #
    with TimingContext("bandit_decision") as t_bandit:
        action_id = router.policy.select_action(features, block_id=block.block_id)

    if timing_acc is not None:
        timing_acc.record(block.block_id, "bandit_decision", t_bandit.elapsed_ms)

    # ------------------------------------------------------------------ #
    # 3. Compression
    # ------------------------------------------------------------------ #
    with TimingContext("compression") as t_comp:
        compressed_bytes, _ = compress_block(block, action_id)

    if timing_acc is not None:
        timing_acc.record(block.block_id, "compression", t_comp.elapsed_ms)

    # ------------------------------------------------------------------ #
    # 4. Compute reward
    # ------------------------------------------------------------------ #
    reward = compute_reward(
        len(block.data),
        len(compressed_bytes),
        t_comp.elapsed_ms
    )

    # ------------------------------------------------------------------ #
    # 5. Policy update WITH block_id
    # ------------------------------------------------------------------ #
    router.policy.update(
        features,
        action_id,
        reward,
        block_id=block.block_id
    )

    # ------------------------------------------------------------------ #
    # 6. Logger consistency check
    # ------------------------------------------------------------------ #
    if hasattr(router.policy, "log"):
        assert router.policy.log[-1]["block_id"] == block.block_id, \
            f"Logger block_id mismatch: expected {block.block_id}, got {router.policy.log[-1]['block_id']}"

    # ------------------------------------------------------------------ #
    # 7. DEBUG: first block sanity check
    # ------------------------------------------------------------------ #
    if block.block_id == 0 and hasattr(router.policy, "log"):
        assert len(router.policy.log) > 0, "Logger is empty after first block"

        first_entry = router.policy.log[0]
        print("DEBUG FIRST LOG ENTRY:", first_entry)

        assert float(first_entry.get("reward", 0.0)) != 0.0, \
            "First reward should not be zero"

    # ------------------------------------------------------------------ #
    # 8. Write output
    # ------------------------------------------------------------------ #
    writer.write_block(
        compressed_bytes,
        action_id,
        block.block_id,
        original_size=len(block.data)
    )

    # ------------------------------------------------------------------ #
    # 9. Return metrics
    # ------------------------------------------------------------------ #
    return {
        "block_id": block.block_id,
        "action_id": action_id,
        "reward": float(reward),
        "original_size": len(block.data),
        "compressed_size": len(compressed_bytes),
        "feature_extraction_ms": t_feat.elapsed_ms,
        "bandit_decision_ms": t_bandit.elapsed_ms,
        "compression_ms": t_comp.elapsed_ms,
    }


# ------------------------------------------------------------------ #
#  Codec version helper                                                #
# ------------------------------------------------------------------ #

def _build_codec_versions() -> bytes:
    codec_versions: dict[str, str] = {}
    for name in available_codecs():
        try:
            name_lower = name.lower()
            if "lz4" in name_lower:
                mod = importlib.import_module("lz4")
                version = getattr(mod, "__version__", "unknown")
            elif "zstd" in name_lower:
                mod = importlib.import_module("zstandard")
                version = getattr(mod, "__version__", "unknown")
            elif "lzma" in name_lower:
                import sys
                version = sys.version.split()[0]
            else:
                version = "unknown"
        except Exception:
            version = "unknown"
        codec_versions[name] = version

    raw = json.dumps(codec_versions).encode("utf-8")
    # Pad or truncate to exactly 32 bytes for header field
    return raw[:32].ljust(32, b" ")


# ------------------------------------------------------------------ #
#  Compressor orchestrator                                             #
# ------------------------------------------------------------------ #

class ORBITCompressor:
    def __init__(
        self,
        config: ORBITConfig,
        extractor,
        policy: BanditPolicy,
        action_space: ActionSpace,
    ) -> None:
        self.config = config
        self.extractor = extractor
        self.policy = policy
        self.action_space = action_space

    def compress_file(self, input_path: str, output_path: str) -> list[dict]:
        # Compute registry checksum
        codec_registry_checksum = 0
        for k in CODEC_REGISTRY.keys():
            codec_registry_checksum ^= k

        # Build codec versions bytes for header
        codec_versions_bytes = _build_codec_versions()

        # Collect all blocks first so we know n_blocks for the header
        reader = StreamingReader(input_path, self.config.block_size)
        blocks = list(split_into_blocks(reader, self.config.block_size))
        n_blocks = len(blocks)

        # Use BinaryWriter's own file management — never open file separately
        router = BlockRouter(self.extractor, self.policy, self.action_space)
        writer = BinaryWriter(output_path)
        writer.open_file(n_blocks, self.config.block_size)

        results: list[dict] = []
        try:
            for block in blocks:
                result = process_block(block, router, self.policy, writer)
                results.append(result)
        finally:
            # Always close — flushes and fsyncs even if an exception occurs
            writer.close_file()

        return results