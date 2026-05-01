"""Generate the three evaluation corpora used for ORBIT experiments.

This script creates a text corpus, a binary corpus, and a mixed corpus for
block-level compression evaluation.
"""

import os
import random
import struct


TEXT_TARGET_BYTES = 10 * 1024 * 1024
BINARY_TARGET_BYTES = 10 * 1024 * 1024
MIXED_TARGET_BYTES = 20 * 1024 * 1024
CHUNK_SIZE = 64 * 1024

VOCABULARY = [
    "adaptive",
    "algorithm",
    "analysis",
    "archive",
    "bandit",
    "binary",
    "block",
    "buffer",
    "cache",
    "codec",
    "compression",
    "compute",
    "context",
    "corpus",
    "data",
    "dataset",
    "decision",
    "delta",
    "entropy",
    "experiment",
    "feature",
    "flow",
    "format",
    "gateway",
    "guided",
    "heuristic",
    "history",
    "index",
    "input",
    "iterate",
    "kernel",
    "latency",
    "learn",
    "metric",
    "model",
    "monitor",
    "noise",
    "online",
    "oracle",
    "output",
    "packet",
    "pattern",
    "predict",
    "profile",
    "policy",
    "probe",
    "quality",
    "random",
    "rate",
    "ratio",
    "reward",
    "routing",
    "sample",
    "search",
    "signal",
    "slope",
    "split",
    "state",
    "stream",
    "structure",
    "system",
    "text",
    "throughput",
    "token",
    "trace",
    "train",
    "update",
    "vector",
    "window",
    "write",
    "zero",
    "accuracy",
    "baseline",
    "benchmark",
    "bitstream",
    "blockwise",
    "branch",
    "checkpoint",
    "cluster",
    "compress",
    "controller",
    "decode",
    "encode",
    "evaluation",
    "featureless",
    "gradient",
    "hash",
    "implementation",
    "inspection",
    "integrate",
    "iteration",
    "learning",
    "manifest",
    "memory",
    "mixed",
    "normalization",
    "observation",
    "parameter",
    "prediction",
    "reproducible",
    "routing",
    "sampling",
    "selection",
    "sparse",
    "synthesis",
    "testing",
    "variance",
    "workflow",
    "zlib",
    "zstd",
    "lz4",
    "lzma",
    "raw",
    "streaming",
    "adaptive",
    "contextual",
    "compression",
    "regret",
    "reward",
    "exploration",
    "exploitation",
    "vectorization",
    "throughput",
    "stability",
    "robustness",
    "validation",
    "verification",
    "measurement",
    "correlation",
    "distribution",
    "entropy",
    "prediction",
    "logging",
    "analysis",
    "feature",
    "sequence",
    "transform",
    "tokenization",
    "boundary",
    "streamline",
    "controller",
    "policy",
    "reward",
    "oracle",
    "dataset",
    "corpus",
    "scientific",
    "research",
    "experiment",
    "artifact",
    "pipeline",
    "buffering",
    "routing",
    "selection",
    "contextual",
    "online",
    "learning",
    "adaptive",
    "compression",
    "codec",
    "block",
    "feature",
    "metric",
    "regret",
    "throughput",
    "evaluation",
    "dataset",
    "binary",
    "text",
    "mixed",
    "structured",
    "unstructured",
    "sample",
    "window",
    "rolling",
    "smoothing",
    "projection",
    "variance",
    "entropy",
    "selection",
    "decision",
    "routing",
    "learning",
    "reward",
    "explore",
    "exploit",
    "compress",
    "decompress",
    "serialize",
    "deserialize",
    "benchmark",
    "measure",
    "compare",
    "repeat",
    "report",
    "trace",
    "debug",
    "inspect",
    "validate",
    "confirm",
    "record",
    "analysis",
    "adaptive",
    "online",
    "context",
    "decision",
    "policy",
    "feature",
    "codec",
    "routing",
    "experiment",
    "evaluation",
    "compression",
    "throughput",
    "regret",
    "reward",
    "binary",
    "text",
    "mixed",
    "structure",
    "signal",
    "noise",
    "sample",
    "stream",
    "block",
    "chunk",
    "window",
    "sequence",
    "vector",
    "state",
    "model",
    "policy",
    "feature",
    "metric",
    "result",
    "output",
    "input",
    "pattern",
    "analysis",
    "training",
    "evaluation",
    "production",
    "research",
    "science",
    "compression",
    "codec",
    "routing",
    "bandit",
    "contextual",
    "linucb",
    "oracle",
    "baseline",
    "candidate",
    "selection",
    "feature",
    "entropy",
    "rle",
    "repetition",
    "convergence",
    "stability",
    "performance",
    "quality",
    "efficiency",
    "robust",
    "scalable",
    "deterministic",
    "stochastic",
    "sequence",
    "token",
    "corpus",
    "language",
    "natural",
    "structured",
    "binary",
    "mixed",
    "dataset",
    "experiment",
    "output",
    "manifest",
    "record",
    "logging",
    "trace",
    "inspection",
    "validation",
    "reproducible",
    "scientific",
    "results",
    "reporting",
    "comparison",
    "ablation",
    "sweep",
    "regret",
    "curve",
    "figure",
    "table",
    "paper",
    "benchmark",
    "workflow",
    "pipeline",
    "compute",
    "analysis",
    "signal",
    "model",
    "policy",
    "reward",
    "selection",
    "feature",
    "compression",
    "routing",
    "block",
    "online",
    "adaptive",
    "codec",
    "decision",
    "learning",
    "exploration",
    "exploitation",
    "search",
    "measure",
    "compare",
    "repeat",
    "sample",
    "stable",
    "robust",
    "fast",
    "accurate",
    "efficient",
    "incremental",
    "guided",
    "context",
    "bandit",
    "lz4",
    "zstd",
    "lzma",
    "raw",
    "codec",
    "data",
    "text",
    "binary",
    "mixed",
    "compression",
    "experiment",
    "evaluation",
    "result",
    "report",
    "stream",
    "block",
    "window",
    "analysis",
    "selection",
    "feature",
    "reward",
    "regret",
    "throughput",
    "stability",
    "accuracy",
    "quality",
    "sample",
    "distribution",
    "measurement",
    "validation",
    "verification",
    "logging",
    "debug",
    "trace",
    "manifest",
    "corpus",
    "dataset",
    "science",
    "research",
    "paper",
    "table",
    "figure",
    "result",
    "summary",
    "adaptive",
    "contextual",
    "bandit",
    "guided",
    "routing",
    "codec",
    "compression",
    "online",
    "incremental",
    "natural",
    "language",
    "structured",
    "binary",
    "mixed",
    "feature",
    "decision",
    "policy",
    "experiment",
    "benchmark",
    "evaluation",
    "analysis",
    "output",
    "input",
    "dataset",
]

VOCABULARY = list(dict.fromkeys(VOCABULARY))
if len(VOCABULARY) < 500:
    VOCABULARY.extend(f"token{i:03d}" for i in range(500 - len(VOCABULARY)))
else:
    VOCABULARY = VOCABULARY[:500]


def ensure_data_dir() -> None:
    os.makedirs("data", exist_ok=True)


def build_text_corpus(target_bytes: int) -> bytes:
    rng = random.Random(42)
    chunks: list[str] = []
    total_bytes = 0
    line_index = 0

    while total_bytes < target_bytes:
        if line_index % 2 == 0:
            word_count = 14 + (line_index % 9)
            words = [rng.choice(VOCABULARY) for _ in range(word_count)]
            line = " ".join(words).capitalize() + f". Observation {line_index:07d}."
        else:
            level = rng.choice(["INFO", "DEBUG", "WARN", "TRACE"])
            module = rng.choice(["router", "compressor", "feature", "policy", "runner"])
            block_id = line_index * 37
            score = rng.random() * 10.0
            ratio = 0.5 + rng.random() * 1.5
            line = (
                f"{level} module={module} block={block_id} "
                f"score={score:.6f} ratio={ratio:.6f} "
                f"event={rng.choice(VOCABULARY)}"
            )
        chunks.append(line + "\n")
        total_bytes += len(chunks[-1].encode("utf-8"))
        line_index += 1

    data = "".join(chunks).encode("utf-8")
    return data[:target_bytes]


def build_binary_corpus(target_bytes: int) -> bytes:
    out = bytearray()
    use_random = True
    pattern = bytes([i % 256 for i in range(CHUNK_SIZE)])

    while len(out) < target_bytes:
        if use_random:
            chunk = os.urandom(CHUNK_SIZE)
        else:
            chunk = pattern
        out.extend(chunk)
        use_random = not use_random

    return bytes(out[:target_bytes])


def build_mixed_corpus(text_bytes: bytes, binary_bytes: bytes, target_bytes: int) -> bytes:
    out = bytearray()
    use_text = True
    max_chunks = target_bytes // CHUNK_SIZE

    for index in range(max_chunks):
        start = (index * CHUNK_SIZE) % len(text_bytes)
        text_chunk = text_bytes[start : start + CHUNK_SIZE]
        if len(text_chunk) < CHUNK_SIZE:
            text_chunk = text_chunk + text_bytes[: CHUNK_SIZE - len(text_chunk)]

        start = (index * CHUNK_SIZE) % len(binary_bytes)
        binary_chunk = binary_bytes[start : start + CHUNK_SIZE]
        if len(binary_chunk) < CHUNK_SIZE:
            binary_chunk = binary_chunk + binary_bytes[: CHUNK_SIZE - len(binary_chunk)]

        out.extend(text_chunk if use_text else binary_chunk)
        use_text = not use_text

    return bytes(out[:target_bytes])


def write_file(path: str, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)


def main() -> None:
    ensure_data_dir()

    text_data = build_text_corpus(TEXT_TARGET_BYTES)
    binary_data = build_binary_corpus(BINARY_TARGET_BYTES)
    mixed_data = build_mixed_corpus(text_data, binary_data, MIXED_TARGET_BYTES)

    write_file(os.path.join("data", "text_corpus.txt"), text_data)
    write_file(os.path.join("data", "binary_corpus.bin"), binary_data)
    write_file(os.path.join("data", "mixed_corpus.bin"), mixed_data)

    print(f"data/text_corpus.txt: {len(text_data)} bytes")
    print(f"data/binary_corpus.bin: {len(binary_data)} bytes")
    print(f"data/mixed_corpus.bin: {len(mixed_data)} bytes")


if __name__ == "__main__":
    main()
