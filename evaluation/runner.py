from __future__ import annotations

import numpy as np
def run_repeated_experiment(input_path: str, config: ORBITConfig, output_dir: str, n_runs: int = 5) -> dict:
    """
    Run run_experiment n_runs times, varying the random seed, and aggregate results.
    Computes mean and std for compression_ratio, mean_reward, and regret_curve.
    Saves results as repeated_results.json in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    for run_idx in range(n_runs):
        run_config = ORBITConfig(
            **{**config.__dict__, "random_seed": (config.random_seed or 0) + run_idx}
        )
        np.random.seed(run_config.random_seed)
        result = run_experiment(input_path, run_config, output_dir)
        all_results.append(result)

    # Aggregate metrics
    def collect_metric(results, key_path):
        vals = []
        for res in results:
            val = res
            for k in key_path:
                val = val.get(k, None)
                if val is None:
                    break
            if val is not None:
                vals.append(val)
        return vals

    compression_ratios = collect_metric(all_results, ["orbit_metrics", "compression_ratio"])
    mean_rewards = collect_metric(all_results, ["orbit_metrics", "mean_reward"])
    regret_curves = collect_metric(all_results, ["regret_curve"])

    # Compute mean/std for scalar metrics
    agg = {
        "compression_ratio_mean": float(np.mean(compression_ratios)) if compression_ratios else None,
        "compression_ratio_std": float(np.std(compression_ratios)) if compression_ratios else None,
        "mean_reward_mean": float(np.mean(mean_rewards)) if mean_rewards else None,
        "mean_reward_std": float(np.std(mean_rewards)) if mean_rewards else None,
    }

    # Compute mean/std for regret_curve (element-wise)
    if regret_curves:
        min_len = min(len(rc) for rc in regret_curves)
        regret_curves_trunc = [rc[:min_len] for rc in regret_curves]
        regret_curve_arr = np.array(regret_curves_trunc)
        agg["regret_curve_mean"] = regret_curve_arr.mean(axis=0).tolist()
        agg["regret_curve_std"] = regret_curve_arr.std(axis=0).tolist()
    else:
        agg["regret_curve_mean"] = []
        agg["regret_curve_std"] = []

    agg["all_results"] = all_results

    with open(os.path.join(output_dir, "repeated_results.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return agg
import os
import json
from codecs import CODEC_REGISTRY
from configs.schema import ORBITConfig
from pipeline.compressor import ORBITCompressor
from evaluation.baseline import run_baseline
from evaluation.metrics import aggregate_block_results

def run_experiment(input_path: str, config: ORBITConfig, output_dir: str) -> dict:
    from codecs import validate_all_codecs
    # Validate all codecs roundtrip before pipeline runs
    roundtrip_results = validate_all_codecs()
    failed = [name for name, ok in roundtrip_results.items() if not ok]
    if failed:
        raise RuntimeError(f"Codec roundtrip validation failed for: {failed}")
    from codecs import available_codecs, CODEC_REGISTRY
    # Assert codec registry matches config
    if len(available_codecs()) != config.n_actions:
        raise RuntimeError(f"Number of available codecs ({len(available_codecs())}) does not match config.n_actions ({config.n_actions}). Available: {available_codecs()}")
    if set(CODEC_REGISTRY.keys()) != set(range(config.n_actions)):
        raise RuntimeError(f"CODEC_REGISTRY keys {set(CODEC_REGISTRY.keys())} do not match expected set {set(range(config.n_actions))} for n_actions={config.n_actions}.")

    import random
    import sys
    import importlib
    import datetime
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    if getattr(config, "random_seed", None) is not None:
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    # --- Write reproducibility_manifest.json before pipeline runs ---
    # ORBIT version from pyproject.toml
    orbit_version = None
    try:
        import toml
        with open(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"), "r", encoding="utf-8") as f:
            pyproject = toml.load(f)
            orbit_version = pyproject.get("project", {}).get("version", None)
    except Exception:
        orbit_version = None

    # Codec library versions
    def get_version_safe(pkg):
        try:
            mod = importlib.import_module(pkg)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return None
    lz4_version = get_version_safe("lz4")
    zstd_version = get_version_safe("zstandard")
    try:
        import lzma
        lzma_version = sys.version.split()[0]  # stdlib, use Python version
    except Exception:
        lzma_version = None
    numpy_version = np.__version__

    # Dataset file path and size
    dataset_path = input_path
    try:
        dataset_size = os.path.getsize(dataset_path)
    except Exception:
        dataset_size = None

    # Timestamp
    timestamp = datetime.datetime.now().isoformat()

    manifest = {
        "config": dict(config.__dict__),
        "orbit_version": orbit_version,
        "codec_versions": {
            "lz4": lz4_version,
            "zstd": zstd_version,
            "lzma": lzma_version,
        },
        "numpy_version": numpy_version,
        "dataset_path": dataset_path,
        "dataset_size_bytes": dataset_size,
        "timestamp": timestamp,
    }
    with open(os.path.join(output_dir, "reproducibility_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    from bandit.linucb import LinUCB
    from bandit.policy import PolicyLogger
    from features.extractor import BlockFeatureExtractor
    from bandit.action_space import ActionSpace
    from io.reader import StreamingReader
    from core.processor import split_into_blocks
    from core.block import Block
    from evaluation.oracle import compute_oracle_actions

    # Setup extractor, policy, action_space
    extractor = BlockFeatureExtractor()
    policy = LinUCB(n_actions=config.n_actions, feature_dim=extractor.feature_dim, alpha=config.alpha)
    logger = PolicyLogger(policy)
    action_space = ActionSpace([k for k in CODEC_REGISTRY.keys()])
    compressor = ORBITCompressor(config, extractor, logger, action_space)


    # Read all blocks for pipeline and oracle (single pass)
    reader = StreamingReader(input_path, config.block_size)
    blocks = list(split_into_blocks(reader, config.block_size))

    # Compute oracle actions using the same block list
    oracle_actions = compute_oracle_actions(blocks, CODEC_REGISTRY)
    for block, oracle_action in zip(blocks, oracle_actions):
        logger.record_oracle_action(block.block_id, oracle_action)

    # Run ORBIT pipeline
    orbit_results = []
    try:
        orbit_results = compressor.compress_file(input_path, os.path.join(output_dir, "orbit_output.bin"))
    except Exception:
        pass

    # Compute regret curve
    regret_curve = logger.compute_cumulative_regret()

    # Save normalized regret curve for this run
    total_bytes = sum(r.get("original_size", 0) for r in orbit_results)
    normalized_regret_curve = logger.compute_normalized_regret(total_bytes)
    regret_records = []
    for idx, entry in enumerate(logger.log):
        block_id = entry.get("block_id")
        block_id_val = int(block_id) if isinstance(block_id, int) else int(idx)
        actual_action = int(entry.get("action", -1))
        oracle_action = int(logger._oracle_actions.get(block_id, -1))
        regret_records.append(
            {
                "block_id": block_id_val,
                "normalized_regret": float(normalized_regret_curve[idx]),
                "cumulative_regret": float(regret_curve[idx]),
                "oracle_action": oracle_action,
                "actual_action": actual_action,
            }
        )

    seed = config.random_seed if getattr(config, "random_seed", None) is not None else "none"
    regret_path = os.path.join(output_dir, f"regret_curve_run{seed}.json")
    with open(regret_path, "w", encoding="utf-8") as f:
        json.dump(regret_records, f, indent=2)


    # Run baselines for all codecs (full-file)
    baseline_metrics = {}
    for codec_id, codec in CODEC_REGISTRY.items():
        result = run_baseline(input_path, codec)
        baseline_metrics[result["codec_name"]] = result

    # Run blockwise baselines for all codecs
    from evaluation.baseline import run_baseline_blockwise
    baselines_blockwise = {}
    for codec_id, codec in CODEC_REGISTRY.items():
        result = run_baseline_blockwise(input_path, codec, config.block_size)
        baselines_blockwise[result["codec_name"]] = result

    # Aggregate ORBIT metrics
    orbit_metrics = aggregate_block_results(orbit_results)

    # Save and return
    combined = {
        "orbit_metrics": orbit_metrics,
        "baseline_metrics": baseline_metrics,
        "baselines_blockwise": baselines_blockwise,
        "regret_curve": regret_curve,
    }
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    return combined



def run_ablation_study(datasets: list, output_dir: str, feature_sets: list[list[str]]) -> list[dict]:
    """
    For each dataset entry and feature_set, run the full pipeline and save results with dataset name and feature_set label.
    """
    import os
    from features.extractor import BlockFeatureExtractor
    from bandit.linucb import LinUCB
    from bandit.action_space import ActionSpace
    from configs.schema import ORBITConfig
    from pipeline.compressor import ORBITCompressor
    from evaluation.metrics import aggregate_block_results

    os.makedirs(output_dir, exist_ok=True)
    config = ORBITConfig.load_yaml("configs/default.yaml")
    results = []
    for dataset in datasets:
        for feature_set in feature_sets:
            extractor = BlockFeatureExtractor(enabled_features=feature_set)
            policy = LinUCB(n_actions=config.n_actions, feature_dim=len(feature_set), alpha=config.alpha)
            action_space = ActionSpace(["raw", "lz4", "zstd", "lzma"][:config.n_actions])
            compressor = ORBITCompressor(config, extractor, policy, action_space)
            try:
                orbit_results = compressor.compress_file(dataset.path, os.path.join(output_dir, f"orbit_{dataset.name}_{'_'.join(feature_set)}.bin"))
            except Exception:
                orbit_results = []
            metrics = aggregate_block_results(orbit_results)
            metrics["feature_set"] = list(feature_set)
            metrics["dataset_name"] = dataset.name
            results.append(metrics)
    with open(os.path.join(output_dir, "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def run_core_comparison(
    manifest_path: str,
    config: ORBITConfig,
    output_dir: str,
    n_runs: int = 5,
) -> None:
    """
    For each DatasetEntry in a manifest, run ORBIT repeated experiments and
    blockwise baselines for every codec, then save a flat comparison table.
    """
    from evaluation.dataset import load_dataset_manifest, validate_manifest
    from evaluation.baseline import run_baseline_blockwise
    from evaluation.metrics import throughput_mbps, validate_comparison_record

    os.makedirs(output_dir, exist_ok=True)
    datasets = load_dataset_manifest(manifest_path)
    missing = validate_manifest(datasets)
    if missing:
        raise RuntimeError(f"Manifest contains missing or unreadable files: {missing}")

    rows: list[dict] = []

    def _mean(values: list[float | None]) -> float | None:
        numeric = [float(v) for v in values if v is not None]
        return float(np.mean(numeric)) if numeric else None

    def _extract_orbit_metric(run_result: dict, metric_name: str) -> float | None:
        orbit_metrics = run_result.get("orbit_metrics", {})
        if not isinstance(orbit_metrics, dict):
            return None

        if metric_name in orbit_metrics and orbit_metrics.get(metric_name) is not None:
            return float(orbit_metrics[metric_name])

        if metric_name == "compression_ratio":
            mean_cr = orbit_metrics.get("mean_compression_ratio")
            if mean_cr is not None:
                return float(mean_cr)
            total_original = orbit_metrics.get("total_original_bytes")
            total_compressed = orbit_metrics.get("total_compressed_bytes")
            if total_original:
                return float(total_compressed) / float(total_original)

        return None

    for dataset in datasets:
        dataset_output_dir = os.path.join(output_dir, f"core_{dataset.name}")
        os.makedirs(dataset_output_dir, exist_ok=True)

        orbit_repeated = run_repeated_experiment(
            dataset.path,
            config,
            dataset_output_dir,
            n_runs=n_runs,
        )

        orbit_runs = orbit_repeated.get("all_results", [])
        compression_values = []
        throughput_values = []
        overhead_values = []
        for run_result in orbit_runs:
            compression_values.append(_extract_orbit_metric(run_result, "compression_ratio"))
            throughput_values.append(_extract_orbit_metric(run_result, "throughput_mbps"))
            overhead_values.append(_extract_orbit_metric(run_result, "overhead_ratio"))

        rows.append(
            {
                "run_id": f"{dataset.name}:orbit",
                "dataset_name": dataset.name,
                "method_name": "orbit",
                "compression_ratio": _mean(compression_values),
                "throughput_mbps": _mean(throughput_values),
                "overhead_ratio": _mean(overhead_values),
                "block_size": config.block_size,
                "dataset_path": dataset.path,
                "method": "orbit",
                "codec_name": "ORBIT",
                "n_runs": n_runs,
                "seed": config.random_seed,
                "compression_ratio_mean": _mean(compression_values),
                "throughput_mbps_mean": _mean(throughput_values),
                "overhead_ratio_mean": _mean(overhead_values),
            }
        )

        for codec_id, codec in CODEC_REGISTRY.items():
            baseline = run_baseline_blockwise(dataset.path, codec, config.block_size)
            rows.append(
                {
                    "run_id": f"{dataset.name}:baseline:{codec_id}",
                    "dataset_name": dataset.name,
                    "method_name": "baseline_blockwise",
                    "compression_ratio": baseline.get("compression_ratio"),
                    "throughput_mbps": throughput_mbps(
                        baseline.get("total_original", 0),
                        baseline.get("elapsed_ms", 0.0),
                    ),
                    "overhead_ratio": 0.0,
                    "block_size": config.block_size,
                    "dataset_path": dataset.path,
                    "method": "baseline_blockwise",
                    "codec_id": codec_id,
                    "codec_name": baseline.get("codec_name", type(codec).__name__),
                    "n_runs": 1,
                    "seed": config.random_seed,
                    "compression_ratio_mean": baseline.get("compression_ratio"),
                    "throughput_mbps_mean": throughput_mbps(
                        baseline.get("total_original", 0),
                        baseline.get("elapsed_ms", 0.0),
                    ),
                    "overhead_ratio_mean": 0.0,
                }
            )

    for record in rows:
        missing_keys = validate_comparison_record(record)
        if missing_keys:
            raise ValueError(
                f"Invalid core comparison record for run_id={record.get('run_id')}: missing keys {missing_keys}"
            )

    with open(os.path.join(output_dir, "core_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_path> <output_dir>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    from configs.schema import ORBITConfig
    config = ORBITConfig.load_yaml("configs/default.yaml")
    result = run_experiment(input_path, config, output_dir)
    print(json.dumps(result, indent=2))
