from __future__ import annotations

import numpy as np
from typing import Any


def ensure_output_dirs(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "manifests"), exist_ok=True)


def safe_save_json(data: Any, path: str) -> None:
    serialized = json.dumps(data, indent=2, default=str)
    with open(path, "w", encoding="utf-8") as f:
        f.write(serialized)
    print(f"Saved: {path}")


def verify_seed_consistency(results: list[dict]) -> bool:
    """
    Validate deterministic seed progression and duplicate (run_id, seed) consistency.
    """
    if not results:
        return True

    pair_payloads: dict[tuple[int, int], str] = {}
    for rec in results:
        if "run_id" not in rec or "seed" not in rec or "base_seed" not in rec:
            print("WARNING: verify_seed_consistency missing one of run_id/seed/base_seed fields")
            return False

        run_id = int(rec["run_id"])
        seed = int(rec["seed"])
        base_seed = int(rec["base_seed"])
        expected_seed = base_seed + run_id

        if seed != expected_seed:
            print(
                f"WARNING: seed mismatch for run_id={run_id}: expected {expected_seed}, got {seed}"
            )
            return False

        pair = (run_id, seed)
        payload = json.dumps(rec.get("result", {}), sort_keys=True, default=str)
        if pair in pair_payloads and pair_payloads[pair] != payload:
            print(
                f"WARNING: duplicate (run_id, seed) pair {pair} has inconsistent results"
            )
            return False
        pair_payloads[pair] = payload

    return True


def run_repeated_experiment(input_path: str, config: ORBITConfig, output_dir: str, n_runs: int = 5) -> dict:
    """
    Run run_experiment n_runs times, varying the random seed, and aggregate results.
    Computes mean and std for compression_ratio, mean_reward, and regret_curve.
    Saves results as repeated_results.json in output_dir.
    """
    ensure_output_dirs(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    normalized_regret_curves = []
    seed_consistency_records = []
    base_seed = int(config.random_seed or 0)
    for run_idx in range(n_runs):
        run_config = ORBITConfig(
            **{**config.__dict__, "random_seed": base_seed + run_idx}
        )
        np.random.seed(run_config.random_seed)
        result = run_experiment(input_path, run_config, output_dir)
        all_results.append(result)
        seed_consistency_records.append(
            {
                "run_id": int(run_idx),
                "base_seed": int(base_seed),
                "seed": int(run_config.random_seed),
                "result": result,
            }
        )

        # Collect the per-run normalized regret curve saved by run_experiment.
        regret_path = os.path.join(output_dir, f"regret_curve_run{run_config.random_seed}.json")
        if os.path.exists(regret_path):
            try:
                with open(regret_path, "r", encoding="utf-8") as f:
                    curve = json.load(f)
                if isinstance(curve, list) and curve:
                    normalized_regret_curves.append(curve)
            except Exception:
                pass

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

    ratios = [
        (r.get("orbit_metrics", {}).get("mean_compression_ratio")
         if r.get("orbit_metrics", {}).get("mean_compression_ratio") is not None
         else r.get("orbit_metrics", {}).get("compression_ratio"))
        for r in all_results
    ]
    ratios = [v for v in ratios if v is not None]
    compression_ratio_mean = float(np.mean(ratios)) if ratios else None
    compression_ratio_std = float(np.std(ratios)) if ratios else None

    rewards = [
        r.get("orbit_metrics", {}).get("mean_reward")
        for r in all_results
    ]
    rewards = [v for v in rewards if v is not None]
    mean_reward_mean = float(np.mean(rewards)) if rewards else None
    mean_reward_std = float(np.std(rewards)) if rewards else None

    regret_curves = collect_metric(all_results, ["regret_curve"])

    # Compute mean/std for scalar metrics
    agg = {
        "mean_compression_ratio": compression_ratio_mean,
        "std_compression_ratio": compression_ratio_std,
        "mean_reward": mean_reward_mean,
        "std_reward": mean_reward_std,
        "compression_ratio_mean": compression_ratio_mean,
        "compression_ratio_std": compression_ratio_std,
        "mean_reward_mean": mean_reward_mean,
        "mean_reward_std": mean_reward_std,
        "n_runs": int(n_runs),
        "dataset_path": str(input_path),
        "input_path": str(input_path),
        "block_size": int(config.block_size),
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

    # Aggregate per-run normalized regret curves element-wise by block index.
    aggregated_regret_records = []
    if normalized_regret_curves:
        min_len = min(len(curve) for curve in normalized_regret_curves)
        curves_trunc = [curve[:min_len] for curve in normalized_regret_curves]

        normalized_regret_arr = np.array(
            [
                [float(entry.get("normalized_regret", 0.0)) for entry in curve]
                for curve in curves_trunc
            ],
            dtype=float,
        )
        mean_vals = normalized_regret_arr.mean(axis=0)
        std_vals = normalized_regret_arr.std(axis=0)

        for idx in range(min_len):
            block_id = curves_trunc[0][idx].get("block_id", idx)
            aggregated_regret_records.append(
                {
                    "block_id": int(block_id),
                    "mean_normalized_regret": float(mean_vals[idx]),
                    "std_normalized_regret": float(std_vals[idx]),
                }
            )

    agg["seed_consistent"] = verify_seed_consistency(seed_consistency_records)

    safe_save_json(aggregated_regret_records, os.path.join(output_dir, "regret_curve_aggregated.json"))
    agg["regret_curve_aggregated"] = aggregated_regret_records

    agg["all_results"] = all_results

    safe_save_json(agg, os.path.join(output_dir, "repeated_results.json"))
    return agg
import os
import json
from orbit_codecs import CODEC_REGISTRY
from orbit_codecs.base import CodecAdapter
from bandit.reward import compute_reward
from configs.schema import ORBITConfig
from pipeline.compressor import ORBITCompressor
from evaluation.baseline import run_baseline
from evaluation.metrics import aggregate_block_results, compression_ratio
from evaluation.oracle import compute_oracle_stats


def generate_ablation_configs(base_config: ORBITConfig) -> list[tuple[str, list[str]]]:
    """
    Return all non-empty feature subsets used for ablation experiments.
    Each entry is (label, feature_list), where label is "+"-joined sorted features.
    """
    from itertools import combinations

    _ = base_config
    features = ["entropy", "rle_ratio", "repetition"]
    configs: list[tuple[str, list[str]]] = []

    for subset_size in range(1, len(features) + 1):
        for subset in combinations(features, subset_size):
            feature_list = sorted(list(subset))
            label = "+".join(feature_list)
            configs.append((label, feature_list))

    return configs

def run_experiment(input_path: str, config: ORBITConfig, output_dir: str) -> dict:
    ensure_output_dirs(output_dir)
    from orbit_codecs import validate_all_codecs, snapshot_registry
    codec_snapshot_start = snapshot_registry()

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
        "codec_snapshot": codec_snapshot_start,
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
    safe_save_json(manifest, os.path.join(output_dir, "reproducibility_manifest.json"))

    # Validate all codecs roundtrip after manifest write and before pipeline runs.
    roundtrip_results = validate_all_codecs()
    failed = [name for name, ok in roundtrip_results.items() if not ok]
    if failed:
        raise RuntimeError(f"Codec roundtrip validation failed for: {failed}")

    from orbit_codecs import available_codecs, CODEC_REGISTRY
    # Assert codec registry matches config
    if len(available_codecs()) != config.n_actions:
        raise RuntimeError(f"Number of available codecs ({len(available_codecs())}) does not match config.n_actions ({config.n_actions}). Available: {available_codecs()}")
    if set(CODEC_REGISTRY.keys()) != set(range(config.n_actions)):
        raise RuntimeError(f"CODEC_REGISTRY keys {set(CODEC_REGISTRY.keys())} do not match expected set {set(range(config.n_actions))} for n_actions={config.n_actions}.")

    from bandit.linucb import LinUCB
    from bandit.policy import PolicyLogger
    from features.extractor import BlockFeatureExtractor
    from bandit.action_space import ActionSpace
    from core.block import Block
    from orbit_io.reader import StreamingReader
    from core.processor import split_into_blocks
    from evaluation.oracle import compute_oracle_actions

    # Setup extractor, policy, action_space
    extractor = BlockFeatureExtractor()
    policy = LinUCB(n_actions=config.n_actions, feature_dim=extractor.feature_dim, alpha=config.alpha)
    logger = PolicyLogger(policy)
    policy_logger = logger
    action_space = ActionSpace([
        CODEC_REGISTRY[k].__class__.__name__ for k in sorted(CODEC_REGISTRY.keys())
    ])
    compressor = ORBITCompressor(config, extractor, logger, action_space)

    # Run ORBIT pipeline
    block_results = compressor.compress_file(
        input_path, os.path.join(output_dir, "orbit_output.bin")
    )

    print(f"Blocks processed: {len(block_results)}")

    reader = StreamingReader(input_path, config.block_size)
    blocks = list(split_into_blocks(reader, config.block_size))

    oracle_actions = compute_oracle_actions(blocks, CODEC_REGISTRY)

    for i, block in enumerate(blocks):
        oracle_action_id = oracle_actions[i]
        codec = CODEC_REGISTRY[oracle_action_id]
        oracle_compressed = codec.compress(block.data)
        oracle_reward = compute_reward(block.size, len(oracle_compressed), 0.0)
        policy_logger.record_oracle_action(block.block_id, oracle_action_id, oracle_reward)

    regret_curve = policy_logger.compute_cumulative_regret()
    rc = regret_curve
    early_slope = (rc[49] - rc[0]) / 49 if len(rc) > 50 else 0
    late_slope = (rc[-1] - rc[-51]) / 50 if len(rc) > 50 else 0
    slope_reduction_pct = (1 - late_slope / early_slope) * 100 if early_slope > 0 else 0


    # Aggregate ORBIT metrics
    orbit_metrics = aggregate_block_results(block_results)
    mean_cr = orbit_metrics.get("mean_compression_ratio")
    orbit_metrics["space_saving_pct"] = (
        float((1.0 - float(mean_cr)) * 100.0) if mean_cr is not None else None
    )
    total_original_bytes = float(orbit_metrics.get("total_original_bytes", 0.0) or 0.0)
    total_elapsed_seconds = (
        sum(float(r.get("compression_ms", 0.0)) for r in block_results) / 1000.0
        if block_results
        else 0.0
    )
    orbit_throughput_mbps = (
        (total_original_bytes / 1e6) / total_elapsed_seconds
        if total_elapsed_seconds > 0.0
        else 0.0
    )
    if block_results:
        burn_in = int(len(block_results) * 0.25)
        converged_results = block_results[burn_in:]
        if converged_results:
            orbit_metrics["converged_ratio"] = float(
                sum(
                    compression_ratio(r["original_size"], r["compressed_size"])
                    for r in converged_results
                    if r["original_size"] > 0
                ) / len(converged_results)
            )
        else:
            orbit_metrics["converged_ratio"] = 0.0
    else:
        orbit_metrics["converged_ratio"] = 0.0

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

    baseline_ratios = [
        float(v.get("compression_ratio"))
        for v in baselines_blockwise.values()
        if isinstance(v, dict) and v.get("compression_ratio") is not None
    ]
    best_baseline_ratio = min(baseline_ratios) if baseline_ratios else None
    orbit_compression_ratio = orbit_metrics.get("mean_compression_ratio")
    orbit_gain_vs_best_baseline = (
        float(best_baseline_ratio - orbit_compression_ratio)
        if best_baseline_ratio is not None and orbit_compression_ratio is not None
        else None
    )
    orbit_gain_pct = (
        float(orbit_gain_vs_best_baseline * 100.0)
        if orbit_gain_vs_best_baseline is not None
        else None
    )

    # Save and return
    combined = {
        "mean_compression_ratio": orbit_metrics.get("mean_compression_ratio"),
        "mean_reward": orbit_metrics.get("mean_reward"),
        "total_original_bytes": orbit_metrics.get("total_original_bytes"),
        "total_compressed_bytes": orbit_metrics.get("total_compressed_bytes"),
        "codec_selection_counts": orbit_metrics.get("codec_selection_counts"),
        "regret_curve": regret_curve,
        "block_results": block_results,
        "orbit_metrics": orbit_metrics,
        "baseline_metrics": baseline_metrics,
        "baselines_blockwise": baselines_blockwise,
        "best_baseline_ratio": best_baseline_ratio,
        "orbit_gain_vs_best_baseline": orbit_gain_vs_best_baseline,
        "orbit_gain_pct": orbit_gain_pct,
        "regret_slope_reduction_pct": float(slope_reduction_pct),
        "orbit_throughput_mbps": float(orbit_throughput_mbps),
    }

    codec_snapshot_end = snapshot_registry()
    if codec_snapshot_end != codec_snapshot_start:
        raise RuntimeError(
            "Codec registry mutated during experiment: "
            f"start={codec_snapshot_start}, end={codec_snapshot_end}"
        )

    ensure_output_dirs(output_dir)
    safe_save_json(block_results, os.path.join(output_dir, "block_results.json"))
    policy_logger.dump_log(
        os.path.join(output_dir, "logs", "policy_log.jsonl")
    )
    policy_logger.dump_weight_snapshots(
        os.path.join(output_dir, "logs", "weight_snapshots.jsonl")
    )

    safe_save_json(combined, os.path.join(output_dir, "results.json"))

    # Save normalized regret curve for this run.
    total_bytes = sum(r.get("original_size", 0) for r in block_results)
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
    regret_path = os.path.join(output_dir, f"regret_curve_run{seed if seed is not None else 0}.json")
    safe_save_json(regret_records, regret_path)

    return_payload = dict(combined)
    return_payload["dataset_name"] = os.path.splitext(os.path.basename(input_path))[0]
    return_payload["dataset_path"] = input_path
    return return_payload



def run_ablation_study(input_path_or_entries: str | list, output_dir: str, feature_sets: list[list[str]]) -> list[dict]:
    """
    For each dataset entry and feature_set, run the full pipeline and save results with dataset name and feature_set label.
    """
    ensure_output_dirs(output_dir)
    import os
    from features.extractor import BlockFeatureExtractor
    from bandit.linucb import LinUCB
    from bandit.action_space import ActionSpace
    from configs.schema import ORBITConfig
    from pipeline.compressor import ORBITCompressor
    from evaluation.metrics import aggregate_block_results

    if isinstance(input_path_or_entries, str):
        from evaluation.dataset import DatasetEntry
        datasets = [DatasetEntry(
            name="default", path=input_path_or_entries,
            size_bytes=0, content_type="mixed", description=""
        )]
    else:
        datasets = input_path_or_entries

    os.makedirs(output_dir, exist_ok=True)
    config = ORBITConfig.load_yaml("configs/default.yaml")
    _ = feature_sets
    ablation_configs = generate_ablation_configs(config)
    results = []
    for entry in datasets:
        for label, feature_set in ablation_configs:
            try:
                extractor = BlockFeatureExtractor(enabled_features=feature_set)
                print(f"Running ablation: {label} (feature_dim={extractor.feature_dim})")
                policy = LinUCB(n_actions=config.n_actions, feature_dim=len(feature_set), alpha=config.alpha)
                action_space = ActionSpace(["raw", "lz4", "zstd", "lzma"][:config.n_actions])
                compressor = ORBITCompressor(config, extractor, policy, action_space)
                orbit_results = compressor.compress_file(entry.path, os.path.join(output_dir, f"orbit_{entry.name}_{label.replace('+', '_')}.bin"))
            except Exception as exc:
                print(f"Ablation failed for feature_set '{label}': {exc}")
                continue
            agg = aggregate_block_results(orbit_results)
            metrics = dict(agg) if isinstance(agg, dict) else {}
            ratio = agg.get("mean_compression_ratio", None) if isinstance(agg, dict) else None
            reward = agg.get("mean_reward", None) if isinstance(agg, dict) else None
            metrics["compression_ratio"] = float(ratio) if ratio is not None else 0.0
            metrics["mean_reward"] = float(reward) if reward is not None else 0.0
            codec_counts = metrics.get("codec_selection_counts", {})
            total_actions = sum(codec_counts.values())
            if total_actions > 0:
                codec_selection_entropy = float(
                    -sum(
                        (count / total_actions) * np.log2(count / total_actions)
                        for count in codec_counts.values()
                        if count > 0
                    )
                )
            else:
                codec_selection_entropy = 0.0

            metrics["codec_selection_entropy"] = codec_selection_entropy
            metrics["feature_set"] = list(feature_set)
            metrics["feature_set_label"] = label
            metrics["dataset_name"] = entry.name
            results.append(metrics)
    safe_save_json(results, os.path.join(output_dir, "ablation_results.json"))
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
    ensure_output_dirs(output_dir)
    import os
    from evaluation.dataset import load_dataset_manifest, validate_manifest
    from evaluation.baseline import run_baseline_blockwise
    from evaluation.metrics import throughput_mbps, validate_comparison_record

    os.makedirs(output_dir, exist_ok=True)
    datasets = load_dataset_manifest(manifest_path)
    missing = validate_manifest(datasets)
    if missing:
        raise RuntimeError(f"Manifest contains missing or unreadable files: {missing}")

    rows: list[dict] = []

    def verify_block_size_consistency(config: ORBITConfig, baseline_results: list[dict]) -> None:
        expected_block_size = config.block_size
        if expected_block_size is None:
            raise AssertionError("Config block_size is None; cannot verify block size consistency.")

        for idx, result in enumerate(baseline_results):
            if "block_size" not in result:
                raise AssertionError(
                    f"Missing block_size in baseline_results[{idx}] for run_id={result.get('run_id')}"
                )
            actual_block_size = result.get("block_size")
            if actual_block_size != expected_block_size:
                raise AssertionError(
                    f"Block size mismatch in run_id={result.get('run_id')}: "
                    f"expected {expected_block_size}, got {actual_block_size}"
                )

        orbit_rows = [r for r in baseline_results if str(r.get("method_name", "")).lower() == "orbit"]
        if not orbit_rows:
            raise AssertionError("No ORBIT row found in baseline_results to verify ORBIT block size.")

        for orbit_row in orbit_rows:
            observed_orbit_block_size = orbit_row.get("orbit_block_size_observed", orbit_row.get("block_size"))
            if observed_orbit_block_size != expected_block_size:
                raise AssertionError(
                    f"ORBIT block size mismatch for run_id={orbit_row.get('run_id')}: "
                    f"expected {expected_block_size}, observed {observed_orbit_block_size}"
                )

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
                return compression_ratio(int(total_original), int(total_compressed))

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

        manifest_path = os.path.join(dataset_output_dir, "reproducibility_manifest.json")
        orbit_block_size_observed = None
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
                orbit_block_size_observed = manifest.get("config", {}).get("block_size")
            except Exception:
                orbit_block_size_observed = None

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
                "orbit_block_size_observed": orbit_block_size_observed,
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

    verify_block_size_consistency(config, rows)

    safe_save_json(rows, os.path.join(output_dir, "core_comparison.json"))


def run_block_size_sweep(
    input_path: str,
    base_config: ORBITConfig,
    output_dir: str,
    block_sizes: list[int] = None,
) -> None:
    """
    Sweep block sizes and collect repeated-experiment summary metrics.
    Saves results as block_size_sweep.json in output_dir.
    """
    ensure_output_dirs(output_dir)
    import os
    import time
    from evaluation.metrics import throughput_mbps, estimate_convergence_block

    os.makedirs(output_dir, exist_ok=True)
    sizes = block_sizes if block_sizes is not None else [1024, 4096, 16384, 65536]
    input_bytes = os.path.getsize(input_path) if os.path.exists(input_path) else 0
    sweep_rows: list[dict] = []

    for block_size in sizes:
        run_config = ORBITConfig(
            **{**base_config.__dict__, "block_size": int(block_size)}
        )
        block_output_dir = os.path.join(output_dir, f"block_size_{int(block_size)}")

        start = time.time()
        repeated = run_repeated_experiment(
            input_path,
            run_config,
            block_output_dir,
            n_runs=3,
        )
        elapsed_ms = (time.time() - start) * 1000.0

        result = repeated
        ratio = result.get("compression_ratio_mean") or result.get("compression_ratio", 0.0)
        if (ratio is None or ratio == 0.0) and result.get("compression_ratio_mean") is not None:
            ratio = result.get("compression_ratio_mean")

        regret_curve = repeated.get("regret_curve_mean", []) or []
        convergence_idx = estimate_convergence_block(
            regret_curve,
            window=100,
            threshold=0.0001,
        )
        regret_convergence_block = convergence_idx if convergence_idx >= 0 else None

        row = {
            "block_size": int(block_size),
            "mean_compression_ratio": float(ratio) if ratio is not None else 0.0,
            "throughput_mbps": float(throughput_mbps(input_bytes * 3, elapsed_ms)),
            "regret_convergence_block": regret_convergence_block,
        }
        sweep_rows.append(row)

    safe_save_json(sweep_rows, os.path.join(output_dir, "block_size_sweep.json"))


def prepare_table1(core_comparison_path: str, output_path: str) -> None:
    """
    Prepare a pivoted summary table from core_comparison.json.
    Output rows are keyed by dataset_name with one column per method_name.
    """
    with open(core_comparison_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("core_comparison.json must contain a list of records")

    grouped: dict[str, dict[str, list[dict]]] = {}
    for rec in records:
        dataset_name = rec.get("dataset_name")
        method_name = rec.get("method_name")
        if dataset_name is None or method_name is None:
            continue
        grouped.setdefault(str(dataset_name), {}).setdefault(str(method_name), []).append(rec)

    table_rows: list[dict] = []
    for dataset_name, method_groups in grouped.items():
        row: dict = {"dataset_name": dataset_name}
        best_baseline_mean: float | None = None
        orbit_mean: float | None = None

        for method_name, method_records in method_groups.items():
            compression_vals = []
            throughput_vals = []

            for rec in method_records:
                cr_val = rec.get("compression_ratio")
                if cr_val is None:
                    cr_val = rec.get("compression_ratio_mean")
                if cr_val is not None:
                    compression_vals.append(float(cr_val))

                tp_val = rec.get("throughput_mbps")
                if tp_val is None:
                    tp_val = rec.get("throughput_mbps_mean")
                if tp_val is None and str(method_name).lower() == "orbit":
                    tp_val = rec.get("orbit_throughput_mbps")
                if tp_val is not None:
                    throughput_vals.append(float(tp_val))

            mean_cr = float(np.mean(compression_vals)) if compression_vals else None
            std_cr = float(np.std(compression_vals)) if compression_vals else None
            mean_tp = float(np.mean(throughput_vals)) if throughput_vals else None

            row[method_name] = {
                "mean_compression_ratio": mean_cr,
                "std_compression_ratio": std_cr,
                "mean_compression_ratio_pm_std": (
                    f"{mean_cr:.6f} ± {std_cr:.6f}" if mean_cr is not None and std_cr is not None else None
                ),
                "throughput_mbps": mean_tp,
            }

            if method_name.lower() == "orbit":
                orbit_mean = mean_cr
            else:
                if mean_cr is not None and (best_baseline_mean is None or mean_cr < best_baseline_mean):
                    best_baseline_mean = mean_cr

        # Compression ratio: lower is better; positive gain means ORBIT improved over best baseline.
        if orbit_mean is not None and best_baseline_mean is not None and best_baseline_mean > 0:
            row["relative_gain_vs_best_baseline"] = float((best_baseline_mean - orbit_mean) / best_baseline_mean)
        else:
            row["relative_gain_vs_best_baseline"] = None

        table_rows.append(row)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    safe_save_json(table_rows, output_path)


def prepare_regret_plot_data(aggregated_regret_path: str, output_path: str) -> None:
    """
    Convert aggregated regret records into plotting-ready arrays.
    Saves JSON with x, y_mean, y_upper, y_lower, and convergence_block.
    """
    from evaluation.metrics import estimate_convergence_block

    with open(aggregated_regret_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("regret_curve_aggregated.json must contain a list of records")

    sorted_records = sorted(
        [rec for rec in records if isinstance(rec, dict)],
        key=lambda rec: int(rec.get("block_id", 0)),
    )

    x: list[int] = []
    y_mean: list[float] = []
    y_upper: list[float] = []
    y_lower: list[float] = []

    for rec in sorted_records:
        block_id = int(rec.get("block_id", 0))
        mean_val = float(rec.get("mean_normalized_regret", 0.0))
        std_val = float(rec.get("std_normalized_regret", 0.0))

        x.append(block_id)
        y_mean.append(mean_val)
        y_upper.append(mean_val + std_val)
        y_lower.append(mean_val - std_val)

    payload = {
        "x": x,
        "y_mean": y_mean,
        "y_upper": y_upper,
        "y_lower": y_lower,
        "convergence_block": int(estimate_convergence_block(y_mean)) if y_mean else -1,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    safe_save_json(payload, output_path)


def prepare_ablation_table(ablation_path: str, output_path: str) -> None:
    """
    Prepare ranked ablation table from ablation_results.json.
    Adds rank, delta_vs_full_features, and is_best fields.
    """
    with open(ablation_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("ablation_results.json must contain a list of records")

    rows = [rec for rec in records if isinstance(rec, dict)]

    def _compression_ratio(rec: dict) -> float:
        ratio = rec.get("compression_ratio") or rec.get("mean_compression_ratio", 0.0)
        return float(ratio)

    grouped_rows: dict[str, list[dict]] = {}
    for rec in rows:
        dataset_name = str(rec.get("dataset_name", "unknown"))
        grouped_rows.setdefault(dataset_name, []).append(rec)

    table_rows: dict[str, list[dict]] = {}
    full_set = {"entropy", "rle_ratio", "repetition"}
    full_set_label = "entropy+repetition+rle_ratio"

    for dataset_name, dataset_rows in grouped_rows.items():
        full_features_ratio = None
        for rec in dataset_rows:
            feature_set = rec.get("feature_set", [])
            feature_set_label = rec.get("feature_set_label")
            is_full_by_list = isinstance(feature_set, list) and set(feature_set) == full_set
            is_full_by_label = str(feature_set_label) == full_set_label
            if is_full_by_list or is_full_by_label:
                full_features_ratio = _compression_ratio(rec)
                break

        sorted_rows = sorted(dataset_rows, key=_compression_ratio)
        ranked_rows: list[dict] = []
        for idx, rec in enumerate(sorted_rows, start=1):
            row = dict(rec)
            ratio_val = _compression_ratio(row)

            row["rank"] = idx
            row["delta_vs_full_features"] = (
                float(ratio_val - full_features_ratio)
                if full_features_ratio is not None
                else None
            )
            row["is_best"] = idx == 1
            ranked_rows.append(row)

        table_rows[dataset_name] = ranked_rows

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    safe_save_json(table_rows, output_path)


def prepare_block_size_plot_data(sweep_path: str, output_path: str) -> None:
    """
    Convert block-size sweep results into plotting-ready arrays.
    Saves JSON with x, y_ratio, y_throughput, y_convergence, and optimal_block_size.
    """
    with open(sweep_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("block_size_sweep.json must contain a list of records")

    valid_rows = [row for row in rows if isinstance(row, dict)]
    sorted_rows = sorted(valid_rows, key=lambda row: int(row.get("block_size", 0)))

    x: list[int] = []
    y_ratio: list[float] = []
    y_throughput: list[float] = []
    y_convergence: list[int] = []

    for row in sorted_rows:
        block_size = int(row.get("block_size", 0))
        ratio = row.get("compression_ratio")
        if ratio is None:
            ratio = row.get("compression_ratio_mean", 0.0)
        throughput = row.get("throughput_mbps", 0.0)
        convergence = row.get("regret_convergence_block", -1)

        x.append(block_size)
        y_ratio.append(float(ratio))
        y_throughput.append(float(throughput))
        y_convergence.append(int(convergence) if convergence is not None else -1)

    optimal_block_size = -1
    if sorted_rows:
        best_row = max(
            sorted_rows,
            key=lambda row: float(
                row.get("compression_ratio")
                if row.get("compression_ratio") is not None
                else row.get("compression_ratio_mean", float("-inf"))
            ),
        )
        optimal_block_size = int(best_row.get("block_size", -1))

    payload = {
        "x": x,
        "y_ratio": y_ratio,
        "y_throughput": y_throughput,
        "y_convergence": y_convergence,
        "optimal_block_size": optimal_block_size,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    safe_save_json(payload, output_path)


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
