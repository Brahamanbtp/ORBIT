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
    os.makedirs(output_dir, exist_ok=True)
    if getattr(config, "random_seed", None) is not None:
        import numpy as np
        np.random.seed(config.random_seed)

    from bandit.linucb import LinUCB
    from bandit.policy import PolicyLogger
    from features.extractor import BlockFeatureExtractor
    from bandit.action_space import ActionSpace
    from core.processor import split_into_blocks
    from core.block import Block
    from evaluation.oracle import compute_oracle_actions

    # Setup extractor, policy, action_space
    extractor = BlockFeatureExtractor()
    policy = LinUCB(n_actions=config.n_actions, feature_dim=extractor.feature_dim, alpha=config.alpha)
    logger = PolicyLogger(policy)
    action_space = ActionSpace([k for k in CODEC_REGISTRY.keys()])
    compressor = ORBITCompressor(config, extractor, logger, action_space)

    # Read all blocks for oracle
    reader = StreamingReader(input_path, config.block_size)
    blocks = list(split_into_blocks(reader, config.block_size))

    # Compute oracle actions
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
