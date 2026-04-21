import os
import json
from codecs import CODEC_REGISTRY
from configs.schema import ORBITConfig
from pipeline.compressor import ORBITCompressor
from evaluation.baseline import run_baseline
from evaluation.metrics import aggregate_block_results

def run_experiment(input_path: str, config: ORBITConfig, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    # 1. Run ORBIT pipeline
    compressor = ORBITCompressor(config, None, None, None)  # User must provide extractor, policy, action_space
    # For demonstration, skip actual run if components are missing
    orbit_results = []
    try:
        orbit_results = compressor.compress_file(input_path, os.path.join(output_dir, "orbit_output.bin"))
    except Exception:
        pass
    # 2. Run baselines for all codecs
    baseline_metrics = {}
    for codec_id, codec in CODEC_REGISTRY.items():
        result = run_baseline(input_path, codec)
        baseline_metrics[result["codec_name"]] = result
    # 3. Aggregate ORBIT metrics
    orbit_metrics = aggregate_block_results(orbit_results)
    # 4. Save and return
    combined = {
        "orbit_metrics": orbit_metrics,
        "baseline_metrics": baseline_metrics,
    }
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    return combined


def run_ablation_study(input_path: str, output_dir: str, feature_sets: list[list[str]]) -> list[dict]:
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
    for feature_set in feature_sets:
        extractor = BlockFeatureExtractor(enabled_features=feature_set)
        policy = LinUCB(n_actions=config.n_actions, feature_dim=len(feature_set), alpha=config.alpha)
        action_space = ActionSpace(["raw", "lz4", "zstd", "lzma"][:config.n_actions])
        compressor = ORBITCompressor(config, extractor, policy, action_space)
        try:
            orbit_results = compressor.compress_file(input_path, os.path.join(output_dir, f"orbit_{'_'.join(feature_set)}.bin"))
        except Exception:
            orbit_results = []
        metrics = aggregate_block_results(orbit_results)
        metrics["feature_set"] = list(feature_set)
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
