from configs.schema import ORBITConfig
from evaluation.dataset import load_dataset_manifest, preflight_check
from evaluation.runner import ensure_output_dirs, run_experiment, run_repeated_experiment
from orbit_codecs import validate_all_codecs
from pipeline import run_pipeline_check


def main() -> None:
    config = ORBITConfig.load_yaml("configs/default.yaml")

    ensure_output_dirs("outputs")

    entries = load_dataset_manifest("evaluation/datasets.yaml")

    try:
        preflight_check(entries, config)
        print("preflight_check: OK")
    except Exception as exc:
        print(f"preflight_check: FAILED ({exc})")
        raise

    codec_results = validate_all_codecs()
    print(f"validate_all_codecs: {codec_results}")

    pipeline_ok = run_pipeline_check()
    print(f"run_pipeline_check: {pipeline_ok}")

    entry = entries[0]
    from evaluation.runner import run_repeated_experiment
    result = run_repeated_experiment(
        entry.path, config, "outputs", n_runs=2
    )
    print("result top-level keys:", list(result.keys()))
    orbit_metrics = result.get("orbit_metrics", {})
    print(
        "run_experiment summary:",
        {
            "dataset_name": entry.name,
            "dataset_path": entry.path,
            "mean_compression_ratio": orbit_metrics.get("mean_compression_ratio"),
            "mean_reward": orbit_metrics.get("mean_reward"),
            "total_original_bytes": orbit_metrics.get("total_original_bytes"),
            "total_compressed_bytes": orbit_metrics.get("total_compressed_bytes"),
        },
    )

    repeated = run_repeated_experiment(
        entry.path, config, "outputs", n_runs=3
    )
    print("repeated std_compression_ratio:",
          repeated.get("std_compression_ratio"))
    print("repeated n_runs:", repeated.get("n_runs"))


if __name__ == "__main__":
    main()
