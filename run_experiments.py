from configs.schema import ORBITConfig
from evaluation.dataset import load_dataset_manifest
from evaluation.runner import (
    generate_ablation_configs,
    prepare_ablation_table,
    prepare_block_size_plot_data,
    prepare_regret_plot_data,
    prepare_table1,
    run_ablation_study,
    run_block_size_sweep,
    run_core_comparison,
)


def main() -> None:
    config = ORBITConfig.load_yaml("configs/default.yaml")

    run_core_comparison("evaluation/datasets.yaml", config, "outputs", n_runs=5)

    datasets = load_dataset_manifest("evaluation/datasets.yaml")
    ablation_configs = generate_ablation_configs(config)
    run_ablation_study(
        datasets,
        "outputs",
        [feature_list for _, feature_list in ablation_configs],
    )

    first_dataset_path = datasets[0].path
    run_block_size_sweep(first_dataset_path, config, "outputs")

    prepare_table1("outputs/core_comparison.json", "outputs/table1.json")
    prepare_regret_plot_data(
        "outputs/core_mixed_corpus/regret_curve_aggregated.json",
        "outputs/regret_plot_data.json",
    )
    prepare_ablation_table("outputs/ablation_results.json", "outputs/ablation_table.json")
    prepare_block_size_plot_data("outputs/block_size_sweep.json", "outputs/block_size_plot.json")

    print("All experiments complete. Outputs in outputs/")


if __name__ == "__main__":
    main()
