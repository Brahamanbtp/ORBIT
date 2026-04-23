from __future__ import annotations


def main() -> None:
    # 1) codecs registry and validation import
    try:
        from codecs import CODEC_REGISTRY, validate_all_codecs

        codec_names = [type(codec).__name__ for _, codec in sorted(CODEC_REGISTRY.items())]
        print(f"PASS 1: codecs imported; codec names found: {codec_names}")
    except Exception as exc:
        print(f"FAIL 1: codecs import/check failed: {exc}")

    # 2) pipeline import and check call
    try:
        from pipeline import run_pipeline_check

        pipeline_ok = run_pipeline_check()
        print(f"PASS 2: run_pipeline_check result: {pipeline_ok}")
    except Exception as exc:
        print(f"FAIL 2: pipeline check failed: {exc}")

    # 3) load default config
    try:
        from configs.schema import ORBITConfig

        config = ORBITConfig.load_yaml("configs/default.yaml")
        print(f"PASS 3: config loaded: {config}")
    except Exception as exc:
        print(f"FAIL 3: config load failed: {exc}")

    # 4) load manifest and validate paths
    try:
        from evaluation.dataset import load_dataset_manifest, validate_manifest

        entries = load_dataset_manifest("evaluation/datasets.yaml")
        missing = validate_manifest(entries)
        print(f"PASS 4: manifest loaded; missing paths: {missing}")
    except Exception as exc:
        print(f"FAIL 4: dataset manifest check failed: {exc}")

    # 5) import runner
    try:
        import evaluation.runner  # noqa: F401

        print("PASS 5: runner imported OK")
    except Exception as exc:
        print(f"FAIL 5: runner import failed: {exc}")

    # 6) final line
    print("Smoke test complete")


if __name__ == "__main__":
    main()
