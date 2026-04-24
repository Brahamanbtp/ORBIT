
def run_pipeline_check() -> bool:
    try:
        from orbit_codecs.raw_adapter import RawAdapter
        from features.extractor import BlockFeatureExtractor
        from bandit.linucb import LinUCB
        from core.block import Block
        from bandit.action_space import ActionSpace
        from pipeline.router import BlockRouter

        _ = RawAdapter()
        extractor = BlockFeatureExtractor(
            enabled_features=["entropy", "rle_ratio", "repetition"]
        )
        linucb = LinUCB(n_actions=1, feature_dim=3, alpha=1.0)
        block = Block(block_id=0, data=bytes(4096), size=4096, offset=0)
        action_space = ActionSpace(action_names=["raw"])
        router = BlockRouter(extractor, linucb, action_space)
        router.route(block)
        return True
    except Exception:
        return False
