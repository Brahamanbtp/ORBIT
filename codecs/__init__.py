
from codecs.lz4_adapter import LZ4Adapter
from codecs.zstd_adapter import ZstdAdapter
from codecs.lzma_adapter import LZMAAdapter
from codecs.raw_adapter import RawAdapter


# Build registry and validate codec_id matches dict key
_adapters = [LZ4Adapter, ZstdAdapter, LZMAAdapter, RawAdapter]
CODEC_REGISTRY: dict[int, object] = {}
CODEC_ID_MAP: dict[str, int] = {}
for adapter_cls in _adapters:
	instance = adapter_cls()
	key = adapter_cls.codec_id
	if instance.codec_id != key:
		raise RuntimeError(f"Codec {adapter_cls.__name__} instance codec_id {instance.codec_id} does not match registry key {key}")
	CODEC_REGISTRY[key] = instance
	CODEC_ID_MAP[adapter_cls.__name__] = key

def get_codec(codec_id: int):
	return CODEC_REGISTRY[codec_id]
