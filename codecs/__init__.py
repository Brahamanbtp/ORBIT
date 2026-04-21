

from utils.logging import logger

_adapter_specs = [
	("LZ4Adapter", "codecs.lz4_adapter", "lz4.frame"),
	("ZstdAdapter", "codecs.zstd_adapter", "zstandard"),
	("LZMAAdapter", "codecs.lzma_adapter", None),  # lzma is stdlib
	("RawAdapter", "codecs.raw_adapter", None),
]

_adapters = []
_available_codec_names = []
for class_name, module_name, required_lib in _adapter_specs:
	try:
		if required_lib:
			__import__(required_lib)
		module = __import__(module_name, fromlist=[class_name])
		adapter_cls = getattr(module, class_name)
		_adapters.append(adapter_cls)
		_available_codec_names.append(class_name)
	except ImportError as e:
		logger.warning(f"Codec {class_name} unavailable: {e}. Skipping.")



# Build registry and validate codec_id matches dict key
CODEC_REGISTRY: dict[int, object] = {}
CODEC_ID_MAP: dict[str, int] = {}
for adapter_cls in _adapters:
	instance = adapter_cls()
	key = adapter_cls.codec_id
	if instance.codec_id != key:
		raise RuntimeError(f"Codec {adapter_cls.__name__} instance codec_id {instance.codec_id} does not match registry key {key}")
	CODEC_REGISTRY[key] = instance
	CODEC_ID_MAP[adapter_cls.__name__] = key

def available_codecs() -> list[str]:
	"""Return names of successfully loaded codecs."""
	return list(_available_codec_names)

def get_codec(codec_id: int):
	return CODEC_REGISTRY[codec_id]
