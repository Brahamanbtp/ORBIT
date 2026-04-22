from utils.byte_utils import validate_roundtrip
import os
def validate_all_codecs(test_data: bytes = None) -> dict:
	"""
	Run validate_roundtrip for each codec in CODEC_REGISTRY. Return dict mapping codec name to bool.
	If test_data is None, use a fixed 512-byte synthetic payload.
	"""
	if test_data is None:
		test_data = os.urandom(512)
	results = {}
	for codec_id, codec in CODEC_REGISTRY.items():
		name = type(codec).__name__
		results[name] = validate_roundtrip(codec, test_data)
	return results


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
