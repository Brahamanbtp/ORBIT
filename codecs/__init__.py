
from codecs.lz4_adapter import LZ4Adapter
from codecs.zstd_adapter import ZstdAdapter
from codecs.lzma_adapter import LZMAAdapter
from codecs.raw_adapter import RawAdapter

CODEC_REGISTRY: dict[int, object] = {
	LZ4Adapter.codec_id: LZ4Adapter(),
	ZstdAdapter.codec_id: ZstdAdapter(),
	LZMAAdapter.codec_id: LZMAAdapter(),
	RawAdapter.codec_id: RawAdapter(),
}

def get_codec(codec_id: int):
	return CODEC_REGISTRY[codec_id]
