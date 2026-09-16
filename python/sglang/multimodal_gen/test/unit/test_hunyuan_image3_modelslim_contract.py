"""Static contracts for HunyuanImage-3 ModelSlim MXFP8 loading.

These checks intentionally avoid importing the NPU runtime so they can run in
the CPU-only development environment used for this integration.
"""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]


def _source(relative_path: str) -> str:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    ast.parse(source)
    return source


def test_hunyuan_pipeline_resolves_and_postprocesses_modelslim_weights():
    source = _source(
        "python/sglang/multimodal_gen/runtime/pipelines/hunyuan_image3.py"
    )

    assert "resolve_transformer_quant_load_spec(" in source
    assert "SRTModelSlimConfig(dict(quant_config.quant_description))" in source
    assert "quant_config=quant_config" in source
    assert "process_model_weights_after_loading(model)" in source
    assert '"quant_model_weights.safetensors.index.json"' in source


def test_hunyuan_model_propagates_quant_config_with_checkpoint_prefix():
    source = _source(
        "python/sglang/multimodal_gen/runtime/models/dits/hunyuan_image3.py"
    )

    assert "quant_config: Optional[QuantizationConfig] = None" in source
    assert "quant_config=quant_config" in source
    assert 'prefix=maybe_prefix(prefix, "model")' in source
    assert "def _split_qkv_tensor(self, qkv):" in source
    assert "inner_size = qkv.shape[-1]" in source


def test_modelslim_accepts_released_hunyuan_fused_expert_names():
    source = _source(
        "python/sglang/srt/layers/quantization/modelslim/modelslim.py"
    )

    assert 'prefix.endswith(".gate_up_proj")' in source
    assert '("gate_and_up_proj", "gate_and_up_proj", "down_proj")' in source


def test_quant_postprocess_includes_fused_moe_methods():
    source = _source(
        "python/sglang/multimodal_gen/runtime/utils/quantization_utils.py"
    )

    assert "FusedMoEMethodBase" in source
    assert "UnquantizedFusedMoEMethod" in source
