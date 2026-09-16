# HunyuanImage-3.0 ModelSlim MXFP8 integration review

## Scope and provenance

- SGLang base: `e-martirosian/sglang:hunyuan_image_3`, commit
  `4cfa9120f8d8be64175fa80bd1c2fc99b7c5aee1`.
- msModelSlim base: commit
  `2124d3045b09e5d4144cc784c01f51a1423a5efe`.
- Requested format: AscendV1 `W8A8_MXFP8` for HunyuanImage-3.0-Instruct,
  including attention projection linears but excluding QuaRot/FA quantization.

## Architecture and quantization boundary

HunyuanImage-3.0 is a unified autoregressive image model. The SGLang port uses
SRT tensor-parallel linear layers and `FusedMoE` for the AR backbone, while its
VAE, vision tower, image aligner and diffusion I/O stay as ordinary floating
PyTorch modules.

The selected recipe quantizes AR-transformer linear projections:

- self-attention `qkv_proj`/`q_proj` and `o_proj`;
- routed expert `gate_and_up_proj` and `down_proj`;
- shared expert MLP projections;
- dense MLP projections, if present in a checkpoint variant.

No `online_quarot` or `fa3_quant` processing is used. Attention projection
outputs and the attention kernel therefore remain FP16/BF16. Router logits,
norms, embeddings, LM head, VAE, vision tower and image projection modules also
remain floating point. The VAE and vision weights are copied unchanged only to
keep the converted directory self-contained; they are not quantized.

## Implemented changes

### msModelSlim

- Added a model-free `modelslim_convert` recipe. It reads safetensors lazily,
  converts matching BF16/FP16 tensors to MXFP8, and copies unmatched tensors.
- The recipe targets the released fused `gate_and_up_proj` layout.
- Added an end-to-end command and metadata-scope/QuaRot check under
  `example/HunyuanImage3/README.md`.

This approach avoids instantiating the full model and requires no calibration
forward. CPU conversion is supported; worker count should be reduced on hosts
with limited RAM.

### SGLang

- The HunyuanImage-3 pipeline now resolves serialized quantization metadata and
  forwards an SRT ModelSlim config into the AR backbone.
- The top-level model propagates the config into all decoder layers and uses
  checkpoint-compatible root prefixes.
- Fused QKV reordering now uses the tensor's actual inner dimension, so the
  same mapping supports both weights and MXFP8 block-scale tensors.
- ModelSlim recognizes released fused expert names and maps a checkpoint
  `gate_and_up_proj` to runtime `gate_up_proj`.
- post-load processing now includes `FusedMoEMethodBase`, enabling MXFP8 weight
  layout conversion for routed experts.
- Unified component loading accepts the AscendV1
  `quant_model_weights.safetensors.index.json` index for VAE and vision tensors.

## Compatibility and risks

### Hardware

The current SGLang NPU MXFP8 dense and MoE implementations are explicitly A5 /
Ascend 950 paths. Generating the checkpoint on CPU does not make those kernels
available on Atlas A2/A3 or Ascend 910B/910C. Those targets should use INT8
W8A8 dynamic quantization instead.

### Runtime validation boundary

This workspace has no usable Ascend runtime (`libhccl.so` is unavailable), so
the following remain target-hardware validation items:

- MXFP8 grouped-matmul dispatch and layout on A5;
- TP=8 expert scale sharding and numerical output;
- peak device memory and end-to-end image quality;
- FSDP combined with serialized MXFP8 (not recommended for the first run).

### Quality risk

Linear-only MXFP8 leaves the attention kernel and MoE router floating point,
but it is not an accuracy guarantee because Q/K/V/O projections are quantized.
Compare a fixed prompt/seed set against BF16 before production use. If quality
regresses, first keep Q/K projections floating; after that, keep the shared MLP
floating and quantize only routed experts.

## Validation performed

- Python bytecode compilation passed for all changed SGLang Python files.
- Four CPU-only source/AST contract checks passed.
- The new YAML passes the `modelslim_convert` Pydantic schema in the configured
  Python environment.
- Full pytest could not run because pytest is not installed.
- Import-level SGLang runtime testing could not run because `orjson` is absent.
- NPU execution was not run because CANN/HCCL runtime libraries are unavailable.

## Recommended first hardware run

1. Generate the AscendV1 directory with the recipe from
   `example/HunyuanImage3/README.md`.
2. Confirm the metadata checker finds quantized attention projections and zero
   QuaRot tensors.
3. Start SGLang with the original `--model-path`, the quantized
   `--transformer-weights-path`, TP=8 and without FSDP/CPU offload for the first
   smoke test.
4. Verify one text-to-image request, then compare deterministic BF16 and MXFP8
   outputs and collect memory/runtime profiling.
