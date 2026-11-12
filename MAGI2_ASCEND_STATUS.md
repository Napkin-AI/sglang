# MAGI-2 Ascend compatibility path

Target: A2/A3, CANN >= 9.1.0, torch_npu >= 2.10.0.post4, four NPUs,
Ulysses=4 and EP=4, full 1920×1088 clip followed by first-frame extraction.
The 64 GB target is treated as a per-NPU limit. Hardware execution and peak
memory are still unverified in this workspace.

## Implemented replacements

| Path | Implementation |
| --- | --- |
| Initial loading | Both DiTs honor `should_start_component_on_cpu()` before loading checkpoint tensors, allowing offload initialization without first materializing both models on NPU. |
| Preview sink attention | Existing Ascend packed FA returns output and fp32 LSE. Apply `output * sigmoid(LSE - sink)` using the live Ulysses-local sink parameters. The preview uses full attention; finite windows remain explicitly unsupported. |
| Preview MoE | Dispatch NPU tensors before importing the CUDA/Triton runner. Use `npu_moe_init_routing_v2`, two `npu_grouped_matmul` calls, the existing native PyTorch `swiglu7_interleaved`, then restore token/top-k order and apply routing weights once. |
| Refiner block-grid attention | Replace CUDA MagiAttention/FlexAttention execution on NPU with native `npu_fused_infer_attention_score`. Pack query blocks with their permitted neighboring video keys and the global tail. Tail/padding queries see all valid keys; padded keys are excluded. GQA and the existing Ulysses collectives are preserved. |

The `sgl_moe_align_block_size` NameError came from entering the Triton MoE runner
on NPU. Its guarded import does not include NPU, and the imported implementation
would still select `sgl_kernel` / CUDA kernels. Adding `_is_npu` to that import
condition would not port the remaining runner. MAGI-2 now bypasses that entire
runner on NPU; the shared CUDA runner and its imports are unchanged.

No new custom kernels were added. mHC already uses its native PyTorch branch on
NPU and remains unchanged. Native routing and grouped-matmul contracts were
checked against the local op-plugin documentation:

- `torch_npu-npu_moe_init_routing_v2.md`: dropless, unquantized routing;
  `row_idx_type=0` gives the inverse permutation, and
  `expert_tokens_num_type=0` gives cumulative group sizes.
- `torch_npu-npu_grouped_matmul.md`: A2/A3 bf16, a single packed activation
  tensor and transposed 3D expert-weight view, `split_item=2`, `group_type=0`,
  `group_list_type=0`; up to 1024 groups.
- `torch_npu-npu_fused_infer_attention_score.md`: packed TND attention with
  separate Q/KV cumulative lengths and GQA.

Those files are under
`/home/amironov/workspace/agent/op-plugin/docs/zh/custom_APIs/torch_npu/`,
checkout `2022ac6a96a50ab6c18c360c2e948f63eebd0dfb`.
They establish the API contract, not execution evidence on a particular device.

## Memory and numerical behavior

MoE processes at most 4096 input rows per chunk, bounding top-k-expanded
activations. It preserves interleaved gate/up weights, alpha=1.702, limit=7,
the `(up + 1)` term, and existing head/expert sharding. GMM uses transposed
views of the current parameters, without retaining copies across offload.
Native GMM rounds its first output to bf16 before activation; the CUDA fused
GEMM/activation can round later. Bitwise CUDA parity is therefore not claimed.

Refiner video batches contain at most 16 query blocks and normally at most
131072 gathered KV tokens. A single neighborhood larger than that budget is
processed alone. Dense tail queries form a separate batch and reuse all valid
KV. The plan caches CPU range metadata; it does not cache gathered KV or create
an S×S mask. Extra launches and duplicated neighbor reads make this a
compatibility implementation, not a performance claim.

None of these bounds guarantees a 64 GB whole-process peak. VAE decode,
prefetch buffers, HCCL buffers, other activations, and allocator reserve must
still be measured on the target machine.

## Full-resolution launch and image extraction

Run from this checkout with its source installed/importable. This retains the
full refiner and default sampling steps. Eager mode is explicit for the first
compatibility run. The model's supported output is a fixed 10-second clip;
`--num-frames 1` is rejected by this branch.

```bash
export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset SGLANG_CACHE_DIT_ENABLED

sglang generate \
  --model-path /home/amironov/models/MAGI-2-preview \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode memory \
  --dit-layerwise-offload \
  --dit-offload-prefetch-size 1 \
  --enable-torch-compile false \
  --width 1920 --height 1088 \
  --prompt "A fox running through a snowy forest" \
  --output-path outputs \
  --output-file-name magi2_npu.mp4 \
  --save-output

# Run after generation succeeds; requires ffmpeg.
ffmpeg -i outputs/magi2_npu.mp4 -frames:v 1 -update 1 outputs/magi2_npu.png
```

## Validation

Local source baseline: SGLang `94c3bf08db7cf7a34a02ef0b2631184954b946d0`.
The target versions above were supplied by the user. Locally, torch is
`2.10.0+cpu` and torch_npu is `2.10.0.post4`, but `libhccl.so`, CANN,
NPU devices and the checkpoint are absent. Full pytest collection is also
blocked by missing application dependencies, starting with `orjson`.
No dependencies were upgraded.

Isolated CPU checks execute the production Python bodies with import dependencies
isolated and external NPU operators emulated. They cover:

- sink correction, ragged/empty sequences, four-way head slicing, live parameter
  rebinding, window rejection, and the existing unsupported-backend rejection;
- initial CPU loading, with the placement regression failing on original code;
- MoE against a per-token/per-expert oracle, including empty experts, top-k
  inverse permutation, activation clipping/interleaving, weighted reduction,
  multiple chunks, an uneven final chunk, and empty input;
- refiner against an independent dense 3D-grid oracle, including ragged blocks,
  multiple packed batches, GQA, no-tail cases, tail interactions and padded keys.

The six isolated CPU checks pass. This does not validate native CANN operators
or four-rank communication. Three actual NPU parity tests were added for sink
attention, MoE forward, and refiner forward; they have not run here.

Run these in the target environment before the full generation:

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_magi2.py \
  -k 'SinkAttentionBackend or LoaderPrecision or Magi2Npu' -v
```

Then run the full command and measure all ranks over loading, denoising,
refinement, and decode. Record exact device/CANN/torch_npu versions, output
quality, allocated/reserved peaks, and device-reported memory. Keep startup
inside the measurement window. Further kernel work is needed only if native
operator validation reveals a missing capability or later profiling justifies
optimization.
