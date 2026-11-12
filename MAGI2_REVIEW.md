# MAGI-2 Ascend patch review

Reviewed `prompt.txt`, the staged EP6/EP12 patch, and its BF16 NPU implementation
on top of commit `3812695b76`. The EP6/EP12 partitioning math passes the CPU
checks below. Full generation, native NPU operator correctness, HCCL, and a
sub-64 GB device peak remain unverified here.

## Findings

1. **Quantization requests could silently remain unquantized — fixed.**
   MAGI-2's native loader does not construct quantized linear modules or apply
   generic diffusion quantization after loading. Its expert path explicitly
   converts weights to bf16 before grouped matmul. The pipeline now rejects
   global and component-specific quantization requests, with a regression test,
   rather than accepting an INT8 configuration that leaves BF16 weights.
2. **PR #44738 is insufficient for MAGI-2 W8A8.** The PR description explicitly
   excludes MoE. Its diffusion quantization adapter only handles `LinearBase`;
   MAGI-2 uses custom `Magi2ModalityLinear`, ordinary `nn.Linear`, and raw 3D
   expert parameters loaded through `load_magi2_modules`. A merge alone does
   not connect those weights or their execution to W8A8. No merge was performed.
3. **EP6/EP12 refiner padding is mathematically consistent.** Padding 8 KV
   heads to 12 and 32 Q heads to 48 preserves four Q heads per KV head. Equal
   head splits then produce 8 Q / 2 KV per rank at degree 6 and 4 Q / 1 KV at
   degree 12. Trimming after the inverse exchange preserves collective shapes
   and restores the checkpoint's 32 output heads. Sequence-shard padding is
   included in the grid's excluded key region by the existing refiner code.
4. **The patch does not prove the memory target.** EP only shards the routed
   expert weights. Dense preview parameters, the refiner, VAEs, activations,
   collective buffers, and allocator reserve remain additional costs. The
   staged launch keeps component offload and is appropriate as an initial
   target-machine experiment, not a validated memory or quality result.

PR examined: [online W8A8 INT8 diffusion #44738](https://github.com/sgl-project/sglang/pull/44738),
files view showing head `917f47a` at review time. Relevant source is
[`w8a8_int_npu.py`](https://github.com/sgl-project/sglang/blob/917f47a/python/sglang/multimodal_gen/runtime/layers/quantization/w8a8_int_npu.py),
especially its `LinearBase` restriction. The local MAGI loader and model do not
invoke that adapter.

## Validation performed

- Eight isolated CPU checks pass; two target-NPU checks skip.
- The existing head-padding simulation exercises degrees 1, 2, 4, 6, 8, and 12
  against dense SDPA and an emulated native packed-FA backend.
- Added a sequence-shard/inverse-exchange check at degrees 2, 4, 6, and 12.
  It replays the actual `_usp_input_all_to_all` and
  `_usp_output_all_to_all` layout helpers against an emulated equal-split
  collective, then compares each rank's output with a dense reference.
  This covers sequence order, head regrouping, output trimming and invalid
  SP pad keys, beyond the existing head-only simulation.
- Loader-independent MoE and block-grid numerical checks also pass in the
  selected review suite, with external native operators emulated.
- The quantization rejection test fails on the reviewed baseline for all
  three cases: global quantization, primary component, and refiner component.
- The existing EP padding test fails in four subcases against the baseline
  without head padding, and passes with the staged patch.
- Python syntax, Ruff's selected correctness rules, formatting/import sorting,
  and diff whitespace checks pass.

These are isolated executions of production Python bodies, with unavailable
application imports separated from the numerical checks. They are not a full
package test run. Full pytest collection stops at missing `orjson`. This
workspace has CPU PyTorch, no usable CANN/HCCL installation, no NPU devices and
no model checkpoint. Native CANN and HCCL execution therefore could not run.
The temporary review harness and log are `/tmp/review_magi2_patch.py` and
`/tmp/review_magi2_patch.log`.

On the target machine, run the normal package tests before generation:

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_magi2.py \
  -k 'Magi2Npu or ServerArgsValidation' -v
```

Then use the EP6/EP12 launch in the cookbook and measure the full startup,
preview, refiner, and decode on every rank. A two-step preview is an execution
smoke test; return to the default step count for quality assessment.

## Memory decision

Using the default 36 routed MoE layers, 12 MoE heads, 256 experts per head,
head width 256, intermediate width 1280, and bf16 weights:

`routed bytes per rank = 36 * (12 / EP) * 256 * 3 * 256 * 1280 * 2`

| Configuration | Routed expert weights per rank only |
| --- | ---: |
| BF16 EP4 | 50.625 GiB |
| BF16 EP6 | 33.750 GiB |
| BF16 EP12 | 16.875 GiB |
| Hypothetical INT8 EP4 | 25.3125 GiB plus scales |

The first three describe the current weight layout. The INT8 row is a storage
calculation, not an implemented or measured mode. EP6/EP12 does not require
PR #44738. If restricted to four NPUs and BF16 offload remains too slow or
large, expert INT8 integration is the next memory feature to implement.

## W8A8 integration plan

1. Reuse PR #44738's registration/configuration and NPU quantization primitives,
   but add explicit MAGI-native loader integration. Dense `LinearBase` support
   alone does not cover the dominant expert tensors.
2. After EP slicing, quantize `w13_weight[E,2I,H]` and `w2[E,H,I]` per expert
   output channel, storing INT8 weights and channel scales. Preserve the
   interleaved gate/up layout. Avoid first loading the whole BF16 component on
   NPU; keep loading peaks bounded as well as steady-state storage.
3. Replace `_run_heads_npu`'s bf16 weight conversion with an explicit quantized
   branch. Route tokens, dynamically quantize sorted activations, invoke INT8
   grouped matmul with weight and token scales, run the same clipped
   SwiGLU-OAI in floating point, dynamically quantize again for the down GEMM,
   and restore routing order with weights applied once. Keep router and mHC
   precision unchanged initially.
4. Include generated scales in checkpoint-coverage, offload, and reload
   handling. Check both NPU generations against the actual CANN stack; the
   generic dense `npu_quant_matmul` path is not the grouped-expert path.
5. Validate quantization/dequantization, expert/head slicing, empty experts,
   routing permutations, activation clipping, and repeated offload/reload.
   Then compare full video/audio quality, measured startup/peak memory and
   timing against BF16. Enable the quantization flag only once this branch
   actually loads and executes quantized weights.
