# MAGI-2 on Ascend A2/A3: expert parallelism and W8A8

These commands use the MAGI-2 branch with refiner GQA head padding. Run them
from this checkout, with its source installed/importable by `sglang`.
They retain both DiTs, the full 1920×1088 output, and layerwise offload.

MAGI-2 reads `ep_size` from a pipeline JSON; it does not expose a dedicated
MAGI-2 `--ep-size` option. Set `--num-gpus` and `--ulysses-degree` to the actual
number of selected NPU devices. The examples make `ep_size` explicit so an
older EP4 pipeline configuration cannot accidentally keep the experts at EP4.

| Launch | Visible devices required | `--num-gpus` | `--ulysses-degree` | JSON `ep_size` |
| --- | ---: | ---: | ---: | ---: |
| EP6 | 6 | 6 | 6 | 6 |
| EP12 | 12 | 12 | 12 | 12 |

Your previous selection `12,13,14,15` contains only four devices. EP6 requires
six visible devices and EP12 requires twelve. Replace the example device IDs
below with devices available on your host. If the host has 16 NPUs, select six
or twelve of them; `--num-gpus 16` is invalid for this model because 16 does
not divide its 24 preview attention heads.

## EP6: six NPUs

```bash
cd /home/amironov/workspace/sglang/fork_xnapkin/sglang

cat > magi2-ep6.json <<'JSON'
{
  "ep_size": 6,
  "enable_refiner": true
}
JSON

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset SGLANG_CACHE_DIT_ENABLED

sglang generate \
  --model-path /home/amironov/models/MAGI-2-preview \
  --pipeline-config-path magi2-ep6.json \
  --num-gpus 6 \
  --ulysses-degree 6 \
  --performance-mode memory \
  --dit-layerwise-offload \
  --dit-offload-prefetch-size 1 \
  --enable-torch-compile false \
  --width 1920 \
  --height 1088 \
  --prompt "A fox running through a snowy forest" \
  --output-path outputs \
  --output-file-name magi2_ep6.mp4 \
  --save-output
```

The startup log should contain `[magi2] expert parallel: world=6 ep=6`.

## EP12: twelve NPUs

```bash
cd /home/amironov/workspace/sglang/fork_xnapkin/sglang

cat > magi2-ep12.json <<'JSON'
{
  "ep_size": 12,
  "enable_refiner": true
}
JSON

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset SGLANG_CACHE_DIT_ENABLED

sglang generate \
  --model-path /home/amironov/models/MAGI-2-preview \
  --pipeline-config-path magi2-ep12.json \
  --num-gpus 12 \
  --ulysses-degree 12 \
  --performance-mode memory \
  --dit-layerwise-offload \
  --dit-offload-prefetch-size 1 \
  --enable-torch-compile false \
  --width 1920 \
  --height 1088 \
  --prompt "A fox running through a snowy forest" \
  --output-path outputs \
  --output-file-name magi2_ep12.mp4 \
  --save-output
```

The startup log should contain `[magi2] expert parallel: world=12 ep=12`.

These are BF16 launches with the default sampling steps. They do not enable
INT8 quantization. Routed expert weights alone are approximately 33.750 GiB
per rank at EP6 and 16.875 GiB at EP12; the whole-launch 64 GB peak still needs
measurement on the target hardware.

## W8A8: four NPUs, including your original device selection

This uses online INT8 weights and per-token INT8 activations in both DiTs'
modality projections and in the preview's routed experts. FP32-sensitive
modules, the text encoder and VAEs retain their original precision. EP defaults
to 4 here, so no pipeline JSON is needed.

```bash
cd /home/amironov/workspace/sglang/fork_xnapkin/sglang

export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset SGLANG_CACHE_DIT_ENABLED

sglang generate \
  --model-path /home/amironov/models/MAGI-2-preview \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --quantization w8a8 \
  --performance-mode memory \
  --dit-layerwise-offload \
  --dit-offload-prefetch-size 1 \
  --enable-torch-compile false \
  --width 1920 --height 1088 \
  --prompt "A fox running through a snowy forest" \
  --output-path outputs \
  --output-file-name magi2_w8a8_ep4.mp4 \
  --save-output
```

The loader logs `W8A8 quantizes ... local weights on CPU before device placement`
for each selected DiT. To quantize only the preview DiT, use
`--component-quantizations.transformer w8a8` instead of `--quantization w8a8`.
You can also add the global quantization flag to the EP6 or EP12 commands above.
FSDP is unsupported for this native quantized path.

The routed expert storage at EP4 becomes approximately 25.3125 GiB plus channel
scales. Full-generation memory and output quality still require validation on
the target NPU stack; this is not a measured 64 GB guarantee.
