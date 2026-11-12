# SPDX-License-Identifier: Apache-2.0
"""MAGI-2 native loading and execution helpers for Ascend online W8A8."""

from __future__ import annotations

import fnmatch

import torch
from torch import nn

_WEIGHT_QUANT_ROWS = 1024


def quantize_magi2_weight(
    weight: torch.Tensor, *, compute_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8, with bounded fp32 CPU scratch."""
    if weight.device.type != "cpu":
        raise ValueError(
            "MAGI-2 W8A8 weights must be quantized on CPU before placement"
        )
    if compute_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("MAGI-2 W8A8 requires bf16 or fp16 DiT precision")
    scale_dtype = torch.bfloat16 if compute_dtype == torch.bfloat16 else torch.float32
    rows = weight.reshape(-1, weight.shape[-1])
    quantized = torch.empty(rows.shape, dtype=torch.int8, device="cpu")
    scales = torch.empty(rows.shape[0], dtype=scale_dtype, device="cpu")
    for start in range(0, rows.shape[0], _WEIGHT_QUANT_ROWS):
        stop = min(start + _WEIGHT_QUANT_ROWS, rows.shape[0])
        values = rows[start:stop].to(compute_dtype).float()
        maxima = values.abs().amax(dim=-1)
        scale = torch.where(maxima > 0, maxima / 127.0, torch.ones_like(maxima))
        scale = scale.clamp_min(torch.finfo(scale_dtype).tiny).to(scale_dtype)
        quantized[start:stop] = (
            (values / scale.float()[:, None]).round().clamp(-127, 127).to(torch.int8)
        )
        scales[start:stop] = scale
    return quantized.view(weight.shape), scales.view(weight.shape[:-1])


def prepare_magi2_w8a8(
    model: nn.Module,
    *,
    compute_dtype: torch.dtype,
    fp32_names: set[str],
    ignored_layers: tuple[str, ...] = (),
) -> dict[str, str]:
    """Install meta INT8 parameters and registered scales before state assignment."""
    from sglang.multimodal_gen.runtime.layers.moe_multihead import Magi2MultiHeadExperts
    from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
        Magi2ModalityLinear,
    )

    if compute_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("MAGI-2 W8A8 requires bf16 or fp16 DiT precision")
    scale_dtype = torch.bfloat16 if compute_dtype == torch.bfloat16 else torch.float32
    specs = {}
    for name, module in model.named_modules():
        if any(
            fnmatch.fnmatchcase(name, p) or fnmatch.fnmatchcase(name, p + ".*")
            for p in ignored_layers
        ):
            continue
        if isinstance(module, Magi2ModalityLinear):
            weights = (("weight", "weight_scale"),)
        elif isinstance(module, Magi2MultiHeadExperts):
            weights = (("w13_weight", "w13_scale"), ("w2", "w2_scale"))
        else:
            continue
        for weight_name, scale_name in weights:
            full_name = f"{name}.{weight_name}" if name else weight_name
            if full_name in fp32_names:
                continue
            weight = getattr(module, weight_name)
            setattr(
                module,
                weight_name,
                nn.Parameter(
                    torch.empty_like(weight, dtype=torch.int8), requires_grad=False
                ),
            )
            module.register_parameter(
                scale_name,
                nn.Parameter(
                    torch.empty(
                        weight.shape[:-1], dtype=scale_dtype, device=weight.device
                    ),
                    requires_grad=False,
                ),
            )
            specs[full_name] = f"{name}.{scale_name}" if name else scale_name
            if isinstance(module, Magi2ModalityLinear):
                module._compute_dtype = compute_dtype
    if not specs:
        raise ValueError("MAGI-2 W8A8 selection contains no quantizable weights")
    return specs


def magi2_w8a8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=output_dtype)
    shape = x.shape[:-1]
    quantized, token_scale = torch.ops.npu.npu_dynamic_quant(
        x.reshape(-1, x.shape[-1]).contiguous()
    )
    out = torch.ops.npu.npu_quant_matmul(
        quantized,
        weight.t(),
        scale,
        pertoken_scale=token_scale,
        output_dtype=output_dtype,
    )
    return out.reshape(*shape, weight.shape[0])


def magi2_grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    groups: torch.Tensor,
    *,
    scale: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    kwargs = {}
    if scale is not None:
        x, token_scale = torch.ops.npu.npu_dynamic_quant(x.contiguous())
        kwargs = {
            "scale": [scale],
            "per_token_scale": [token_scale],
            "output_dtype": output_dtype,
        }
    return torch.ops.npu.npu_grouped_matmul(
        x=[x],
        weight=[weight],
        group_list=groups,
        split_item=2,
        group_type=0,
        group_list_type=0,
        **kwargs,
    )[0]
