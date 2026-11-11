# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
    fused_infer_attention_varlen,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class Magi2SinkAttention(nn.Module):
    """Packed sink attention with Ulysses head sharding on CUDA and Ascend."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        sink_token_num: int = 0,
        window_size: tuple[int, int] = (-1, -1),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.softmax_scale = softmax_scale or head_dim**-0.5
        self.window_size = window_size

        # fp32: compared against attention logits, and the reference avoids bf16 here.
        self.sinks = (
            nn.Parameter(torch.zeros(sink_token_num, num_heads, dtype=torch.float32))
            if sink_token_num
            else None
        )
        self._impl = None

    def _local_sinks(self) -> torch.Tensor | None:
        """``_usp_input_all_to_all_packed_qkv`` gives rank *r* heads ``[r * h_local, (r + 1) * h_local)``."""
        if self.sinks is None:
            return None
        if self.sinks.shape[0] != 1:
            raise ValueError(
                "MAGI-2 attention supports one sink logit per head, so "
                f"sink_token_num must be 1; got {self.sinks.shape[0]}"
            )
        # The checkpoint stores [sink_token_num, num_heads]; the kernel wants flat.
        sinks = self.sinks[0]
        world_size = get_ulysses_parallel_world_size()
        if world_size <= 1:
            return sinks
        h_local = self.num_heads // world_size
        start = get_ulysses_parallel_rank() * h_local
        return sinks[start : start + h_local]

    def _ensure_impl(self, dtype: torch.dtype) -> None:
        if self._impl is not None:
            return
        backend = get_attn_backend(
            self.head_dim,
            dtype,
            supported_attention_backends={AttentionBackendEnum.FA},
            selected_attention_backend=AttentionBackendEnum.FA,
            attention_requirements=AttentionRequirements(packed_varlen=True),
        )
        impl_cls = backend.get_impl_cls()
        # The platform FA resolver can still hand back SDPA, which drops sinks.
        if not issubclass(impl_cls, FlashAttentionImpl):
            raise RuntimeError(
                "MAGI-2 attention needs FlashAttention for its sink logits and "
                f"window, but the platform resolved {backend.get_enum().name}; "
                "run on a GPU with FA support"
            )
        world_size = get_ulysses_parallel_world_size()
        local_heads = self.num_heads // world_size
        local_kv_heads = max(1, self.num_kv_heads // world_size)
        sinks = self._local_sinks()
        self._impl = impl_cls(
            num_heads=local_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.softmax_scale,
            num_kv_heads=local_kv_heads,
            # The parameter stays fp32, but FA3 wants sinks in the q dtype.
            sinks=None if sinks is None else sinks.to(dtype).contiguous(),
            window_size=self.window_size,
        )

    def _forward_npu(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # The preview uses full attention. Do not silently drop a window if a
        # future configuration adds one: the packed Ascend helper is unmasked.
        if self.window_size != (-1, -1):
            raise NotImplementedError(
                "MAGI-2 Ascend sink attention currently supports only full "
                "attention (window_size=(-1, -1))"
            )
        sinks = self._local_sinks()
        result = fused_infer_attention_varlen(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            softmax_scale=self.softmax_scale,
            return_softmax_lse=sinks is not None,
        )
        if sinks is None:
            return result

        out, lse = result  # LSE is [local_heads, packed_tokens], in fp32.
        # The sink has a logit but no value: multiply ordinary attention by
        # Z / (Z + exp(sink)) = sigmoid(log(Z) - sink). This avoids an S x S
        # allocation and does not require a native learnable-sink operator.
        # Read the live parameter each time; layerwise offload rebinds storage.
        factor = torch.sigmoid(lse - sinks.float()[:, None]).transpose(0, 1)
        return (out.float() * factor[..., None]).to(out.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.layers.usp import (
            _usp_input_all_to_all_packed_qkv,
            _usp_output_all_to_all,
        )

        ulysses = get_ulysses_parallel_world_size() > 1
        if ulysses:
            q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)

        if q.device.type == "npu":
            out = self._forward_npu(q, k, v, cu_seqlens=cu_seqlens)
        else:
            self._ensure_impl(q.dtype)
            out = self._impl.forward_varlen(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        if ulysses:
            out = _usp_output_all_to_all(out[None], head_dim=2)[0]
        return out
