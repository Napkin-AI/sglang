from dataclasses import dataclass
from typing import Any, List, Optional

import math

import torch
from einops import rearrange

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn import (
    LaserAttentionBackend,
)

import attentions

logger = init_logger(__name__)


class RainFusionAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.RAIN_FUSION_ATTN

    @staticmethod
    def get_impl_cls() -> type["RainFusionAttentionImpl"]:
        return RainFusionAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["RainFusionAttentionMetadata"]:
        return RainFusionAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["RainFusionAttentionMetadataBuilder"]:
        return RainFusionAttentionMetadataBuilder


@dataclass
class RainFusionAttentionMetadata(AttentionMetadata):
    current_timestep: int
    skip_first_steps: int
    sparsity: float
    latent_shape: list[int]


class RainFusionAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(
        self,
        current_timestep: int,
        skip_first_steps: int,
        sparsity: float,
        raw_latent_shape: list[int],
        patch_size: tuple[int, int, int],
        **kwargs: dict[str, Any],
    ) -> RainFusionAttentionMetadata:
        if not (skip_first_steps >= 0 and 0.0 <= sparsity < 1.0):
            raise ValueError(
                (
                    "Invalid attention metadata values."
                    f"Sparsity should be in [0, 1), skip_first_steps should be non-negative."
                    f"Got sparsity={sparsity}, skip_first_steps={skip_first_steps}"
                )
            )

        if sparsity == 0.0:
            logger.warning(
                (
                    "Sparsity is set to 0.0, which means no tokens will be dropped."
                    "For better perfomance use Laser Attention or increase sparsity."
                )
            )

        latent_shape = raw_latent_shape[-3:]
        latent_shape = [latent_shape[i] // patch_size[i] for i in range(3)]

        return RainFusionAttentionMetadata(
            current_timestep=current_timestep,
            skip_first_steps=skip_first_steps,
            sparsity=sparsity,
            latent_shape=latent_shape,
        )


class RainFusionAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.block_size = 128

        self.laser_attn_impl = LaserAttentionBackend.get_impl_cls()(
            num_heads,
            head_size,
            causal,
            softmax_scale,
            num_kv_heads,
            prefix,
            **extra_impl_args,
        )

    def _avgpool(
        self, input_tensor: torch.Tensor, pool_size: int = 128
    ) -> torch.Tensor:
        batch, seqlen, headnum, dim = input_tensor.shape

        num_full_blocks = seqlen // pool_size
        tail_size = seqlen % pool_size

        if num_full_blocks > 0:
            full_blocks = input_tensor[:, : num_full_blocks * pool_size, :, :]
            full_blocks_reshaped = full_blocks.view(
                batch, num_full_blocks, pool_size, headnum, dim
            )
            full_pooled = full_blocks_reshaped.mean(dim=2)
        else:
            full_pooled = torch.empty(0, device=input_tensor.device)
        if tail_size > 0:
            tail_block = input_tensor[:, num_full_blocks * pool_size :, :, :]
            tail_reshaped = tail_block.view(batch, 1, tail_size, headnum, dim)
            tail_pooled = tail_reshaped.mean(dim=2)
        else:
            tail_pooled = torch.empty(0, device=input_tensor.device)

        if num_full_blocks > 0 and tail_size > 0:
            output_tensor = torch.cat([full_pooled, tail_pooled], dim=1)
        elif num_full_blocks > 0:
            output_tensor = full_pooled
        else:
            output_tensor = tail_pooled

        return output_tensor

    def _get_mask_index(self, mask: torch.Tensor) -> torch.Tensor:
        b, n, s, _ = mask.shape
        device = mask.device

        mask_reshaped = mask.reshape(-1, s, s)
        batch_size = mask_reshaped.shape[0]

        row_indices = torch.arange(s, device=device).expand(batch_size, s, -1)
        sorted_vals = torch.where(mask_reshaped, row_indices, 1e9).to(torch.float32)
        sorted_vals, _ = torch.sort(sorted_vals, dim=-1)
        valid_count = mask_reshaped.sum(dim=-1, keepdim=True)
        keep_mask = row_indices < valid_count
        result = torch.where(keep_mask, sorted_vals, -1)

        pos_matrix = result.reshape(b, n, s, s).to(torch.int64)
        return pos_matrix

    def _get_blockwise_mask(
        self,
        qkv_pool: torch.Tensor,
        text_len: int,
        sparsity: float,
        scale: float,
        pool_size: int,
        latent_shape: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tq, hq, wq = latent_shape
        first_frame_len = hq * wq

        query_pool, key_pool, value_pool = torch.chunk(qkv_pool, 3, dim=0)
        attn_scores_head = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
        score_matrix = torch.nn.functional.softmax(attn_scores_head, dim=-1)

        cols = score_matrix.shape[-1]

        keep_len = math.ceil(cols * (1 - sparsity))
        topk_values, _ = torch.topk(score_matrix, k=keep_len, dim=-1)
        thresholds = topk_values[..., -1:]
        mask = score_matrix >= thresholds
        text_block_num = (text_len + pool_size - 1) // pool_size

        if text_block_num > 0:
            mask[:, :, -text_block_num:, :] = True
            mask[:, :, :, -text_block_num:] = True

        firstframe_block_num = (first_frame_len + pool_size - 1) // pool_size
        if firstframe_block_num > 0:
            mask[:, :, :firstframe_block_num, :] = True
            mask[:, :, :, :firstframe_block_num] = True
        select_idx = self._get_mask_index(mask)
        select_idx = select_idx[0].transpose(0, 1)
        select_num_idx = mask[0].transpose(0, 1).sum(dim=-1)
        return select_idx, select_num_idx

    def _rearrange_with_remaining(
        self, tensor: torch.Tensor, latent_shape: tuple[int, int, int]
    ) -> torch.Tensor:
        """
        b (f hn hb wn wb) n d -> b (f hn wn hb wb) n d
        or
        b n (f hn hb wn wb) d -> b n (f hn wn hb wb) d
        """
        tq, hq, wq = latent_shape
        first_frame_len, frame_num = hq * wq, tq

        b, s, n, d = tensor.shape

        if (hq % 8 != 0) or (wq % 8 != 0):
            tensor_first = tensor[:, :first_frame_len, :, :]
            tensor = tensor[:, first_frame_len:, :, :]
            tensor_hwt = rearrange(
                tensor, "b (f h w) n d -> b f h w n d", f=frame_num - 1, h=hq, w=wq
            )
            if hq % 8 != 0:
                tensor_hwt, tensor_h_r = torch.split(tensor_hwt, hq - (hq % 8), dim=2)
                tensor_h_r = tensor_h_r.reshape(b, frame_num - 1, -1, n, d)
            if wq % 8 != 0:
                tensor_hwt, tensor_w_r = torch.split(tensor_hwt, wq - (wq % 8), dim=3)
                tensor_w_r = tensor_w_r.reshape(b, frame_num - 1, -1, n, d)
            tensor_hwt = rearrange(
                tensor_hwt,
                "b f (hn hb) (wn wb) n d -> b f (hn wn hb wb) n d",
                f=frame_num - 1,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )
            if hq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=2)
            if wq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=2)
            tensor_hwt = tensor_hwt.reshape(b, -1, n, d)
            tensor_hwt = torch.cat([tensor_first, tensor_hwt], dim=1)
        else:
            tensor_hwt = rearrange(
                tensor,
                "b (f hn hb wn wb) n d -> b (f hn wn hb wb) n d",
                f=frame_num,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )

        return tensor_hwt

    def _inv_rearrange_with_remaining(
        self, tensor: torch.Tensor, latent_shape: tuple[int, int, int]
    ) -> torch.Tensor:
        tq, hq, wq = latent_shape
        first_frame_len, frame_num = hq * wq, tq

        b, s, n, d = tensor.shape

        if (hq % 8 != 0) or (wq % 8 != 0):
            tensor_first = tensor[:, :first_frame_len, :, :]
            tensor = tensor[:, first_frame_len:, :, :]
            tensor_hwt = rearrange(
                tensor, "b (f h w) n d -> b f h w n d", f=frame_num - 1, h=hq, w=wq
            )
            if hq % 8 != 0:
                tensor_hwt, tensor_h_r = torch.split(tensor_hwt, hq - (hq % 8), dim=2)
            if wq % 8 != 0:
                tensor_hwt, tensor_w_r = torch.split(tensor_hwt, wq - (wq % 8), dim=3)
            tensor_hwt = tensor_hwt.reshape(b, frame_num - 1, -1, n, d)
            tensor_hwt = rearrange(
                tensor_hwt,
                "b f (hn wn hb wb) n d -> b f (hn hb) (wn wb) n d",
                f=frame_num - 1,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )
            if wq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=3)
            if hq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=2)
            tensor_hwt = tensor_hwt.reshape(b, -1, n, d)
            tensor_hwt = torch.cat([tensor_first, tensor_hwt], dim=1)
        else:
            tensor_hwt = rearrange(
                tensor,
                "b (f hn wn hb wb) n h -> b (f hn hb wn wb) n h",
                f=frame_num,
                hb=8,
                wb=8,
                hn=hq // 8,
                wn=wq // 8,
            )

        return tensor_hwt

    def _do_tensor_rearrange_pooling(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        text_len: int,
        pool_size: int,
        latent_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tensor block rearrangement + pooling operation
        """
        tensor = torch.cat((query, key, value), dim=0)
        if text_len != 0:
            tensor_t = tensor[:, :text_len, :, :]
            tensor_i = tensor[:, text_len:, :, :]

            tensor_i_2 = self._rearrange_with_remaining(tensor_i, latent_shape)
            tensor_i_pool = self._avgpool(tensor_i_2, pool_size)
            tensor_t_pool = self._avgpool(tensor_t, pool_size)

            tensor = torch.concat((tensor_i_2, tensor_t), dim=1)
            tensor_pool = torch.concat((tensor_i_pool, tensor_t_pool), dim=1)
        else:
            tensor = self._rearrange_with_remaining(tensor, latent_shape)
            tensor_pool = self._avgpool(tensor, pool_size)

        query_, key_, value_ = torch.chunk(tensor, 3, dim=0)
        return query_, key_, value_, tensor_pool

    def _do_tensor_inv_rearrange(
        self, tensor: torch.Tensor, text_len: int, latent_shape: tuple[int, int, int]
    ):
        if text_len != 0:
            tensor_t = tensor[:, -text_len:, :, :]
            tensor_i = tensor[:, :-text_len, :, :]

            tensor_i = self._inv_rearrange_with_remaining(tensor_i, latent_shape)
            tensor = torch.concat((tensor_t, tensor_i), dim=1)
        else:
            tensor = self._inv_rearrange_with_remaining(tensor, latent_shape)

        return tensor

    def _rain_fusion_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        select_idx: torch.Tensor,
        select_num_idx: torch.Tensor,
        blockshape: List[int],
        scale: float = 1.0,
        head_num: int = 1,
        input_layout: str = "TND",
        actual_seq_lengths=Optional[torch.Tensor],
        actual_seq_lengths_kv=Optional[torch.Tensor],
        inner_precise=0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return getattr(torch.ops.attentions, "rainfusionattention")(
            query=query,
            key=key,
            value=value,
            select_idx=select_idx,
            select_num_idx=select_num_idx,
            blockshape=blockshape,
            attn_mask=None,
            actual_seq_qlen=actual_seq_lengths,
            actual_seq_kvlen=actual_seq_lengths_kv,
            block_table=None,
            q_input_layout=input_layout,
            kv_input_layout=input_layout,
            head_num=head_num,
            mask_type=0,
            scale=scale,
            inner_precise=inner_precise,
            block_size=0,
        )

    def _rain_fusion_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        latent_shape: tuple[int, int, int],
        sparsity: float,
    ):

        inner_precise = 0
        text_len = 0

        q, k, v, qkv_pool = self._do_tensor_rearrange_pooling(
            query, key, value, text_len, self.block_size, latent_shape
        )

        select_idx, select_num_idx = self._get_blockwise_mask(
            qkv_pool,
            text_len,
            sparsity,
            self.softmax_scale,
            self.block_size,
            latent_shape,
        )

        batch_size, seqlen_q, head_num, head_dim = q.shape
        seqlen_kv = k.shape[1]

        layout = "TND"
        q = q.reshape(-1, head_num, head_dim)
        k = k.reshape(-1, head_num, head_dim)
        v = v.reshape(-1, head_num, head_dim)

        actual_seq_lengths = [seqlen_q for _ in range(batch_size)]
        actual_seq_lengths_kv = [seqlen_kv for _ in range(batch_size)]

        out, _ = self._rain_fusion_attention(
            q,
            k,
            v,
            scale=self.softmax_scale,
            head_num=head_num,
            input_layout=layout,
            select_idx=select_idx,
            select_num_idx=select_num_idx,
            blockshape=[self.block_size, self.block_size],
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            inner_precise=inner_precise,
        )

        out = out.reshape(batch_size, seqlen_q, head_num, head_dim)
        out = self._do_tensor_inv_rearrange(out, text_len, latent_shape)
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if attn_metadata.current_timestep < attn_metadata.skip_first_steps:
            output = self.laser_attn_impl.forward(
                query,
                key,
                value,
                attn_metadata,
            )
        else:
            output = self._rain_fusion_sparse_attention(
                query,
                key,
                value,
                attn_metadata.latent_shape,
                attn_metadata.sparsity,
            )
        return output
