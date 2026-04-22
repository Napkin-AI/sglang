import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# Import to use torch.ops.attentions, install package with sgl_kernel_npu
try:
    import attentions  # noqa: F401
except ImportError as e:
    raise ImportError(
        (
            "The required 'attentions' package is not installed."
            "The package can be installed with sgl_kernel_npu"
        )
    ) from e

logger = init_logger(__name__)


class LaserAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.LASER_ATTN

    @staticmethod
    def get_impl_cls() -> type["LaserAttentionImpl"]:
        return LaserAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError("LA do not have special metadata builder.")


class LaserAttentionImpl(AttentionImpl):

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
        self.softmax_scale = softmax_scale

        self.seqlen_base = 256
        self.seqlen_index = -2
        self.dim_index = -1
        self.dim_base = 128
        self.max_token = 2**31 - 1
        self.seq_len_pad_base = 256

    def _pad(self, input_tensor: torch.Tensor, base: int, dim: int) -> torch.Tensor:

        shape_value = input_tensor.size(dim)
        if shape_value % base != 0:
            pad_size = ((shape_value // base) + 1) * base - shape_value
            padding_shape = list(input_tensor.shape)
            padding_shape[dim] = pad_size
            padding = torch.zeros(
                padding_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )
            return torch.cat([input_tensor, padding], dim=dim)

        return input_tensor

    def _la_preprocess_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        if q.dtype != torch.float16:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        q = self._pad(q, self.seqlen_base, self.seqlen_index)
        q = self._pad(q, self.dim_base, self.dim_index)

        k = self._pad(k, self.seqlen_base, self.seqlen_index)
        k = self._pad(k, self.dim_base, self.dim_index)

        v = self._pad(v, self.seqlen_base, self.seqlen_index)
        v = self._pad(v, self.dim_base, self.dim_index)

        return q, k, v

    def _la_postprocess_output(
        self,
        attention_out: torch.Tensor,
        dtype: torch.dtype,
        qseqlen: int,
        head_dim: int,
    ) -> torch.Tensor:
        if dtype != attention_out.dtype:
            attention_out = attention_out.to(dtype)

        attention_out = attention_out[:, :, :qseqlen, :head_dim]
        attention_out = attention_out.transpose(1, 2).contiguous()
        return attention_out

    def _laser_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        head_num: int,
        pre_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return getattr(torch.ops.attentions, "la")(
            query=query,
            key=key,
            value=value,
            atten_mask=None,
            alibi_mask=None,
            drop_mask=None,
            scale_value=self.softmax_scale,
            head_num=head_num,
            input_layout="BNSD",
            keep_prob=1.0,
            pre_tokens=pre_tokens,
            next_tokens=1,
            is_highPrecision=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        q_seqlen, head_dim = query.shape[1], query.shape[3]
        kv_seqlen = key.shape[1]

        pre_tokens = self.max_token
        if kv_seqlen % self.seq_len_pad_base != 0:
            pre_tokens = (
                kv_seqlen // self.seq_len_pad_base + 1
            ) * self.seq_len_pad_base - kv_seqlen

        q, k, v = self._la_preprocess_input(query, key, value)
        _, la_output = self._laser_attention(q, k, v, q.shape[1], pre_tokens)
        output = self._la_postprocess_output(la_output, query.dtype, q_seqlen, head_dim)

        return output
