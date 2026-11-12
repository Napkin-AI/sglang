# SPDX-License-Identifier: Apache-2.0
"""Native-loader W8A8 contracts, with CPU oracles and optional Ascend parity."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.multimodal_gen.configs.pipeline_configs.magi2 import magi2_w8a8_options
from sglang.multimodal_gen.runtime.layers import moe_multihead
from sglang.multimodal_gen.runtime.layers.moe_multihead import Magi2MultiHeadExperts
from sglang.multimodal_gen.runtime.layers.quantization import magi2_w8a8
from sglang.multimodal_gen.runtime.loader.component_loaders import magi2_loader
from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    Magi2ModalityLinear,
    Magi2ModalityRMSNorm,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase


class _TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_adapter = nn.Linear(32, 32, dtype=torch.float32)
        block = nn.Module()
        block.proj = Magi2ModalityLinear(32, 32, num_modality=3)
        block.norm = Magi2ModalityRMSNorm(32)
        block.mlp = nn.Module()
        block.mlp.moe_mlp = nn.Module()
        block.mlp.moe_mlp.experts = Magi2MultiHeadExperts(
            num_heads=2, num_experts_per_head=3, head_dim=32, intermediate_size=32
        )
        self.blocks = nn.ModuleList([block])


def _load_tiny(*, quantize=True, ignored_layers=(), compute_dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(82)
    with torch.device("meta"):
        model = _TinyDiT()
    state = {
        name: torch.randn(t.shape, generator=generator).to(compute_dtype)
        for name, t in model.state_dict().items()
    }
    # Check the real expert fusion and transpose before channel quantization.
    prefix = "blocks.0.mlp.moe_mlp."
    w13 = state.pop(prefix + "experts.w13_weight")
    state[prefix + "W_gate"] = w13[:, ::2].transpose(1, 2).contiguous()
    state[prefix + "W_up"] = w13[:, 1::2].transpose(1, 2).contiguous()
    state[prefix + "W_down"] = (
        state.pop(prefix + "experts.w2").transpose(1, 2).contiguous()
    )
    with (
        mock.patch.object(
            magi2_loader,
            "_read_weight_map",
            return_value={name: "unused" for name in state},
        ),
        mock.patch.object(
            magi2_loader, "_read_shard", side_effect=lambda *a, **kw: dict(state)
        ),
    ):
        model = magi2_loader._load_dit_from_shards(
            model,
            checkpoint_dir="unused",
            renames=magi2_loader._PREVIEW_RENAMES,
            config_mapping={},
            shard=magi2_loader.Magi2ExpertShard(
                ep_size=1, ep_rank=0, num_heads=2, num_experts=3
            ),
            param_dtype=compute_dtype,
            device=torch.device("cpu"),
            max_workers=1,
            quantize=quantize,
            ignored_layers=ignored_layers,
        )
    return model


def _dynamic_quant(x):
    scale = x.float().abs().amax(-1) / 127
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (x.float() / scale[..., None]).round().clamp(-127, 127).to(torch.int8), scale


def _quant_matmul(x, weight, scale, *, pertoken_scale, output_dtype):
    assert x.dtype == weight.dtype == torch.int8
    assert pertoken_scale.dtype == torch.float32
    return (
        (x.float() @ weight.float()) * pertoken_scale[:, None] * scale.float()[None]
    ).to(output_dtype)


def _grouped_matmul(
    *,
    x,
    weight,
    group_list,
    split_item,
    group_type,
    group_list_type,
    scale,
    per_token_scale,
    output_dtype,
):
    assert (split_item, group_type, group_list_type) == (2, 0, 0)
    assert weight[0].dtype == x[0].dtype == torch.int8
    assert scale[0].dtype == (
        torch.bfloat16 if output_dtype == torch.bfloat16 else torch.float32
    )
    start, pieces = 0, []
    for expert, stop in enumerate(group_list.tolist()):
        pieces.append(
            _quant_matmul(
                x[0][start:stop],
                weight[0][expert],
                scale[0][expert],
                pertoken_scale=per_token_scale[0][start:stop],
                output_dtype=output_dtype,
            )
        )
        start = stop
    return [torch.cat(pieces)]


def _routing(x, ids, **kwargs):
    assert kwargs["row_idx_type"] == 0 and kwargs["quant_mode"] == -1
    order = ids.flatten().argsort(stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    groups = torch.bincount(ids.flatten(), minlength=kwargs["expert_num"]).cumsum(0)
    return x[order // ids.shape[1]], inverse.int(), groups, None


@contextlib.contextmanager
def _native_ops():
    with (
        mock.patch.object(
            torch.ops.npu, "npu_dynamic_quant", _dynamic_quant, create=True
        ),
        mock.patch.object(
            torch.ops.npu, "npu_quant_matmul", _quant_matmul, create=True
        ),
        mock.patch.object(
            torch.ops.npu, "npu_grouped_matmul", _grouped_matmul, create=True
        ),
        mock.patch.object(
            torch.ops.npu, "npu_moe_init_routing_v2", _routing, create=True
        ),
    ):
        yield


def _expert_reference(experts, tokens, ids, routing_weights):
    """Each original token/top-k slot independently, without sorting/grouped ops."""
    out = torch.zeros_like(tokens, dtype=torch.float32)
    for row in range(tokens.shape[0]):
        qx, xs = _dynamic_quant(tokens[row : row + 1])
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            gate_up = _quant_matmul(
                qx,
                experts.w13_weight[expert].t(),
                experts.w13_scale[expert],
                pertoken_scale=xs,
                output_dtype=tokens.dtype,
            ).float()
            gate, up = gate_up[..., ::2].clamp(max=7), gate_up[..., 1::2].clamp(-7, 7)
            activation = (gate * torch.sigmoid(1.702 * gate) * (up + 1)).to(
                tokens.dtype
            )
            qa, sa = _dynamic_quant(activation)
            down = _quant_matmul(
                qa,
                experts.w2[expert].t(),
                experts.w2_scale[expert],
                pertoken_scale=sa,
                output_dtype=tokens.dtype,
            )
            out[row] += down[0].float() * routing_weights[row, slot]
    return out.to(tokens.dtype)


class TestMagi2W8A8(CustomTestCase):
    def test_unsupported_precision_fails_before_checkpoint_reads(self):
        with torch.device("meta"):
            model = _TinyDiT()
        with (
            mock.patch.object(
                magi2_loader,
                "_read_weight_map",
                side_effect=AssertionError("must not read"),
            ),
            self.assertRaisesRegex(ValueError, "requires bf16 or fp16"),
        ):
            magi2_loader._load_dit_from_shards(
                model,
                checkpoint_dir="unused",
                renames={},
                config_mapping={},
                shard=None,
                param_dtype=torch.float32,
                device=torch.device("cpu"),
                max_workers=1,
                quantize=True,
            )

    def test_expert_sharding_commutes_with_channel_quantization(self):
        weight = torch.randn(12, 64, 32).bfloat16()
        full_q, full_s = magi2_w8a8.quantize_magi2_weight(
            weight, compute_dtype=torch.bfloat16
        )
        for degree in (2, 3, 4, 6, 12):
            for rank in range(degree):
                shard = magi2_loader.Magi2ExpertShard(
                    ep_size=degree, ep_rank=rank, num_heads=12, num_experts=1
                )
                start, stop = shard.row_range
                local_q, local_s = magi2_w8a8.quantize_magi2_weight(
                    weight[start:stop], compute_dtype=torch.bfloat16
                )
                torch.testing.assert_close(local_q, full_q[start:stop])
                torch.testing.assert_close(local_s, full_s[start:stop])

    def test_channel_quantization_zero_rows_and_rounding_bound(self):
        weight = torch.randn(3, 7, 32).bfloat16()
        weight[0, 0] = 0
        weight[1] *= 64
        with mock.patch.object(magi2_w8a8, "_WEIGHT_QUANT_ROWS", 2):
            quantized, scales = magi2_w8a8.quantize_magi2_weight(
                weight, compute_dtype=torch.bfloat16
            )
        restored = quantized.float() * scales.float()[..., None]
        self.assertEqual(scales.shape, weight.shape[:-1])
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertTrue(bool(torch.isfinite(scales).all()))
        self.assertTrue(bool((scales > 0).all()))
        self.assertTrue(bool((restored[0, 0] == 0).all()))
        # Clamp may add a BF16 scale-rounding error at the largest element.
        self.assertTrue(
            bool(
                (
                    (restored - weight.float()).abs() <= scales.float()[..., None] * 1.1
                ).all()
            )
        )

    def test_native_loader_stores_int8_and_reloads_registered_scales(self):
        quantized, reference = _load_tiny(), _load_tiny(quantize=False)
        qstate = quantized.state_dict()
        self.assertLess(
            sum(t.nbytes for t in qstate.values()),
            sum(t.nbytes for t in reference.state_dict().values()) * 0.65,
        )
        for name, tensor in qstate.items():
            if name.endswith(("w13_weight", ".w2", ".proj.weight")):
                self.assertEqual(tensor.dtype, torch.int8)
            elif name.startswith("pre_adapter.") or name.endswith("norm.weight"):
                torch.testing.assert_close(tensor, reference.state_dict()[name])
                self.assertEqual(tensor.dtype, torch.float32)
        parameters = dict(quantized.named_parameters())
        for suffix in ("w13_scale", "w2_scale"):
            self.assertIn("blocks.0.mlp.moe_mlp.experts." + suffix, parameters)
        self.assertEqual(quantized.blocks[0].proj.compute_dtype, torch.bfloat16)
        quantized.to("meta")
        quantized.load_state_dict(qstate, strict=True, assign=True)
        for name, tensor in quantized.state_dict().items():
            torch.testing.assert_close(tensor, qstate[name])

    def test_ignored_layers_preserve_only_selected_module_subtrees(self):
        model = _load_tiny(ignored_layers=("blocks.0.proj",))
        self.assertEqual(model.blocks[0].proj.weight.dtype, torch.bfloat16)
        self.assertIsNone(model.blocks[0].proj.weight_scale)
        self.assertEqual(model.blocks[0].mlp.moe_mlp.experts.w2.dtype, torch.int8)
        model = _load_tiny(ignored_layers=("blocks.*.mlp",))
        self.assertEqual(model.blocks[0].proj.weight.dtype, torch.int8)
        self.assertEqual(model.blocks[0].mlp.moe_mlp.experts.w2.dtype, torch.bfloat16)

    def test_component_selection_and_exclusions(self):
        args = SimpleNamespace(
            quantization=None,
            component_quantizations={"transformer": "w8a8"},
            quantization_ignored_layers=["blocks.0"],
            component_quantization_ignored_layers={"transformer": ["blocks.1"]},
        )
        self.assertEqual(magi2_w8a8_options(args, "transformer"), (True, ("blocks.1",)))
        self.assertEqual(magi2_w8a8_options(args, "transformer_2"), (False, ()))
        args.quantization = "w8a8"
        self.assertEqual(
            magi2_w8a8_options(args, "transformer_2"), (True, ("blocks.0",))
        )

    def test_modality_projection_executes_int8_and_uses_live_parameters(self):
        model = _load_tiny()
        proj = model.blocks[0].proj
        x = torch.randn(7, 32)
        ids = torch.tensor([2, 2, 0, 1, 1, 2, 0])
        with _native_ops():
            actual = proj(x, ids)
            expected = torch.cat(
                [
                    magi2_w8a8.magi2_w8a8_linear(
                        x[i : i + 1].bfloat16(),
                        proj.weight.view(3, 32, 32)[int(ids[i])],
                        proj.weight_scale.view(3, 32)[int(ids[i])],
                        output_dtype=torch.bfloat16,
                    )
                    for i in range(7)
                ]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            before = actual.clone()
            proj.weight = nn.Parameter(
                torch.zeros_like(proj.weight), requires_grad=False
            )
            torch.testing.assert_close(proj(x, ids), torch.zeros_like(before))
            self.assertGreater(before.abs().max().item(), 0)

    def test_expert_quantization_sorting_and_multiple_chunks(self):
        for dtype in (torch.bfloat16, torch.float16):
            experts = _load_tiny(compute_dtype=dtype).blocks[0].mlp.moe_mlp.experts
            tokens = (torch.randn(11, 32) * 3).to(dtype)
            ids = torch.tensor([[5, 1] if i % 2 else [3, 5] for i in range(11)])
            weights = torch.rand(11, 2) * 2
            expected = _expert_reference(experts, tokens, ids, weights)
            with (
                _native_ops(),
                mock.patch.object(moe_multihead, "_NPU_EXPERT_TOKEN_CHUNK", 4),
            ):
                actual = experts._run_heads_npu(
                    tokens,
                    ids,
                    weights,
                    experts.w13_weight,
                    experts.w2,
                    w13_scale=experts.w13_scale,
                    w2_scale=experts.w2_scale,
                )
                empty = experts._run_heads_npu(
                    tokens[:0],
                    ids[:0],
                    weights[:0],
                    experts.w13_weight,
                    experts.w2,
                    w13_scale=experts.w13_scale,
                    w2_scale=experts.w2_scale,
                )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(empty.shape, (0, 32))

    @unittest.skipUnless(current_platform.is_npu(), "requires Ascend NPU")
    def test_native_npu_projection_and_experts(self):
        model = _load_tiny()
        x = torch.randn(12, 32).bfloat16()
        x[0] = 0
        ids = torch.tensor(
            [
                [3 * (row % 2) + (row // 2) % 3, 3 * (row % 2) + ((row // 2) + 1) % 3]
                for row in range(12)
            ]
        )
        weights = torch.rand(12, 2)
        expected = _expert_reference(
            model.blocks[0].mlp.moe_mlp.experts, x, ids, weights
        )
        with _native_ops():
            expected_proj = model.blocks[0].proj(x, torch.tensor([1]))
        model = model.to("npu")
        with torch.no_grad():
            actual_proj = model.blocks[0].proj(
                x.to("npu"), torch.tensor([1], device="npu")
            )
            experts = model.blocks[0].mlp.moe_mlp.experts
            # Exercise scale rebasing/slicing when the head/expert ceiling splits a call.
            experts.heads_per_call = 1
            actual = experts(x.to("npu"), ids.to("npu"), weights.to("npu"))
        torch.testing.assert_close(
            actual_proj.cpu(), expected_proj, rtol=0.03, atol=0.08
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.25)

    @unittest.skipUnless(current_platform.is_npu(), "requires Ascend NPU")
    def test_native_layerwise_offload_reloads_int8_weights_and_scales(self):
        from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
            LayerwiseOffloadManager,
        )

        model = _load_tiny()
        x = torch.randn(7, 32).bfloat16()
        with _native_ops():
            expected = model.blocks[0].proj(x, torch.tensor([2]))
        manager = LayerwiseOffloadManager(
            model,
            layers_attr_str="blocks",
            num_layers=1,
            enabled=True,
            pin_cpu_memory=False,
            prefetch_size=1,
        )
        try:
            for _ in range(3):
                manager.prefetch_layer(0, non_blocking=False)
                torch.npu.synchronize()
                proj = model.blocks[0].proj
                self.assertEqual(proj.weight.dtype, torch.int8)
                self.assertEqual(proj.weight_scale.dtype, torch.bfloat16)
                with torch.no_grad():
                    actual = proj(x.to("npu"), torch.tensor([2], device="npu"))
                torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.08)
                manager.release_layer(0, force=True)
        finally:
            torch.npu.synchronize()
            manager.release_all()
            manager.release_host_stores()


if __name__ == "__main__":
    unittest.main()
