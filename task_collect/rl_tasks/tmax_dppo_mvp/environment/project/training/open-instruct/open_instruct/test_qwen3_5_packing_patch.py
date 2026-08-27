import unittest
from unittest.mock import patch

import torch

from open_instruct import qwen3_5_packing_patch


def _set_deepspeed_attr(param, name, value):
    setattr(param, name, value)


class TestQwen35PackingPatch(unittest.TestCase):
    def test_use_torch_causal_conv1d_temporarily_disables_extension(self):
        class DummyGatedDeltaNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.causal_conv1d_fn = object()

        model = torch.nn.Sequential(DummyGatedDeltaNet(), DummyGatedDeltaNet())
        original_functions = [module.causal_conv1d_fn for module in model]

        with (
            patch.object(
                qwen3_5_packing_patch.modeling_qwen3_5,
                "Qwen3_5GatedDeltaNet",
                DummyGatedDeltaNet,
            ),
            qwen3_5_packing_patch.use_torch_causal_conv1d(model),
        ):
            self.assertTrue(all(module.causal_conv1d_fn is None for module in model))

        self.assertEqual([module.causal_conv1d_fn for module in model], original_functions)

    def test_use_torch_causal_conv1d_restores_extension_after_error(self):
        class DummyGatedDeltaNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.causal_conv1d_fn = object()

        model = DummyGatedDeltaNet()
        original_function = model.causal_conv1d_fn

        with (
            patch.object(
                qwen3_5_packing_patch.modeling_qwen3_5,
                "Qwen3_5GatedDeltaNet",
                DummyGatedDeltaNet,
            ),
            self.assertRaisesRegex(RuntimeError, "boom"),
            qwen3_5_packing_patch.use_torch_causal_conv1d(model),
        ):
            raise RuntimeError("boom")

        self.assertIs(model.causal_conv1d_fn, original_function)

    def test_register_zero3_external_conv1d_parameter(self):
        module = torch.nn.Module()
        module.conv1d = torch.nn.Conv1d(4, 4, kernel_size=3, groups=4, bias=False)
        _set_deepspeed_attr(module.conv1d.weight, "ds_id", 0)

        with patch.object(qwen3_5_packing_patch.deepspeed.zero, "register_external_parameter") as register:
            registered = qwen3_5_packing_patch._register_zero3_external_conv1d_parameter(module)

        self.assertTrue(registered)
        register.assert_called_once_with(module, module.conv1d.weight)

    def test_register_zero3_external_conv1d_parameter_is_idempotent(self):
        module = torch.nn.Module()
        module.conv1d = torch.nn.Conv1d(4, 4, kernel_size=3, groups=4, bias=False)
        _set_deepspeed_attr(module.conv1d.weight, "ds_id", 0)

        with patch.object(qwen3_5_packing_patch.deepspeed.zero, "register_external_parameter") as register:
            first = qwen3_5_packing_patch._register_zero3_external_conv1d_parameter(module)
            second = qwen3_5_packing_patch._register_zero3_external_conv1d_parameter(module)

        self.assertTrue(first)
        self.assertFalse(second)
        register.assert_called_once_with(module, module.conv1d.weight)

    def test_register_zero3_external_conv1d_parameter_reregisters_replaced_weight(self):
        module = torch.nn.Module()
        module.conv1d = torch.nn.Conv1d(4, 4, kernel_size=3, groups=4, bias=False)
        _set_deepspeed_attr(module.conv1d.weight, "ds_id", 0)

        with patch.object(qwen3_5_packing_patch.deepspeed.zero, "register_external_parameter") as register:
            qwen3_5_packing_patch._register_zero3_external_conv1d_parameter(module)
            old_weight = module.conv1d.weight
            module.conv1d.weight = torch.nn.Parameter(torch.randn_like(old_weight))
            _set_deepspeed_attr(module.conv1d.weight, "ds_id", 1)

            registered = qwen3_5_packing_patch._register_zero3_external_conv1d_parameter(module)

        self.assertTrue(registered)
        self.assertEqual(register.call_count, 2)

    def test_register_zero3_external_conv1d_parameter_ignores_non_zero3_weight(self):
        module = torch.nn.Module()
        module.conv1d = torch.nn.Conv1d(4, 4, kernel_size=3, groups=4, bias=False)

        with patch.object(qwen3_5_packing_patch.deepspeed.zero, "register_external_parameter") as register:
            registered = qwen3_5_packing_patch._register_zero3_external_conv1d_parameter(module)

        self.assertFalse(registered)
        register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
