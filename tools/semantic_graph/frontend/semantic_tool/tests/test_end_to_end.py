"""Generic adversarial capture -> lower -> interpreter -> scalarization fixtures.

Functions below are inspected, tiny CPU arithmetic probes. Source execution is
an acceptance oracle only, outside the independent semantic interpreter.
"""
from copy import deepcopy
import unittest
import torch

from semantic_tool.capture import capture, CaptureError, UntrustedCallableError
from semantic_tool.core import UnsupportedOperator, validate_graph
from semantic_tool.evaluate import evaluate
from semantic_tool.semantics import lower_capture
from semantic_tool.scalarize import instantiate


class EndToEndFixtures(unittest.TestCase):
    def check(self, function, *args):
        before = [x.clone() if isinstance(x, torch.Tensor) else x for x in args]
        expected = function(*args)
        names = [f"input{j}" for j in range(len(args))]
        graph = lower_capture(capture(function, args, names, trusted=True))
        validate_graph(graph)
        result = evaluate(graph, dict(zip(names, args)))
        self.compare(result, expected)
        for value, old in zip(args, before):
            self.compare(value, old)
        scalar = instantiate(graph, max_nodes=5000, max_edges=200000)
        self.assertEqual(scalar["format"], "scalar-dependence-v1")
        return graph, scalar

    def compare(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        elif isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, (list, tuple))
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.compare(a, e)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.compare(actual[key], expected[key])
        else:
            self.assertEqual(actual, expected)

    def test_multiaxis_broadcast_python_and_zero_dim_scalar(self):
        def fn(x, y, scalar, coefficient):
            return (x + y) * scalar + coefficient
        graph, _ = self.check(fn, torch.arange(6.0).reshape(2, 1, 3),
                              torch.arange(4.0).reshape(1, 4, 1), torch.tensor(0.25), 0.125)
        self.assertEqual(graph["inputs"][2]["kind"], "tensor")
        self.assertEqual(graph["inputs"][2]["shape"], [])
        self.assertEqual(graph["inputs"][3]["kind"], "scalar")

    def test_negative_nonadjacent_sum_axes(self):
        def fn(x):
            return torch.sum(x, dim=(-1, 0), keepdim=True)
        graph, _ = self.check(fn, torch.arange(24.0).reshape(2, 3, 4))
        self.assertEqual(graph["nodes"][-1]["attributes"]["axes"], [0, 2])

    def test_mean_amax_and_keepdim_false(self):
        def fn(x):
            return {"mean": x.mean((0, -1)), "max": torch.amax(x, dim=(0, -1))}
        self.check(fn, torch.arange(24.0).reshape(2, 3, 4) / 13)

    def test_bfloat16_cast_roundtrip_boundary(self):
        def fn(x):
            rounded = x.to(torch.bfloat16)
            return rounded, rounded.to(torch.float32) - x
        self.check(fn, torch.tensor([1.001, -2.003, 3.141592, 0.0031]))

    def test_float16_softmax_float32_boundary(self):
        def fn(x):
            return torch.softmax(x, dim=-1, dtype=torch.float32).to(torch.float16)
        self.check(fn, torch.tensor([[1000, 1001, 999], [-1000, -999, -1001]], dtype=torch.float16))

    def test_transpose_noncontiguous_reshape(self):
        def fn(x):
            y = x.transpose(0, 2)
            return y, y.reshape(6, 4), y.contiguous()
        self.check(fn, torch.arange(24.0).reshape(2, 4, 3))

    def test_noncontiguous_input_select_slice(self):
        def fn(x):
            return x[:, -1, 1::2], x.permute(2, 0, 1)
        self.check(fn, torch.arange(60.0).reshape(3, 4, 5).transpose(0, 2))

    def test_batched_broadcast_matmul(self):
        def fn(x, y):
            return torch.matmul(x, y)
        self.check(fn, torch.arange(12.0).reshape(2, 1, 2, 3),
                   torch.arange(18.0).reshape(1, 3, 3, 2))

    def test_zero_dim_reduction(self):
        def fn(x):
            return x.sum(), x.mean(-1), x.amax(0)
        self.check(fn, torch.tensor(1.25))

    def test_functional_subslice_update(self):
        def fn(x, update):
            y = x.clone()
            y[1:4:2, 1:5:2] = update
            return x, y, y.sum()
        graph, _ = self.check(fn, torch.arange(20.0).reshape(4, 5), torch.tensor([[11., 12.], [13., 14.]]))
        self.assertIn("slice_scatter", {n["op"] for n in graph["nodes"]})

    def test_functional_selected_update(self):
        def fn(x, row):
            y = x.clone()
            y[-1] = row
            return y, x
        graph, _ = self.check(fn, torch.arange(12.0).reshape(3, 4), torch.tensor([4., 3., 2., 1.]))
        self.assertIn("select_scatter", {n["op"] for n in graph["nodes"]})

    def test_empty_tensor_sum(self):
        def fn(x):
            return x.sum(dim=1), x.mean(dim=1)
        self.check(fn, torch.empty(2, 0, 3))

    def test_integer_division_rounding(self):
        def fn(x):
            return torch.div(x, 2, rounding_mode="floor"), torch.div(x, 2, rounding_mode="trunc"), x / 2
        self.check(fn, torch.tensor([-7, -3, 3, 7], dtype=torch.int32))

    def test_rsub_alpha_and_dtype(self):
        def fn(x):
            return torch.rsub(x, 2.0, alpha=0.25)
        self.check(fn, torch.tensor([-2.0, 3.0, 4.0], dtype=torch.bfloat16))

    def test_alias_squeeze_unsqueeze_expand(self):
        def fn(x):
            return x.unsqueeze(-1).expand(2, 3, 2), x.unsqueeze(0).squeeze(0).detach()
        self.check(fn, torch.arange(6.0).reshape(2, 3))

    def test_return_dictionary_reserved_literal_keys(self):
        def fn(x):
            return {"dtype": x + 1}, {"device": x + 2}, {"memory_format": x}, {"ref": x + 3}
        self.check(fn, torch.arange(3.0))

    def test_unsupported_sort_fails_closed(self):
        def fn(x):
            return torch.sort(x)
        captured = capture(fn, (torch.tensor([2., 1.]),), trusted=True)
        with self.assertRaises(UnsupportedOperator):
            lower_capture(captured)

    def test_data_dependent_control_flow_rejected(self):
        def fn(x):
            return x + 1 if x.sum().item() > 0 else x - 1
        with self.assertRaises(CaptureError):
            capture(fn, (torch.ones(3),), trusted=True)

    def test_input_mutation_rejected(self):
        def fn(x):
            x.add_(1)
            return x
        x = torch.ones(3)
        with self.assertRaises(CaptureError):
            capture(fn, (x,), trusted=True)
        torch.testing.assert_close(x, torch.ones(3), rtol=0, atol=0)

    def test_untrusted_callable_not_run(self):
        side_effects = []
        def fn(x):
            side_effects.append("executed")
            return x
        with self.assertRaises(UntrustedCallableError):
            capture(fn, (torch.ones(2),))
        self.assertEqual(side_effects, [])


if __name__ == "__main__":
    unittest.main()
