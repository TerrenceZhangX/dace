"""Independent synthetic fixtures; no task IDs, source formulas, or GPU claims."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from semantic_tool.evaluate import evaluate, EvaluationError, UnsupportedOperator


def meta(identifier, value):
    if isinstance(value, torch.Tensor):
        return {"id": identifier, "name": identifier, "kind": "tensor", "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."), "strides": list(value.stride())}
    return {"id": identifier, "name": identifier, "kind": "scalar", "dtype": type(value).__name__,
            "specialized_value": value}


def ref(identifier):
    return {"ref": identifier}


class Fixture:
    def __init__(self, **inputs):
        self.inputs = inputs
        self.graph = {"format": "semantic-graph-v1", "inputs": [meta(k, v) for k, v in inputs.items()],
                      "nodes": [], "constants": {}, "outputs": None}

    def node(self, op, args, expected, kwargs=None, attributes=None, identifier=None):
        identifier = identifier or f"n{len(self.graph['nodes'])}"
        self.graph["nodes"].append({"id": identifier, "op": op, "args": args,
                                    "kwargs": kwargs or {}, "attributes": attributes or {},
                                    "outputs": [meta(identifier, expected)]})
        self.graph["outputs"] = ref(identifier)
        return ref(identifier)

    def evaluate(self):
        return evaluate(self.graph, self.inputs)


class EvaluatorFixtures(unittest.TestCase):
    def assertExact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)

    def test_broadcast_and_wrapped_scalars(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.arange(6, dtype=dtype).reshape(2, 1, 3) / 8
                y = torch.tensor([1, -2, 3, 4], dtype=dtype).reshape(1, 4, 1)
                f = Fixture(x=x, y=y)
                a = f.node("add", [ref("x"), ref("y")], torch.add(x, y, alpha=2), {"alpha": 2})
                expected = torch.add(x, y, alpha=2) * 0.125
                f.node("mul", [a, 0.125], expected)
                self.assertExact(f.evaluate(), expected)

    def test_pointwise_whitelist(self):
        x = torch.tensor([0.25, 0.5, 1.5], dtype=torch.float64)
        unary = {"tanh": torch.tanh, "exp": torch.exp, "sigmoid": torch.sigmoid,
                 "sqrt": torch.sqrt, "rsqrt": torch.rsqrt, "neg": torch.neg}
        binary = {"sub": torch.sub, "div": torch.div, "pow": torch.pow, "maximum": torch.maximum}
        for op, fn in unary.items():
            with self.subTest(op=op):
                f = Fixture(x=x)
                f.node(op, [ref("x")], fn(x))
                self.assertExact(f.evaluate(), fn(x))
        y = torch.tensor([2.0, 3.0, 4.0], dtype=x.dtype)
        for op, fn in binary.items():
            with self.subTest(op=op):
                f = Fixture(x=x, y=y)
                f.node(op, [ref("x"), ref("y")], fn(x, y))
                self.assertExact(f.evaluate(), fn(x, y))

    def test_integer_division_and_promotion(self):
        x = torch.tensor([-9, -5, 5, 9], dtype=torch.int32)
        for mode in (None, "floor", "trunc"):
            with self.subTest(mode=mode):
                f = Fixture(x=x)
                expected = torch.div(x, 2, rounding_mode=mode)
                f.node("div", [ref("x"), 2], expected, {"rounding_mode": mode})
                self.assertExact(f.evaluate(), expected)

    def test_cast_preserves_rounding_and_narrowing(self):
        x = torch.tensor([1.0003, -1.0003, 123.456, 0.003001], dtype=torch.float32)
        f = Fixture(x=x)
        half = f.node("cast", [ref("x")], x.to(torch.float16), {"dtype": {"dtype": "float16"}})
        widened = f.node("cast", [half], x.half().float(), attributes={"dtype": "float32"})
        f.node("sub", [widened, ref("x")], x.half().float() - x)
        self.assertExact(f.evaluate(), x.half().float() - x)
        self.assertGreater(torch.count_nonzero(f.evaluate()).item(), 0)

    def test_reductions_multiaxis_negative_keepdim_and_empty(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            x = (torch.arange(24, dtype=dtype).reshape(2, 3, 4) - 10) / 7
            for op, fn in (("sum", torch.sum), ("mean", torch.mean), ("amax", torch.amax)):
                for axes in ([-1, 0], [], None):
                    with self.subTest(dtype=dtype, op=op, axes=axes):
                        expected = fn(x, dim=axes, keepdim=True)
                        f = Fixture(x=x)
                        f.node(op, [ref("x"), axes, True], expected)
                        self.assertExact(f.evaluate(), expected)
        x = torch.empty(2, 0, 3)
        for op, fn in (("sum", torch.sum), ("mean", torch.mean)):
            f = Fixture(x=x)
            f.node(op, [ref("x"), [1], False], fn(x, dim=1))
            self.assertExact(f.evaluate(), fn(x, dim=1))
        x = torch.arange(6, dtype=torch.int32).reshape(2, 3)
        f = Fixture(x=x)
        f.node("sum", [ref("x"), [1]], x.sum(1))
        self.assertExact(f.evaluate(), x.sum(1))

    def test_softmax_stability_dtype_and_half_to_float(self):
        x = torch.tensor([[1000.0, 1001.0, 999.0], [-1000, -1001, -999]])
        f = Fixture(x=x)
        expected = torch.softmax(x, -1, dtype=torch.float64)
        f.node("softmax", [ref("x"), -1, {"dtype": "float64"}], expected)
        self.assertExact(f.evaluate(), expected)
        f = Fixture(x=x.half())
        expected = torch.softmax(x.half(), -1, dtype=torch.float32)
        f.node("softmax", [ref("x"), -1, True], expected)
        self.assertExact(f.evaluate(), expected)
        f = Fixture(x=x)
        f.node("softmax", [ref("x")], torch.softmax(x, 1), attributes={"dim": 1})
        self.assertExact(f.evaluate(), torch.softmax(x, 1))

    def test_contractions_broadcast_vectors_and_noncontiguous(self):
        pairs = [
            ("mm", torch.arange(12.0).reshape(3, 4).t(), torch.arange(6.0).reshape(3, 2)),
            ("bmm", torch.arange(24.0).reshape(2, 3, 4), torch.arange(40.0).reshape(2, 4, 5)),
            ("matmul", torch.arange(24.0).reshape(2, 1, 3, 4), torch.arange(40.0).reshape(1, 5, 4, 2)),
            ("matmul", torch.arange(4.0), torch.arange(4.0)),
            ("matmul", torch.arange(4.0), torch.arange(24.0).reshape(2, 4, 3)),
            ("matmul", torch.arange(24.0).reshape(2, 3, 4), torch.arange(4.0)),
        ]
        for op, x, y in pairs:
            with self.subTest(op=op, shapes=(x.shape, y.shape)):
                fn = {"mm": torch.mm, "bmm": torch.bmm, "matmul": torch.matmul}[op]
                expected = fn(x, y)
                f = Fixture(x=x, y=y)
                f.node(op, [ref("x"), ref("y")], expected)
                self.assertExact(f.evaluate(), expected)

    def test_noncontiguous_coordinate_maps_and_structured_returns(self):
        x = torch.arange(60.0).reshape(3, 4, 5).permute(2, 0, 1)
        f = Fixture(x=x)
        sl = f.node("slice", [ref("x"), -1, -4, 999, 2], x[..., -4:999:2])
        se = f.node("select", [sl, 1, -1], x[..., -4:999:2].select(1, -1))
        pe = f.node("permute", [se, [1, 0]], x[..., -4:999:2].select(1, -1).permute(1, 0))
        tr = f.node("transpose", [ref("x"), 0, -1], x.transpose(0, -1))
        ex = f.node("expand", [se, [2, 5, 2]], x[..., -4:999:2].select(1, -1).expand(2, 5, 2))
        expected_flat = x[..., -4:999:2].select(1, -1).permute(1, 0).reshape(-1)
        fl = f.node("reshape", [pe, [-1]], expected_flat)
        f.graph["outputs"] = {"nested": [fl, {"old": ref("x"), "transpose": tr}], "again": fl, "expand": ex, "literal": 7}
        result = f.evaluate()
        self.assertExact(result["nested"][0], expected_flat)
        self.assertExact(result["nested"][1]["old"], x)
        self.assertExact(result["nested"][1]["transpose"], x.transpose(0, -1))
        self.assertIs(result["nested"][0], result["again"])
        self.assertEqual(result["literal"], 7)

    def test_functional_ssa_scatter_preserves_old_values_and_inputs(self):
        base = torch.arange(30.0).reshape(5, 6).t()
        # Scatter into a noncontiguous 6x5 destination, then into a selected row.
        src = torch.arange(15.0).reshape(3, 5) + 100
        original, source_original = base.clone(), src.clone()
        f = Fixture(base=base, src=src, row=torch.arange(5.0) - 50)
        first_expected = torch.slice_scatter(base, src, dim=0, start=1, end=6, step=2)
        first = f.node("slice_scatter", [ref("base"), ref("src"), 0, 1, 6, 2], first_expected)
        second_expected = torch.select_scatter(first_expected, f.inputs["row"], dim=0, index=-1)
        second = f.node("select_scatter", [first, ref("row"), 0, -1], second_expected)
        f.graph["outputs"] = [ref("base"), first, second]
        result = f.evaluate()
        self.assertExact(result[0], original)
        self.assertExact(result[1], first_expected)
        self.assertExact(result[2], second_expected)
        self.assertExact(base, original)
        self.assertExact(src, source_original)
        self.assertExact(result[2][::2], base[::2])

    def test_functional_copy_broadcast_cast_and_no_input_mutation(self):
        dst = torch.full((2, 3), -9, dtype=torch.float16)
        src = torch.tensor([0.12345, -2.71828, 3.14159])
        f = Fixture(dst=dst, src=src)
        expected = src.expand(2, 3).half()
        copy = f.node("copy", [ref("dst"), ref("src")], expected)
        f.graph["outputs"] = [ref("dst"), copy]
        old, new = f.evaluate()
        self.assertExact(old, torch.full_like(dst, -9))
        self.assertExact(new, expected)
        self.assertNotEqual(new.data_ptr(), src.data_ptr())
        self.assertNotEqual(new.data_ptr(), dst.data_ptr())

    def test_clone_contiguous_constants_scalar_specialization(self):
        x = torch.arange(12.0).reshape(3, 4).t()
        f = Fixture(x=x, alpha=0.5)
        cl = f.node("clone", [ref("x")], x.clone(), {"memory_format": {"memory_format": "preserve_format"}})
        co = f.node("contiguous", [cl], x.contiguous())
        f.graph["constants"]["c"] = {"shape": [], "dtype": "float32", "values": 2.0}
        m = f.node("mul", [co, ref("alpha")], x * 0.5)
        f.node("add", [m, ref("c")], x * 0.5 + 2)
        self.assertExact(f.evaluate(), x * 0.5 + 2)
        f.inputs["alpha"] = 0.25
        with self.assertRaisesRegex(EvaluationError, "specialized"):
            f.evaluate()

    def test_zeros_like_encoded_layout_and_meta_device(self):
        x = torch.ones(3, 2, dtype=torch.float16).t()
        f = Fixture(x=x)
        expected = torch.zeros_like(x, dtype=torch.float32)
        f.node("zeros_like", [ref("x")], expected,
               {"dtype": {"dtype": "float32"}, "device": {"device": "meta"},
                "layout": {"layout": "strided"}, "pin_memory": False})
        self.assertExact(f.evaluate(), expected)

    def test_unknown_ops_targets_and_callbacks_fail_closed(self):
        x = torch.ones(2)
        for op in ("aten.sin.default", "copy_", "__import__", "definitely_unknown"):
            f = Fixture(x=x)
            f.node(op, [ref("x")], x)
            f.graph["nodes"][0]["target"] = "aten.add.Tensor"
            with self.subTest(op=op), self.assertRaises(UnsupportedOperator):
                f.evaluate()
        f = Fixture(x=x)
        f.node("add", [ref("x"), 1], x + 1)
        # Raw target never selects code: normalized op remains the dispatcher.
        f.graph["nodes"][0]["target"] = "malicious.callback"
        with patch("torch.ops.aten.add.Tensor", side_effect=AssertionError("raw dispatch")):
            self.assertExact(f.evaluate(), x + 1)

    def test_invalid_graphs_rejected(self):
        x = torch.ones(2, 3)
        base = Fixture(x=x)
        base.node("add", [ref("x"), 1], x + 1)
        mutations = [
            ("arity", lambda g: g["nodes"][0].update(args=[ref("x")])),
            ("arity", lambda g: g["nodes"][0].update(args=[ref("x"), 1, 1, 1])),
            ("keyword", lambda g: g["nodes"][0].update(kwargs={"out": ref("x")})),
            ("shape", lambda g: g["nodes"][0]["outputs"][0].update(shape=[6])),
            ("dtype", lambda g: g["nodes"][0]["outputs"][0].update(dtype="float64")),
            ("arity", lambda g: g["nodes"][0]["outputs"].append(meta("dropped", x))),
            ("duplicate", lambda g: g["nodes"][0]["outputs"][0].update(id="x")),
            ("reference", lambda g: g.update(outputs=[ref("n0"), {"hidden": ref("absent")} ])),
            ("reference", lambda g: g["nodes"][0].update(args=[ref("future"), 1])),
        ]
        for expected, mutate in mutations:
            graph = deepcopy(base.graph)
            mutate(graph)
            with self.subTest(expected=expected), self.assertRaisesRegex(EvaluationError, expected):
                evaluate(graph, base.inputs)
        with self.assertRaisesRegex(EvaluationError, "shape"):
            evaluate(base.graph, {"x": torch.ones(3, 2)})
        with self.assertRaisesRegex(EvaluationError, "unexpected"):
            evaluate(base.graph, {"x": x, "unused": 1})

    def test_primitive_shape_and_axis_errors_rejected(self):
        x = torch.ones(2, 3)
        for op, args in (("permute", [ref("x"), [0, 0]]),
                         ("sum", [ref("x"), [0, 0]]),
                         ("select", [ref("x"), 1, 99]),
                         ("slice", [ref("x"), 0, None, None, -1]),
                         ("reshape", [ref("x"), [5]]),
                         ("view", [ref("x"), [5]])):
            f = Fixture(x=x)
            f.node(op, args, x)
            with self.subTest(op=op), self.assertRaises(EvaluationError):
                f.evaluate()

    def test_mutation_detector_on_adversarial_primitive(self):
        x = torch.ones(3)
        f = Fixture(x=x)
        f.node("tanh", [ref("x")], torch.tanh(x))
        def broken_primitive(tensor):
            tensor.add_(1)
            return tensor
        with patch("torch.tanh", broken_primitive), self.assertRaisesRegex(EvaluationError, "mutation"):
            f.evaluate()

    def test_copy_variant_has_independent_storage(self):
        x = torch.arange(6.0).reshape(2, 3)
        f = Fixture(x=x)
        f.node("permute", [ref("x"), [1, 0]], x.t(), attributes={"copy": True})
        result = f.evaluate()
        self.assertExact(result, x.t())
        self.assertNotEqual(result.untyped_storage().data_ptr(), x.untyped_storage().data_ptr())

    def test_view_refuses_noncontiguous_implicit_copy(self):
        x = torch.arange(12.0).reshape(3, 4).t()
        f = Fixture(x=x)
        f.node("view", [ref("x"), [-1]], x.reshape(-1))
        with self.assertRaises(EvaluationError):
            f.evaluate()

    def test_unannotated_attribute_metadata_conflict_rejected(self):
        x = torch.ones(2, 3)
        for attrs in ({"shape": [3, 2]}, {"dtype": "float64"}):
            f = Fixture(x=x)
            f.node("add", [ref("x"), 1], x + 1, attributes=attrs)
            with self.subTest(attrs=attrs), self.assertRaisesRegex(EvaluationError, "attribute"):
                f.evaluate()

    def test_nan_and_signed_zero_inputs_are_unchanged(self):
        x = torch.tensor([float("nan"), -0.0, 0.0, float("inf")])
        f = Fixture(x=x)
        f.node("clone", [ref("x")], x.clone())
        self.assertExact(f.evaluate(), x)


if __name__ == "__main__":
    unittest.main()
