"""Small CPU-only contract fixtures: no task imports, CUDA, or source replay."""
from copy import deepcopy
import itertools
import json
import math
import unittest
import warnings

import torch

from semantic_tool.ops_reduction import lower_reduction_extension, execute_reduction_extension
from semantic_tool.scalarize import instantiate, ancestors


def ref(name):
    return {"ref": name}


def meta(name, tensor):
    return {"id": name, "kind": "tensor", "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."), "strides": list(tensor.stride())}


def fixture(target, inputs, expected, tail=(), kwargs=None):
    values = {chr(97 + j): meta(chr(97 + j), value) for j, value in enumerate(inputs)}
    node = {"id": "z", "target": target, "args": [ref(name) for name in values] + list(tail),
            "kwargs": kwargs or {}, "outputs": [meta("z", expected)]}
    return node, values


def lower(target, inputs, expected, tail=(), kwargs=None):
    node, values = fixture(target, inputs, expected, tail, kwargs)
    result = lower_reduction_extension(node, values)
    assert result is not None
    return result


def execute(node, inputs, tail=(), kwargs=None):
    return execute_reduction_extension(node, list(inputs) + list(tail),
                                       node["kwargs"] if kwargs is None else kwargs, node["attributes"])


def points(shape):
    return itertools.product(*(range(extent) for extent in shape))


def read_at(read, coord, reductions=()):
    env = {f"i{j}": value for j, value in enumerate(coord)}
    env.update({f"r{j}": value for j, value in enumerate(reductions)})
    return tuple(eval(expr, {"__builtins__": {}}, env) for expr in read["indices"])


def semantic_graph(node, inputs):
    # These fixture assertions test graph expansion independent of main's
    # integration timing. Lowerer output is already directly verified below.
    node = deepcopy(node)
    node.pop("target")
    return {"format": "semantic-graph-v1", "inputs": [meta(chr(97+j), x) for j, x in enumerate(inputs)],
            "constants": {}, "nodes": [node], "outputs": ref("z")}


class ReductionExtensionTests(unittest.TestCase):
    def test_var_nonadjacent_axes_index_groups_match_torch(self):
        x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 7
        for keepdim in (False, True):
            for correction in (0, 1, 0.5, -1):
                with self.subTest(keepdim=keepdim, correction=correction):
                    expected = torch.var(x, dim=(2, 0), correction=correction, keepdim=keepdim)
                    node = lower("aten.var.correction", [x], expected, tail=[[-1, 0]],
                                 kwargs={"correction": correction, "keepdim": keepdim})
                    sem = node["semantics"]
                    self.assertEqual(sem["reduction"]["variables"], [{"var": "r0", "extent": 2}, {"var": "r1", "extent": 4}])
                    self.assertEqual(sem["reduction"]["tree"], "unspecified")
                    self.assertTrue(sem["reduction"]["functional"])
                    self.assertEqual(sem["reads"][0]["role"], "reduction")
                    self.assertEqual(sem["reads"][0]["indices"], ["r0", "i1" if keepdim else "i0", "r1"])
                    for coord in points(expected.shape):
                        group = torch.stack([x[read_at(sem["reads"][0], coord, rc)] for rc in points([2, 4])])
                        actual = ((group - group.mean()).abs().square().sum() / (group.numel() - correction))
                        torch.testing.assert_close(actual, expected[coord])
                    torch.testing.assert_close(execute(node, [x], tail=[[-1, 0]]), expected)

    def test_var_legacy_overloads_and_default_correction(self):
        x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
        cases = [
            ("aten.var.default", [], {}, torch.var(x)),
            ("aten.var.default", [False], {}, torch.var(x, unbiased=False)),
            ("aten.var.dim", [[-1], False, True], {}, torch.var(x, dim=-1, unbiased=False, keepdim=True)),
            ("aten.var.dim", [None], {}, torch.var(x)),
            ("aten.var.correction", [], {}, torch.var(x)),
            ("aten.var.correction", [[]], {"correction": None}, torch.var(x)),
        ]
        for target, tail, kwargs, expected in cases:
            with self.subTest(target=target, tail=tail):
                node = lower(target, [x], expected, tail, kwargs)
                torch.testing.assert_close(execute(node, [x], tail), expected)
                self.assertEqual(node["semantics"]["domain"], [{"var": f"i{j}", "extent": n} for j, n in enumerate(expected.shape)])

    def test_var_scalar_and_empty_statistics(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for x, dim, correction in [(torch.tensor(3.), [0], 0), (torch.tensor(3.), [-1], 1),
                                       (torch.empty(2, 0), [1], 0), (torch.empty(2, 0), [1], -1),
                                       (torch.empty(0, 3), [1], 1)]:
                with self.subTest(shape=x.shape, correction=correction):
                    expected = torch.var(x, dim=dim, correction=correction)
                    node = lower("aten.var.correction", [x], expected, [dim], {"correction": correction})
                    torch.testing.assert_close(execute(node, [x], [dim]), expected, equal_nan=True)
                    stats = node["semantics"]["reduction"]["statistics"]
                    self.assertFalse(stats["finite_output_guaranteed"])
                    self.assertEqual(stats["empty_group"], "nan")
                    if not x.ndim:
                        self.assertEqual(node["semantics"]["reduction"]["variables"], [])
                        self.assertEqual(node["semantics"]["reduction"]["group_size"], 1)

    def test_var_nonpositive_dof_nan_and_inf_not_all_finite(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for x in (torch.tensor([1.]), torch.tensor([1., 1.]), torch.tensor([1., 2.])):
                for correction in (x.numel(), x.numel() + 1):
                    expected = torch.var(x, correction=correction)
                    node = lower("aten.var.correction", [x], expected, kwargs={"correction": correction})
                    red = node["semantics"]["reduction"]
                    self.assertEqual(red["finalize"]["divisor"], 0)
                    self.assertEqual(red["statistics"]["nonpositive_degrees_of_freedom"]["zero_centered_sum"], "nan (0/0)")
                    self.assertFalse(red["statistics"]["finite_output_guaranteed"])
                    actual = execute(node, [x])
                    torch.testing.assert_close(actual, expected, equal_nan=True)
                    if x.numel() == 2 and x[0] != x[1]:
                        self.assertTrue(torch.isposinf(actual).item())
                    else:
                        self.assertTrue(torch.isnan(actual).item())

    def test_var_nonfinite_input_and_complex_real_result(self):
        for x in (torch.tensor([float("nan"), 2.]), torch.tensor([float("inf"), 2.]),
                  torch.tensor([[1+2j, 2-1j], [3+4j, 4+2j]], dtype=torch.complex128)):
            expected = torch.var(x, dim=-1, correction=0)
            node = lower("aten.var.correction", [x], expected, [[-1]], {"correction": 0})
            torch.testing.assert_close(execute(node, [x], [[-1]]), expected, equal_nan=True)
            arith = node["semantics"]["arithmetic"]
            self.assertEqual(arith["computation_dtype"], "ATen-opmath-of-promoted-inputs")
            self.assertEqual(arith["accumulation_dtype"], "ATen-opmath-of-promoted-inputs")
            if x.is_complex():
                self.assertEqual(arith["input_dtypes"], ["complex128"])
                self.assertEqual(arith["output_dtype"], "float64")

    def test_complex_var_nonpositive_dof_component_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for items in ([1+1j, 1+2j], [1+1j, 2+1j], [1+1j, 2+2j]):
                x = torch.tensor(items, dtype=torch.complex64)
                expected = torch.var(x, correction=2)
                node = lower("aten.var.correction", [x], expected, kwargs={"correction": 2})
                torch.testing.assert_close(execute(node, [x]), expected, equal_nan=True)
                stats = node["semantics"]["reduction"]["statistics"]
                self.assertIn("complex_component_rule", stats["nonpositive_degrees_of_freedom"])
                self.assertFalse(stats["finite_output_guaranteed"])

    def test_var_exact_output_cone(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        expected = x.var(dim=(0, 2), correction=0)
        node = lower("aten.var.correction", [x], expected, [[0, 2]], {"correction": 0})
        expanded = instantiate(semantic_graph(node, [x]))
        self.assertEqual(ancestors(expanded, "z[1]"), {f"a[{i},1,{k}]" for i in range(2) for k in range(4)})

    def test_gelu_modes_pointwise_indices_formula_and_cpu(self):
        x = torch.linspace(-3, 3, 12, dtype=torch.float64).reshape(3, 4).transpose(0, 1)
        for approximate in ("none", "tanh"):
            expected = torch.nn.functional.gelu(x, approximate=approximate)
            node = lower("aten.gelu.default", [x], expected, kwargs={"approximate": approximate})
            self.assertEqual(node["attributes"]["approximate"], approximate)
            self.assertIsNone(node["semantics"]["reduction"])
            read = node["semantics"]["reads"][0]
            self.assertEqual(read, {"value": "a", "indices": ["i0", "i1"], "role": "pointwise"})
            for coord in points(x.shape):
                value = x[read_at(read, coord)].item()
                gate = math.erf(value / math.sqrt(2)) if approximate == "none" else math.tanh(math.sqrt(2/math.pi) * (value + 0.044715*value**3))
                self.assertAlmostEqual(expected[coord].item(), 0.5 * value * (1 + gate), places=12)
            torch.testing.assert_close(execute(node, [x]), expected)
            expanded = instantiate(semantic_graph(node, [x]))
            self.assertEqual(ancestors(expanded, "z[1,2]"), {"a[1,2]"})

    def test_gelu_default_scalar_empty_and_low_precision_opmath(self):
        for x in (torch.tensor(-0.5), torch.empty(2, 0), torch.linspace(-2, 2, 9).half(),
                  torch.linspace(-2, 2, 9).bfloat16()):
            expected = torch.nn.functional.gelu(x)
            node = lower("torch.ops.aten.gelu.default", [x], expected)
            torch.testing.assert_close(execute(node, [x]), expected)
            self.assertEqual(node["attributes"]["approximate"], "none")
            self.assertEqual(node["semantics"]["arithmetic"]["computation_dtype"], "ATen-opmath-of-promoted-inputs")

    def test_addmm_broadcast_maps_alpha_beta_and_values(self):
        a = torch.arange(6, dtype=torch.float64).reshape(3, 2).t()
        b = torch.arange(12, dtype=torch.float64).reshape(4, 3).t()
        for bias_shape in ((), (1,), (4,), (1, 4), (2, 1), (2, 4)):
            for alpha, beta in ((1, 1), (0.25, -2), (0, 2)):
                with self.subTest(bias_shape=bias_shape, alpha=alpha, beta=beta):
                    bias = torch.arange(max(math.prod(bias_shape), 1), dtype=torch.float64).reshape(bias_shape) + 1
                    expected = torch.addmm(bias, a, b, alpha=alpha, beta=beta)
                    node = lower("aten.addmm.default", [bias, a, b], expected, kwargs={"alpha": alpha, "beta": beta})
                    reads = node["semantics"]["reads"]
                    self.assertEqual([r["role"] for r in reads], ["reduction", "reduction", "pointwise"])
                    self.assertEqual(node["semantics"]["reduction"]["tree"], "unspecified")
                    for coord in points(expected.shape):
                        contraction = sum(a[read_at(reads[0], coord, [k])] * b[read_at(reads[1], coord, [k])] for k in range(3))
                        value = alpha * contraction + beta * bias[read_at(reads[2], coord)]
                        torch.testing.assert_close(value, expected[coord])
                    torch.testing.assert_close(execute(node, [bias, a, b]), expected)

    def test_addmm_beta_zero_no_bias_value_read_or_cone(self):
        bias = torch.tensor([float("nan"), float("inf"), -float("inf"), 99.])
        a, b = torch.arange(6.).reshape(2, 3), torch.arange(12.).reshape(3, 4)
        expected = torch.addmm(bias, a, b, alpha=0.5, beta=0)
        node = lower("aten.addmm.default", [bias, a, b], expected, kwargs={"alpha": 0.5, "beta": 0})
        self.assertEqual({r["value"] for r in node["semantics"]["reads"]}, {"b", "c"})
        self.assertEqual(node["semantics"]["metadata_reads"], [{"value": "a", "fields": ["shape", "dtype"]}])
        actual = execute(node, [bias, a, b])
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())
        expanded = instantiate(semantic_graph(node, [bias, a, b]))
        self.assertEqual(ancestors(expanded, "z[1,2]"), {f"b[1,{k}]" for k in range(3)} | {f"c[{k},2]" for k in range(3)})

    def test_addmm_bias_cone_is_pointwise_not_whole_bias(self):
        bias, a, b = torch.ones(4), torch.ones(2, 3), torch.ones(3, 4)
        expected = torch.addmm(bias, a, b)
        node = lower("aten.addmm.default", [bias, a, b], expected)
        cone = ancestors(instantiate(semantic_graph(node, [bias, a, b])), "z[1,2]")
        self.assertEqual(cone, {"a[2]"} | {f"b[1,{k}]" for k in range(3)} | {f"c[{k},2]" for k in range(3)})

    def test_addmm_empty_contraction_keeps_bias_outside_zero_loop(self):
        bias, a, b = torch.arange(4.), torch.empty(2, 0), torch.empty(0, 4)
        for beta in (0, 2):
            expected = torch.addmm(bias, a, b, beta=beta)
            node = lower("aten.addmm.default", [bias, a, b], expected, kwargs={"beta": beta})
            self.assertIsNone(node["semantics"]["reduction"])
            self.assertEqual(node["attributes"]["contraction_extent"], 0)
            torch.testing.assert_close(execute(node, [bias, a, b]), expected)
            cone = ancestors(instantiate(semantic_graph(node, [bias, a, b])), "z[1,2]")
            self.assertEqual(cone, {"a[2]"} if beta else set())

    def test_addmm_low_precision_and_integer_fixture(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int64):
            bias, a, b = torch.ones(4, dtype=dtype), torch.ones(2, 3, dtype=dtype), torch.ones(3, 4, dtype=dtype)
            expected = torch.addmm(bias, a, b, alpha=2, beta=3)
            node = lower("aten.addmm.default", [bias, a, b], expected, kwargs={"alpha": 2, "beta": 3})
            torch.testing.assert_close(execute(node, [bias, a, b]), expected)
            self.assertEqual(node["semantics"]["arithmetic"]["accumulation_dtype"], "ATen-opmath-of-promoted-inputs")

    def test_specialized_scalar_is_preserved_and_unresolved_rejected(self):
        x = torch.arange(6.).reshape(2, 3)
        expected = x.var(dim=1, correction=0)
        node, values = fixture("aten.var.correction", [x], expected, [[1]], {"correction": ref("s")})
        values["s"] = {"id": "s", "kind": "scalar", "dtype": "int", "specialized_value": 0}
        lowered = lower_reduction_extension(node, values)
        self.assertEqual(lowered["attributes"]["correction"], 0)
        self.assertEqual(lowered["semantics"]["specializations"], [{"value": "s", "equals": 0}])
        del values["s"]["specialized_value"]
        with self.assertRaisesRegex(ValueError, "specialized"):
            lower_reduction_extension(node, values)

    def test_unknown_targets_are_not_silently_accepted(self):
        for target in ("aten.var.out", "aten.var.correction_out", "aten.gelu.out", "aten.addmm_.default", "aten.addmm.out", "aten.sin.default", "custom.addmm.default"):
            self.assertIsNone(lower_reduction_extension({"target": target}, {}))
            self.assertIsNone(execute_reduction_extension({"target": target, "op": "addmm"}, [], {}, {}))

    def test_signature_bad_shape_dtype_axes_and_options_fail_closed(self):
        x = torch.ones(2, 3)
        bad = [
            ("aten.var.correction", [x], torch.ones(2), [[1, -1]], {}),
            ("aten.var.correction", [x], torch.ones(2), [[3]], {}),
            ("aten.var.correction", [x], torch.ones(2), [[True]], {}),
            ("aten.var.correction", [x], torch.ones(2), [[1]], {"keepdim": 1}),
            ("aten.var.correction", [x], torch.ones(2), [[1]], {"correction": "0"}),
            ("aten.var.correction", [x], torch.ones(2), [[1]], {"correction": float("nan")}),
            ("aten.var.correction", [x], torch.ones(2), [[1], 0], {}),
            ("aten.var.dim", [x], torch.ones(2), [], {}),
            ("aten.var.dim", [x], torch.ones(2), [[1], 1], {}),
            ("aten.var.default", [x], torch.tensor(0.), [], {"dim": [1]}),
            ("aten.var.default", [x.to(torch.int64)], torch.tensor(0.), [], {}),
            ("aten.gelu.default", [x], x, [], {"approximate": "fast"}),
            ("aten.gelu.default", [x], x, ["tanh"], {}),
            ("aten.gelu.default", [x], x, [], {"out": None}),
            ("aten.gelu.default", [x], x.half(), [], {}),
            ("aten.gelu.default", [x], torch.ones(3, 2), [], {}),
            ("aten.addmm.default", [torch.ones(5), x, torch.ones(3, 4)], torch.ones(2, 4), [], {}),
            ("aten.addmm.default", [torch.ones(4), x, torch.ones(2, 4)], torch.ones(2, 4), [], {}),
            ("aten.addmm.default", [torch.ones(4), x, torch.ones(3, 4).double()], torch.ones(2, 4), [], {}),
            ("aten.addmm.default", [torch.ones(4), x, torch.ones(3, 4)], torch.ones(2, 4), [0], {}),
        ]
        for target, inputs, expected, tail, kwargs in bad:
            with self.subTest(target=target, tail=tail, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    lower(target, inputs, expected, tail, kwargs)

    def test_duplicate_unknown_ref_and_multioutput_fail_closed(self):
        x = torch.ones(2)
        node, values = fixture("aten.gelu.default", [x], x)
        node["kwargs"]["self"] = ref("a")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            lower_reduction_extension(node, values)
        node["kwargs"] = {}
        node["args"] = [ref("unknown")]
        with self.assertRaisesRegex(ValueError, "undefined"):
            lower_reduction_extension(node, values)
        node["args"] = [ref("a")]
        node["outputs"].append(meta("other", x))
        with self.assertRaisesRegex(ValueError, "one tensor output"):
            lower_reduction_extension(node, values)

    def test_executor_closed_signature_and_cpu_only(self):
        node = {"target": "aten.gelu.default"}
        with self.assertRaisesRegex(ValueError, "CPU"):
            execute_reduction_extension(node, [torch.empty(2, device="meta")], {}, {})
        with self.assertRaisesRegex(ValueError, "unknown keyword"):
            execute_reduction_extension(node, [torch.ones(2)], {"ignored": 1}, {})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            execute_reduction_extension(node, [torch.ones(2)], {"self": torch.ones(2)}, {})
        x = torch.tensor([-1., 1.])
        actual = execute_reduction_extension({"op": "gelu"}, [x], {}, {"approximate": "tanh"})
        torch.testing.assert_close(actual, torch.nn.functional.gelu(x, approximate="tanh"))
        with self.assertRaisesRegex(ValueError, "positional"):
            execute_reduction_extension(node, [x, "tanh"], {}, {})

    def test_public_capture_lower_scalarize_evaluate_five_functions(self):
        from semantic_tool.capture import capture
        from semantic_tool.semantics import lower_capture
        from semantic_tool.evaluate import evaluate
        from semantic_tool.scalarize import value_key

        x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 5
        bias = torch.tensor([float("nan"), float("inf"), 3., 4.], dtype=torch.float64)
        a, b = torch.arange(6., dtype=torch.float64).reshape(2, 3), torch.arange(12., dtype=torch.float64).reshape(3, 4)
        cases = [
            (lambda x: torch.var(x, dim=(0, -1), correction=0, keepdim=True), [x], "var", (0, 1, 0)),
            (lambda x: torch.nn.functional.gelu(x, approximate="none"), [x], "gelu", (1, 1, 2)),
            (lambda x: torch.nn.functional.gelu(x, approximate="tanh"), [x], "gelu", (1, 1, 2)),
            (lambda bias, a, b: torch.addmm(bias, a, b, alpha=0.25, beta=0), [bias, a, b], "addmm", (1, 2)),
            (lambda bias, a, b: torch.addmm(bias, a, b, alpha=2, beta=-1), [torch.arange(4., dtype=torch.float64), a, b], "addmm", (1, 2)),
        ]
        for function, inputs, op, coord in cases:
            with self.subTest(op=op, function=function):
                captured = capture(function, inputs, trusted=True)
                graph = lower_capture(captured)
                self.assertEqual([node["op"] for node in graph["nodes"]], [op])
                expanded = instantiate(graph)
                actual = evaluate(graph, {meta["id"]: value for meta, value in zip(graph["inputs"], inputs)})
                torch.testing.assert_close(actual, function(*inputs), equal_nan=True)
                node = graph["nodes"][0]
                cone = ancestors(expanded, value_key(node["outputs"][0]["id"], coord))
                ids = [meta["id"] for meta in graph["inputs"]]
                if op == "var":
                    expected_cone = {value_key(ids[0], (i, 1, k)) for i in range(2) for k in range(4)}
                elif op == "gelu":
                    expected_cone = {value_key(ids[0], coord)}
                else:
                    expected_cone = {value_key(ids[1], (1, k)) for k in range(3)} | {value_key(ids[2], (k, 2)) for k in range(3)}
                    if node["attributes"]["beta"] != 0:
                        expected_cone.add(value_key(ids[0], (2,)))
                self.assertEqual(cone, expected_cone)

    def test_lowerer_is_pure_and_json_roundtrips(self):
        x = torch.arange(6.).reshape(2, 3)
        node, values = fixture("aten.var.correction", [x], x.var(dim=1), [[1]])
        before = deepcopy((node, values))
        result = lower_reduction_extension(node, values)
        self.assertEqual((node, values), before)
        self.assertEqual(json.loads(json.dumps(result)), result)


if __name__ == "__main__":
    unittest.main()
