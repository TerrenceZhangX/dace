"""Independent small convolution fixtures; no capture/source-task dependency.

Numerically recompute every output from the emitted guarded reads and scalar
reduction variables, rather than comparing only unordered dependency sets.
"""
from copy import deepcopy
import itertools
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as functional

from semantic_tool.core import UnsupportedOperator
from semantic_tool.ops_convolution import (
    execute_convolution_extension, lower_convolution_extension,
)
from semantic_tool.scalarize import index_eval, instantiate


def metadata(name, shape, dtype="float64"):
    shape = list(shape)
    return {"id": name, "name": name, "kind": "tensor", "shape": shape,
            "dtype": dtype, "strides": [math.prod(shape[j + 1:]) for j in range(len(shape))]}


def fixture(source=(1, 4, 7), weight=(6, 2, 3), *, groups=2, stride=None,
            padding=None, dilation=None, bias=True, dtype="float64"):
    dimensions = len(source) - 2
    stride = list(stride or [1] * dimensions)
    padding = list(padding or [0] * dimensions)
    dilation = list(dilation or [1] * dimensions)
    output = [source[0], weight[0]] + [
        (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
        for size, kernel, step, pad, dil in zip(source[2:], weight[2:], stride, padding, dilation)
    ]
    values = {"x": metadata("x", source, dtype), "w": metadata("w", weight, dtype)}
    if bias:
        values["b"] = metadata("b", [weight[0]], dtype)
    node = {"id": "conv", "target": "aten.convolution.default",
            "args": [{"ref": "x"}, {"ref": "w"}, {"ref": "b"} if bias else None,
                     stride, padding, dilation, False, [0] * dimensions, groups],
            "kwargs": {}, "outputs": [metadata("conv", output, dtype)]}
    return node, values


def tensors_for(values, dtype=None):
    """Binary fractions with distinct channel/spatial values expose swapped maps."""
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32, "float64": torch.float64}
    result = {}
    for offset, (name, meta) in enumerate(values.items()):
        count = math.prod(meta["shape"])
        data = ((torch.arange(count, dtype=torch.float64) * (offset + 3) + offset) % 23 - 11) / 16
        result[name] = data.reshape(meta["shape"]).to(dtype or dtypes[meta["dtype"]])
    return result


def resolve(value, tensors):
    if isinstance(value, dict) and set(value) == {"ref"}:
        return tensors[value["ref"]]
    if isinstance(value, list):
        return [resolve(item, tensors) for item in value]
    if isinstance(value, dict):
        return {key: resolve(item, tensors) for key, item in value.items()}
    return value


def cpu_reference(node, tensors):
    args = resolve(node["args"], tensors)
    primitive = {1: functional.conv1d, 2: functional.conv2d, 3: functional.conv3d}[len(args[0].shape) - 2]
    return primitive(args[0], args[1], args[2], args[3], args[4], args[5], args[8])


def recompute_from_reads(test, node, tensors):
    sem = node["semantics"]
    reduction = sem["reduction"]
    result = torch.zeros(node["outputs"][0]["shape"], dtype=torch.float64)
    guard_skips, total_bias_reads, visited = 0, 0, 0
    for output_index in itertools.product(*(range(d["extent"]) for d in sem["domain"])):
        env = {d["var"]: coordinate for d, coordinate in zip(sem["domain"], output_index)}
        total, bias_total, bias_reads = 0.0, 0.0, 0
        for coordinate in itertools.product(*(range(v["extent"]) for v in reduction["variables"])):
            local = {**env, **{v["var"]: c for v, c in zip(reduction["variables"], coordinate)}}
            product_operands = {}
            for read in sem["reads"]:
                if not index_eval(read.get("guard", True), local):
                    if read["operand"] == 0:
                        guard_skips += 1
                    continue
                index = tuple(index_eval(expr, local) for expr in read["indices"])
                source = tensors[read["value"]]
                test.assertEqual(len(index), source.ndim)
                test.assertTrue(all(type(i) is int and 0 <= i < extent for i, extent in zip(index, source.shape)),
                                (read, output_index, coordinate, index))
                if read["role"] == "bias":
                    test.assertEqual(read["scope"], "output")
                    test.assertEqual(read["frequency"], "once-per-output")
                    bias_total += float(source[index])
                    bias_reads += 1
                else:
                    test.assertNotIn(read["operand"], product_operands)
                    product_operands[read["operand"]] = float(source[index])
            if product_operands:
                test.assertEqual(set(product_operands), {0, 1})
                total += product_operands[0] * product_operands[1]
            else:
                test.assertFalse(index_eval(reduction["term_guard"], local))
                test.assertEqual(reduction["guard_false_value"], 0)
        test.assertEqual(bias_reads, int(node["attributes"]["has_bias"]))
        total_bias_reads += bias_reads
        visited += 1
        result[output_index] = total + bias_total
    return result, {"skips": guard_skips, "bias_reads": total_bias_reads, "outputs": visited}


class ConvolutionExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def check_case(self, source, weight, **options):
        capture_node, values = fixture(source, weight, **options)
        node_before, values_before = deepcopy(capture_node), deepcopy(values)
        node = lower_convolution_extension(capture_node, values)
        self.assertEqual(capture_node, node_before)
        self.assertEqual(values, values_before)
        self.assertEqual(node["op"], "convolution")
        self.assertEqual(node["semantics"]["reduction"]["tree"], "unspecified")
        self.assertEqual(node["semantics"]["order"], "data-only")
        self.assertTrue(node["semantics"]["arithmetic"]["finite_inputs_required"])
        self.assertTrue(node["attributes"]["finite_float_contract"])
        tensors = tensors_for(values)
        expected = cpu_reference(capture_node, tensors)
        recomputed, counts = recompute_from_reads(self, node, tensors)
        tolerance = {"float64": 1e-12, "float32": 1e-6, "float16": 2e-3, "bfloat16": 2e-2}[values["x"]["dtype"]]
        torch.testing.assert_close(recomputed, expected.double(), rtol=tolerance, atol=tolerance)
        actual = execute_convolution_extension(node, resolve(node["args"], tensors),
                                               resolve(node["kwargs"], tensors), node["attributes"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(counts["outputs"], expected.numel())
        self.assertEqual(counts["bias_reads"], expected.numel() if options.get("bias", True) else 0)
        return node, counts, tensors

    def test_1d_regular_grouped_depthwise_and_depthwise_multiplier(self):
        cases = [
            ((2, 3, 7), (5, 3, 3), dict(groups=1, padding=[1])),
            ((2, 4, 8), (6, 2, 3), dict(groups=2, stride=[2], padding=[1])),
            ((1, 3, 7), (3, 1, 3), dict(groups=3, padding=[2], bias=False)),
            ((2, 3, 7), (6, 1, 2), dict(groups=3, stride=[2], dilation=[2], padding=[2])),
            ((1, 2, 9), (3, 2, 1), dict(groups=1, bias=False)),
        ]
        for source, weight, options in cases:
            with self.subTest(source=source, weight=weight, **options):
                self.check_case(source, weight, **options)

    def test_2d_group_tail_nonsquare_dilated_per_axis_padding(self):
        cases = [
            ((1, 4, 7, 8), (6, 2, 2, 3), dict(groups=2, stride=[2, 3], padding=[1, 2], dilation=[2, 1])),
            ((2, 2, 5, 4), (4, 1, 3, 2), dict(groups=2, stride=[1, 2], padding=[2, 0], dilation=[1, 2], bias=False)),
            ((1, 3, 7, 7), (2, 3, 3, 3), dict(groups=1, stride=[2, 2], padding=[1, 1])),
        ]
        for source, weight, options in cases:
            with self.subTest(source=source, weight=weight):
                self.check_case(source, weight, **options)

    def test_3d_regular_grouped_depthwise(self):
        cases = [
            ((1, 2, 3, 4, 5), (3, 2, 2, 2, 2), dict(groups=1, stride=[1, 2, 2], padding=[0, 1, 0])),
            ((1, 4, 3, 4, 4), (6, 2, 2, 2, 2), dict(groups=2, stride=[2, 1, 2], dilation=[1, 2, 1], padding=[1, 1, 0], bias=False)),
            ((1, 2, 2, 3, 3), (4, 1, 2, 2, 2), dict(groups=2, padding=[1, 0, 1])),
        ]
        for source, weight, options in cases:
            with self.subTest(source=source, weight=weight):
                self.check_case(source, weight, **options)

    def test_bias_once_even_at_entirely_padded_output(self):
        node, counts, _ = self.check_case((1, 2, 1), (4, 1, 2), groups=2, padding=[4], dilation=[2])
        self.assertGreater(counts["skips"], 0)
        self.assertEqual(node["semantics"]["reduction"]["finalize"]["op"], "add-bias")
        self.assertEqual(node["semantics"]["reads"][0]["guard"], node["semantics"]["reads"][1]["guard"])

    def test_no_bias_and_zero_padding_no_fabricated_reads(self):
        node, counts, _ = self.check_case((1, 4, 7), (6, 2, 3), groups=2, bias=False)
        self.assertEqual(len(node["semantics"]["reads"]), 2)
        self.assertNotIn("finalize", node["semantics"]["reduction"])
        self.assertEqual(counts["skips"], 0)

    def test_empty_batch_supported_without_phantom_reads(self):
        for source, weight in [((0, 2, 5), (3, 2, 3)), ((0, 2, 4, 5), (3, 2, 2, 2)),
                               ((0, 2, 3, 4, 5), (3, 2, 2, 2, 2))]:
            with self.subTest(source=source):
                _, counts, _ = self.check_case(source, weight, groups=1)
                self.assertEqual(counts, {"skips": 0, "bias_reads": 0, "outputs": 0})

    def test_supported_floating_dtypes(self):
        for dtype in ("float16", "bfloat16", "float32", "float64"):
            with self.subTest(dtype=dtype):
                self.check_case((1, 4, 5), (6, 2, 3), groups=2, padding=[1], dtype=dtype)

    def test_noncontiguous_logical_coordinates(self):
        capture_node, values = fixture((1, 4, 5, 5), (6, 2, 2, 2), groups=2, padding=[1, 0])
        node = lower_convolution_extension(capture_node, values)
        tensors = tensors_for(values)
        tensors["x"] = tensors["x"].transpose(-1, -2)
        tensors["w"] = tensors["w"].transpose(-1, -2)
        self.assertFalse(tensors["x"].is_contiguous())
        expected = cpu_reference(capture_node, tensors)
        recomputed, _ = recompute_from_reads(self, node, tensors)
        torch.testing.assert_close(recomputed, expected, rtol=0, atol=0)
        actual = execute_convolution_extension(node, resolve(node["args"], tensors), {}, node["attributes"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_kwargs_and_specialized_structural_scalars(self):
        capture_node, values = fixture((1, 4, 5, 5), (6, 2, 3, 3), groups=2, padding=[1, 1])
        names = ("input", "weight", "bias", "stride", "padding", "dilation", "transposed", "output_padding", "groups")
        capture_node["kwargs"] = dict(zip(names, capture_node["args"]))
        capture_node["args"] = []
        capture_node["kwargs"]["groups"] = {"ref": "g"}
        capture_node["kwargs"]["stride"] = [{"ref": "s"}]
        values["g"] = {"id": "g", "kind": "scalar", "dtype": "int", "specialized_value": 2}
        values["s"] = {"id": "s", "kind": "scalar", "dtype": "int", "specialized_value": 1}
        node = lower_convolution_extension(capture_node, values)
        self.assertEqual(node["attributes"]["groups"], 2)
        self.assertEqual(node["attributes"]["stride"], [1, 1])
        tensors = tensors_for({key: values[key] for key in ("x", "w", "b")})
        tensors.update(g=2, s=1)
        actual = execute_convolution_extension(node, [], resolve(node["kwargs"], tensors), node["attributes"])
        expected = functional.conv2d(tensors["x"], tensors["w"], tensors["b"], padding=1, groups=2)
        torch.testing.assert_close(actual, expected)

    def test_helper_explicit_attributes_fallback(self):
        capture_node, values = fixture()
        node = lower_convolution_extension(capture_node, values)
        tensors = tensors_for(values)
        actual = execute_convolution_extension(node, [tensors["x"], tensors["w"], tensors["b"]], {}, node["attributes"])
        torch.testing.assert_close(actual, cpu_reference(capture_node, tensors))
        no_bias, no_bias_values = fixture(bias=False)
        no_bias_node = lower_convolution_extension(no_bias, no_bias_values)
        data = tensors_for(no_bias_values)
        actual = execute_convolution_extension(no_bias_node, [], {"input": data["x"], "weight": data["w"]}, no_bias_node["attributes"])
        torch.testing.assert_close(actual, cpu_reference(no_bias, data))

    def test_singleton_vector_expansion(self):
        capture_node, values = fixture((1, 2, 5, 5), (3, 2, 3, 3), groups=1, padding=[1, 1])
        for position in (3, 4, 5, 7):
            capture_node["args"][position] = capture_node["args"][position][:1]
        node = lower_convolution_extension(capture_node, values)
        self.assertEqual(node["attributes"]["padding"], [1, 1])
        tensors = tensors_for(values)
        actual = execute_convolution_extension(node, resolve(node["args"], tensors), {}, node["attributes"])
        torch.testing.assert_close(actual, functional.conv2d(tensors["x"], tensors["w"], tensors["b"], padding=1))

    def test_scalarization_bias_once_and_group_dependencies(self):
        capture_node, values = fixture((1, 4, 3), (6, 2, 2), groups=2, padding=[1])
        node = lower_convolution_extension(capture_node, values)
        # This standalone module intentionally does not require parent integration
        # into lower_capture/verify_semantics. The mapping itself is instantiated.
        scalar_node = deepcopy(node)
        scalar_node.pop("target")
        graph = {"format": "semantic-graph-v1", "inputs": list(values.values()), "nodes": [scalar_node],
                 "outputs": {"ref": "conv"}, "constants": {}}
        expanded = instantiate(graph)
        self.assertEqual(expanded["metadata"]["tree_choice"], "unspecified")
        output = "conv[0,4,0]"
        ancestors = {src for src, dst in expanded["edges"] if dst == output}
        self.assertEqual(ancestors, {"x[0,2,0]", "x[0,3,0]", "w[4,0,1]", "w[4,1,1]", "b[4]"})
        self.assertEqual(sum(src.startswith("b[") for src, _ in expanded["edges"]),
                         math.prod(node["outputs"][0]["shape"]))

    def test_unrelated_targets_return_none_and_never_dispatch(self):
        capture_node, values = fixture()
        for target in ("aten.conv1d.default", "aten.convolution.out", "aten._convolution.default",
                       "aten.convolution_backward.default", "aten.evil.default", "convolution"):
            with self.subTest(target=target):
                node = dict(capture_node, target=target)
                self.assertIsNone(lower_convolution_extension(node, values))
                node["op"] = "convolution"
                with patch.object(functional, "conv1d", side_effect=AssertionError("unexpected execution")):
                    with self.assertRaises(UnsupportedOperator):
                        execute_convolution_extension(node, [], {}, {})
        normal = lower_convolution_extension(capture_node, values)
        normal["op"] = "conv_transpose1d"
        with self.assertRaises(UnsupportedOperator):
            execute_convolution_extension(normal, [], {}, {})
        capture_node["target"] = "torch.ops.aten.convolution.default"
        self.assertEqual(lower_convolution_extension(capture_node, values)["op"], "convolution")

    def test_invalid_structural_parameters_fail_closed(self):
        cases = [
            (3, [0]), (3, [-1]), (3, [1.0]), (3, [True]), (3, []), (3, [1, 2]), (3, 1),
            (4, [-1]), (4, [1, 2]), (4, "same"), (4, [[1, 2]]),
            (5, [0]), (5, [-1]), (5, [2.0]), (6, True), (6, 0),
            (7, [1]), (7, [-1]), (8, 0), (8, -1), (8, 3), (8, 2.0), (8, True),
        ]
        for position, bad in cases:
            with self.subTest(position=position, bad=bad):
                capture_node, values = fixture()
                capture_node["args"][position] = bad
                with self.assertRaisesRegex(UnsupportedOperator, "conv: aten.convolution.default:"):
                    lower_convolution_extension(capture_node, values)
        capture_node, values = fixture((1, 2, 5, 5), (3, 2, 3, 3), groups=1)
        capture_node["args"][4] = [1, 2, 3, 4]  # not left/right pads; requires explicit pad op
        with self.assertRaisesRegex(UnsupportedOperator, "symmetric"):
            lower_convolution_extension(capture_node, values)

    def test_invalid_metadata_references_and_binding_fail_closed(self):
        def bad_shape(name, shape):
            return lambda node, values: values[name].update(shape=shape)
        edits = [
            bad_shape("x", [1, 0, 7]), bad_shape("x", [1, 4, 0]),
            bad_shape("w", [0, 2, 3]), bad_shape("w", [6, 0, 3]), bad_shape("w", [6, 2, 0]),
            bad_shape("w", [5, 2, 3]), bad_shape("w", [6, 3, 3]),
            bad_shape("x", [4, 7]), bad_shape("x", [1, 4, 2, 2, 2, 2]),
            bad_shape("w", [6, 2, 3, 3]), bad_shape("w", [6, 2, 99]),
            bad_shape("x", [1, -4, 7]), bad_shape("x", [True, 4, 7]),
            bad_shape("b", [1, 6]), bad_shape("b", [5]),
            lambda n, v: v["w"].update(dtype="float32"),
            lambda n, v: v["x"].update(dtype="complex64"),
            lambda n, v: v["x"].update(dtype="int64"),
            lambda n, v: v["b"].update(dtype="float32"),
            lambda n, v: v["x"].update(kind="scalar"),
            lambda n, v: n["outputs"][0].update(shape=[1, 6, 6]),
            lambda n, v: n["outputs"][0].update(dtype="float32"),
            lambda n, v: n.update(outputs=[]),
            lambda n, v: n["outputs"].append(deepcopy(n["outputs"][0])),
            lambda n, v: n["args"].__setitem__(0, {"ref": "unknown"}),
            lambda n, v: n["args"].__setitem__(1, 2.0),
            lambda n, v: n["args"].__setitem__(2, False),
            lambda n, v: n["args"].__setitem__(8, {"ref": "w"}),
            lambda n, v: n["args"].append(123),
            lambda n, v: n["args"].pop(),
            lambda n, v: n["kwargs"].update(groups=2),
            lambda n, v: n["kwargs"].update(cudnn_enabled=True),
        ]
        for number, edit in enumerate(edits):
            with self.subTest(edit=number):
                capture_node, values = fixture()
                edit(capture_node, values)
                with self.assertRaises(UnsupportedOperator):
                    lower_convolution_extension(capture_node, values)

    def test_nonfinite_constants_inputs_and_outputs_fail_closed(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                capture_node, values = fixture()
                bad_values = deepcopy(values)
                bad_values["w"]["values"] = [[[value]]]
                with self.assertRaisesRegex(UnsupportedOperator, "finite"):
                    lower_convolution_extension(capture_node, bad_values)
                node = lower_convolution_extension(capture_node, values)
                for name in ("x", "w", "b"):
                    tensors = tensors_for(values)
                    tensors[name].reshape(-1)[0] = value
                    with self.assertRaisesRegex(UnsupportedOperator, "finite"):
                        execute_convolution_extension(node, resolve(node["args"], tensors), {}, node["attributes"])
        capture_node, values = fixture((1, 1, 1), (1, 1, 1), groups=1, bias=False, dtype="float32")
        node = lower_convolution_extension(capture_node, values)
        with self.assertRaisesRegex(UnsupportedOperator, "result violates the finite"):
            execute_convolution_extension(node, [torch.full((1, 1, 1), 3e38), torch.full((1, 1, 1), 2.0), None], {}, node["attributes"])

    def _check_real_callable_pipeline(self, function, args, spatial_rank, groups):
        # These locally defined functions contain only a named F.conv primitive;
        # trusted=True acknowledges that inspected Python is executed by tracing.
        from semantic_tool.capture import capture
        from semantic_tool.evaluate import evaluate
        from semantic_tool.semantics import lower_capture
        from semantic_tool.verify import verify_semantics

        captured = capture(function, args, trusted=True)
        self.assertEqual([node["target"] for node in captured["nodes"]], ["aten.convolution.default"])
        graph = lower_capture(captured)
        self.assertTrue(verify_semantics(graph)["pass"])
        node = graph["nodes"][0]
        self.assertEqual(node["attributes"]["spatial_rank"], spatial_rank)
        self.assertEqual(node["attributes"]["groups"], groups)
        expanded = instantiate(graph)
        self.assertEqual(expanded["metadata"]["tree_choice"], "unspecified")
        self.assertEqual(expanded["metadata"]["reduction_groups"], math.prod(node["outputs"][0]["shape"]))
        bindings = {meta["id"]: value for meta, value in zip(captured["inputs"], args)}
        expected = function(*args)
        torch.testing.assert_close(evaluate(graph, bindings), expected, rtol=0, atol=0)
        # Captured SSA names flow through the same independent read-based oracle.
        recomputed, counts = recompute_from_reads(self, node, bindings)
        torch.testing.assert_close(recomputed, expected.double(), rtol=1e-12, atol=1e-12)
        self.assertEqual(counts["bias_reads"], expected.numel() if node["attributes"]["has_bias"] else 0)

    def test_real_function_capture_lower_scalarize_evaluate_grouped_1d(self):
        def function(x, weight, bias):
            return functional.conv1d(x, weight, bias, stride=2, padding=2, dilation=2, groups=2)
        _, values = fixture((2, 4, 7), (6, 2, 3), groups=2, stride=[2], padding=[2], dilation=[2])
        data = tensors_for(values)
        self._check_real_callable_pipeline(function, (data["x"], data["w"], data["b"]), 1, 2)

    def test_real_function_capture_lower_scalarize_evaluate_grouped_2d(self):
        def function(x, weight):
            return functional.conv2d(x, weight, None, stride=(2, 1), padding=(1, 2), dilation=(1, 2), groups=2)
        _, values = fixture((1, 4, 5, 6), (6, 2, 2, 3), groups=2,
                            stride=[2, 1], padding=[1, 2], dilation=[1, 2], bias=False)
        data = tensors_for(values)
        self._check_real_callable_pipeline(function, (data["x"], data["w"]), 2, 2)

    def test_real_function_capture_lower_scalarize_evaluate_grouped_3d(self):
        def function(x, weight, bias):
            return functional.conv3d(x, weight, bias, stride=(1, 2, 1), padding=(1, 0, 1), dilation=(2, 1, 1), groups=2)
        _, values = fixture((1, 4, 3, 4, 4), (6, 2, 2, 2, 2), groups=2,
                            stride=[1, 2, 1], padding=[1, 0, 1], dilation=[2, 1, 1])
        data = tensors_for(values)
        self._check_real_callable_pipeline(function, (data["x"], data["w"], data["b"]), 3, 2)

    def test_complex64_and_complex128_excluded_before_numeric_dispatch(self):
        for dtype in (torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype):
                capture_node, values = fixture()
                bad_values = deepcopy(values)
                for meta in bad_values.values():
                    meta["dtype"] = str(dtype).removeprefix("torch.")
                with self.assertRaisesRegex(UnsupportedOperator, "real floating dtype"):
                    lower_convolution_extension(capture_node, bad_values)
                node = lower_convolution_extension(capture_node, values)
                data = {name: tensor.to(dtype) for name, tensor in tensors_for(values).items()}
                with patch.object(functional, "conv1d", side_effect=AssertionError("complex dispatch forbidden")):
                    with self.assertRaisesRegex(UnsupportedOperator, "real floating dtype"):
                        execute_convolution_extension(node, resolve(node["args"], data), {}, node["attributes"])

    def test_helper_rejects_metadata_tampering_and_bad_runtime_args(self):
        capture_node, values = fixture()
        node = lower_convolution_extension(capture_node, values)
        tensors = tensors_for(values)
        args = resolve(node["args"], tensors)
        bad = deepcopy(node)
        bad["outputs"][0]["shape"][-1] += 1
        with self.assertRaisesRegex(UnsupportedOperator, "shape"):
            execute_convolution_extension(bad, args, {}, bad["attributes"])
        bad = deepcopy(node)
        bad["outputs"][0]["dtype"] = "float32"
        with self.assertRaisesRegex(UnsupportedOperator, "dtype"):
            execute_convolution_extension(bad, args, {}, bad["attributes"])
        bad_attrs = dict(node["attributes"], groups=1)
        with self.assertRaisesRegex(UnsupportedOperator, "attribute"):
            execute_convolution_extension(node, args, {}, bad_attrs)
        with self.assertRaisesRegex(UnsupportedOperator, "duplicate"):
            execute_convolution_extension(node, args, {"groups": 2}, node["attributes"])
        for bad_input in (torch.empty(values["x"]["shape"], device="meta"), 1.0, tensors["x"].float()):
            with self.subTest(bad_input=type(bad_input).__name__):
                with self.assertRaises(UnsupportedOperator):
                    execute_convolution_extension(node, [bad_input, *args[1:]], {}, node["attributes"])
        with patch.object(functional, "conv1d", side_effect=RuntimeError("CPU primitive rejected layout")):
            with self.assertRaisesRegex(UnsupportedOperator, "CPU primitive rejected"):
                execute_convolution_extension(node, args, {}, node["attributes"])


if __name__ == "__main__":
    unittest.main()
