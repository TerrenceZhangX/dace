"""Contract-fixture tests of ATen lowering and exact CPU coordinate relations.

These tests need neither task IDs nor the capture module.  CPU torch verifies
logical maps independently; reduction trees are deliberately not instantiated.
"""
import itertools
import json
import math
import unittest

import torch

from semantic_tool.semantics import UnsupportedOperator, lower_capture


def ref(name):
    return {"ref": name}


def meta(name, shape, dtype="float32", strides=None, **extra):
    shape = list(shape)
    if strides is None:
        strides = [math.prod(shape[j+1:]) for j in range(len(shape))]
    return {"id": name, "name": name, "shape": shape, "dtype": dtype,
            "strides": list(strides), "kind": "tensor", **extra}


def fixture(target, shapes, out_shape, tail=(), kwargs=None, dtype="float32", out_dtype=None):
    inputs = [meta(chr(97+j), shape, dtype) for j, shape in enumerate(shapes)]
    node = {"id": "z", "target": target, "args": [ref(v["id"]) for v in inputs] + list(tail),
            "kwargs": kwargs or {}, "outputs": [meta("z", out_shape, out_dtype or dtype)]}
    return {"format": "aten-capture-v1", "inputs": inputs, "nodes": [node],
            "outputs": ref("z"), "constants": {}, "metadata": {}}


def lower(*args, **kwargs):
    return lower_capture(fixture(*args, **kwargs))["nodes"][0]


def points(shape):
    return itertools.product(*(range(n) for n in shape))


def index(expression, env):
    return eval(expression, {"__builtins__": {}, "min": min, "max": max}, env)


def read_at(read, coords, **reduction):
    env = {f"i{j}": i for j, i in enumerate(coords)}
    env.update(reduction)
    if not index(read.get("guard", "True"), env):
        return None
    return tuple(index(expr, env) for expr in read["indices"])


class SemanticContractTests(unittest.TestCase):
    def test_basic_unary_targets(self):
        for op in ("tanh", "exp", "sigmoid", "sqrt", "rsqrt", "neg", "abs", "relu", "log", "sin", "cos", "erf"):
            with self.subTest(op=op):
                node = lower(f"aten.{op}.default", [[2, 3]], [2, 3])
                self.assertEqual(node["op"], op)
                self.assertEqual(node["semantics"]["reads"], [{"value": "a", "indices": ["i0", "i1"], "role": "pointwise"}])
                self.assertIsNone(node["semantics"]["reduction"])
                self.assertEqual(node["semantics"]["order"], "data-only")

    def test_binary_broadcast_scalar0d(self):
        for op in ("add", "sub", "mul", "div", "pow", "maximum"):
            overload = "Tensor_Tensor" if op == "pow" else "default" if op == "maximum" else "Tensor"
            with self.subTest(op=op):
                node = lower(f"aten.{op}.{overload}", [[2, 1, 3], [1, 4, 1]], [2, 4, 3])
                self.assertEqual([r["indices"] for r in node["semantics"]["reads"]], [["i0", "0", "i2"], ["0", "i1", "0"]])
        node = lower("aten.add.Tensor", [[2, 3], []], [2, 3])
        self.assertEqual(node["semantics"]["reads"][1]["indices"], [])

    def test_scalar_placeholder_is_real_value_read(self):
        graph = fixture("aten.mul.Scalar", [[2]], [2], tail=[ref("s")])
        graph["inputs"].append({"id": "s", "kind": "scalar", "dtype": "float", "specialized_value": 3.5})
        node = lower_capture(graph)["nodes"][0]
        self.assertEqual(node["semantics"]["reads"][1], {"value": "s", "indices": [], "role": "pointwise"})

    def test_literal_alpha_and_rounding(self):
        node = lower("aten.add.Tensor", [[2], [2]], [2], kwargs={"alpha": 0.25})
        self.assertEqual(node["attributes"]["alpha"], 0.25)
        self.assertEqual(len(node["semantics"]["reads"]), 2)
        node = lower("aten.div.Tensor_mode", [[2], [2]], [2], kwargs={"rounding_mode": "floor"})
        self.assertEqual(node["attributes"]["rounding_mode"], "floor")
        self.assertEqual(node["semantics"]["arithmetic"]["rounding"], "at-output")

    def test_rsub_keeps_operand_order_and_alpha(self):
        node = lower("aten.rsub.Scalar", [[2, 3]], [2, 3], tail=[1], kwargs={"alpha": 2})
        self.assertEqual(node["op"], "rsub")
        self.assertEqual(node["args"], [ref("a"), 1])
        self.assertEqual(node["attributes"]["alpha"], 2)
        self.assertEqual(node["semantics"]["reads"][0]["indices"], ["i0", "i1"])

    def test_broadcast_indices_match_cpu(self):
        a = torch.arange(6).reshape(2, 1, 3)
        b = torch.arange(4).reshape(1, 4, 1)
        expected = a + b
        node = lower("aten.add.Tensor", [list(a.shape), list(b.shape)], list(expected.shape), dtype="int64")
        for coords in points(expected.shape):
            ia = read_at(node["semantics"]["reads"][0], coords)
            ib = read_at(node["semantics"]["reads"][1], coords)
            self.assertEqual((a[ia] + b[ib]).item(), expected[coords].item())

    def test_invalid_output_shape_is_rejected(self):
        with self.assertRaisesRegex(UnsupportedOperator, "z: aten.add.Tensor"):
            lower("aten.add.Tensor", [[2, 3], [2, 3]], [3, 2])
        with self.assertRaises(UnsupportedOperator):
            lower("aten.add.Tensor", [[2, 3], [4]], [2, 3])

    def test_empty_domain_broadcast(self):
        node = lower("aten.add.Tensor", [[0, 3], [1, 3]], [0, 3])
        self.assertEqual(node["semantics"]["domain"][0]["extent"], 0)

    def test_scalar_output_domain(self):
        node = lower("aten.mul.Tensor", [[], []], [])
        self.assertEqual(node["semantics"]["domain"], [])
        self.assertTrue(all(r["indices"] == [] for r in node["semantics"]["reads"]))

    def test_row_reduction(self):
        for op in ("sum.dim_IntList", "mean.dim", "amax.default"):
            node = lower("aten." + op, [[2, 3]], [2, 1], tail=[[-1], True])
            sem = node["semantics"]
            self.assertEqual(node["attributes"]["axes"], [1])
            self.assertEqual(sem["reads"][0]["indices"], ["i0", "r0"])
            self.assertEqual(sem["reduction"]["variables"], [{"var": "r0", "extent": 3}])
            self.assertEqual(sem["reduction"]["tree"], "unspecified")
            self.assertEqual(sem["domain"], [{"var": "i0", "extent": 2}, {"var": "i1", "extent": 1}])

    def test_global_reduction(self):
        node = lower("aten.sum.default", [[2, 3, 4]], [])
        self.assertEqual(node["semantics"]["reads"][0]["indices"], ["r0", "r1", "r2"])
        self.assertEqual(node["semantics"]["reduction"]["group_size"], 24)
        self.assertEqual(node["semantics"]["domain"], [])

    def test_nonadjacent_reduction_axes(self):
        node = lower("aten.mean.dim", [[2, 3, 4]], [3], tail=[[2, 0], False])
        self.assertEqual(node["semantics"]["reads"][0]["indices"], ["r0", "i0", "r1"])
        self.assertEqual(node["semantics"]["reduction"]["finalize"]["divisor"], 8)

    def test_empty_axes_mean_is_global(self):
        node = lower("aten.mean.dim", [[2, 3]], [], tail=[[], False])
        self.assertEqual(node["attributes"]["axes"], [0, 1])
        self.assertEqual(node["semantics"]["reduction"]["finalize"]["divisor"], 6)

    def test_scalar_reduction_has_one_member(self):
        node = lower("aten.sum.dim_IntList", [[]], [], tail=[[0], True])
        self.assertEqual(node["semantics"]["reduction"]["variables"], [])
        self.assertEqual(node["semantics"]["reduction"]["group_size"], 1)
        self.assertEqual(node["semantics"]["reads"][0]["indices"], [])

    def test_reduction_indices_match_cpu(self):
        source = torch.arange(24).reshape(2, 3, 4)
        node = lower("aten.sum.dim_IntList", [[2, 3, 4]], [3], tail=[[0, 2], False], dtype="int64")
        expected = source.sum((0, 2))
        for coords in points(expected.shape):
            total = sum(source[read_at(node["semantics"]["reads"][0], coords, r0=r0, r1=r1)].item()
                        for r0, r1 in points([2, 4]))
            self.assertEqual(total, expected[coords].item())

    def test_reduction_dtype_effect_is_explicit(self):
        node = lower("aten.sum.default", [[3]], [], kwargs={"dtype": {"dtype": "float64"}}, out_dtype="float64")
        self.assertEqual(node["attributes"]["dtype"], "float64")
        self.assertEqual(node["semantics"]["arithmetic"]["accumulation_dtype"], "float64")
        self.assertEqual(node["semantics"]["arithmetic"]["input_dtypes"], ["float32"])
        node = lower("aten.mean.default", [[3]], [], dtype="float16")
        self.assertEqual(node["semantics"]["arithmetic"]["accumulation_dtype"], "float32")

    def test_softmax_has_whole_row_dependency(self):
        node = lower("aten._softmax.default", [[2, 3, 4]], [2, 3, 4], tail=[1, False])
        self.assertEqual(node["op"], "softmax")
        self.assertEqual(node["attributes"]["dim"], 1)
        self.assertFalse(node["attributes"]["half_to_float"])
        self.assertEqual(node["semantics"]["reads"][0], {"value": "a", "indices": ["i0", "r0", "i2"], "role": "row"})
        self.assertEqual(node["semantics"]["reduction"]["variables"], [{"var": "r0", "extent": 3}])
        self.assertTrue(node["semantics"]["reduction"]["functional"])

    def test_softmax_cast_signature_not_confused_with_half_flag(self):
        node = lower("aten.softmax.int", [[2, 3]], [2, 3], tail=[-1, {"dtype": "float64"}], out_dtype="float64")
        self.assertEqual(node["attributes"]["dtype"], "float64")
        self.assertFalse(node["attributes"]["half_to_float"])
        node = lower("aten._softmax.default", [[2, 3]], [2, 3], tail=[-1, True], dtype="float16", out_dtype="float32")
        self.assertTrue(node["attributes"]["half_to_float"])

    def test_matrix_contraction(self):
        node = lower("aten.mm.default", [[2, 3], [3, 4]], [2, 4])
        self.assertEqual([r["indices"] for r in node["semantics"]["reads"]], [["i0", "r0"], ["r0", "i1"]])
        self.assertEqual(node["semantics"]["reduction"]["operator"], "sum-of-products")
        self.assertEqual(node["semantics"]["reduction"]["tree"], "unspecified")

    def test_batched_matrix_contraction(self):
        node = lower("aten.bmm.default", [[5, 2, 3], [5, 3, 4]], [5, 2, 4])
        self.assertEqual([r["indices"] for r in node["semantics"]["reads"]], [["i0", "i1", "r0"], ["i0", "r0", "i2"]])
        with self.assertRaises(UnsupportedOperator):
            lower("aten.bmm.default", [[1, 2, 3], [5, 3, 4]], [5, 2, 4])

    def test_broadcast_batched_contraction(self):
        node = lower("aten.matmul.default", [[2, 1, 3, 4], [5, 4, 6]], [2, 5, 3, 6])
        self.assertEqual([r["indices"] for r in node["semantics"]["reads"]], [["i0", "0", "i2", "r0"], ["i1", "r0", "i3"]])

    def test_vector_matmul_cases(self):
        cases = [([[3], [3]], [], [["r0"], ["r0"]]),
                 ([[3], [2, 3, 4]], [2, 4], [["r0"], ["i0", "r0", "i1"]]),
                 ([[2, 4, 3], [3]], [2, 4], [["i0", "i1", "r0"], ["r0"]])]
        for shapes, out_shape, maps in cases:
            with self.subTest(shapes=shapes):
                node = lower("aten.matmul.default", shapes, out_shape)
                self.assertEqual([r["indices"] for r in node["semantics"]["reads"]], maps)

    def test_matmul_indices_match_cpu(self):
        a = torch.arange(2*1*3*4).reshape(2, 1, 3, 4)
        b = torch.arange(5*4*6).reshape(5, 4, 6)
        expected = a @ b
        node = lower("aten.matmul.default", [list(a.shape), list(b.shape)], list(expected.shape), dtype="int64")
        ra, rb = node["semantics"]["reads"]
        for coords in points(expected.shape):
            total = sum(a[read_at(ra, coords, r0=r)].item() * b[read_at(rb, coords, r0=r)].item() for r in range(4))
            self.assertEqual(total, expected[coords].item())

    def test_transpose_and_permute(self):
        node = lower("aten.transpose.int", [[2, 3, 4]], [4, 3, 2], tail=[0, -1])
        self.assertEqual(node["attributes"]["dims"], [0, 2])
        self.assertEqual(node["semantics"]["reads"][0]["indices"], ["i2", "i1", "i0"])
        node = lower("aten.permute.default", [[2, 3, 4]], [4, 2, 3], tail=[[2, 0, 1]])
        self.assertEqual(node["semantics"]["reads"][0]["indices"], ["i1", "i2", "i0"])
        self.assertEqual(node["semantics"]["layout"]["alias"], "view")

    def test_transpose_copy_does_not_demand_materialization(self):
        node = lower("aten.transpose_copy.int", [[2, 3]], [3, 2], tail=[0, 1])
        self.assertEqual(node["op"], "transpose")
        self.assertEqual(node["semantics"]["layout"]["alias"], "none")
        self.assertFalse(node["semantics"]["layout"]["requires_materialization"])

    def test_reshape_linearization_matches_cpu(self):
        a = torch.arange(24).reshape(2, 3, 4)
        for op in ("view", "reshape", "_unsafe_view", "view_copy", "reshape_copy"):
            with self.subTest(op=op):
                node = lower(f"aten.{op}.default", [[2, 3, 4]], [4, 6], tail=[[4, -1]], dtype="int64")
                expected = a.reshape(4, 6)
                for coords in points(expected.shape):
                    self.assertEqual(a[read_at(node["semantics"]["reads"][0], coords)].item(), expected[coords].item())

    def test_noncontiguous_reshape_has_logical_not_storage_map(self):
        a = torch.arange(12).reshape(3, 4).t()
        graph = fixture("aten.reshape.default", [[4, 3]], [2, 6], tail=[[2, 6]], dtype="int64")
        graph["inputs"][0]["strides"] = [1, 4]
        node = lower_capture(graph)["nodes"][0]
        self.assertEqual(node["semantics"]["layout"]["input_strides"], [1, 4])
        self.assertEqual(node["semantics"]["layout"]["alias"], "may-alias")
        expected = a.reshape(2, 6)
        for coords in points(expected.shape):
            self.assertEqual(a[read_at(node["semantics"]["reads"][0], coords)].item(), expected[coords].item())

    def test_storage_offsets_and_strides_are_preserved(self):
        graph = fixture("aten.select.int", [[2, 3]], [3], tail=[0, 1])
        graph["inputs"][0].update(strides=[5, 1], storage_offset=4)
        graph["nodes"][0]["outputs"][0].update(strides=[1], storage_offset=9)
        node = lower_capture(graph)["nodes"][0]
        layout = node["semantics"]["layout"]
        self.assertEqual(layout["input_storage_offset"], 4)
        self.assertEqual(layout["output_storage_offset"], 9)
        for i in range(3):
            self.assertEqual(index(layout["input_storage_index"], {"i0": i}), 9+i)
            self.assertEqual(index(layout["output_storage_index"], {"i0": i}), 9+i)

    def test_layout_operation_cannot_silently_cast(self):
        with self.assertRaisesRegex(UnsupportedOperator, "cannot change element dtype"):
            lower("aten.clone.default", [[2]], [2], out_dtype="float16")

    def test_empty_reshape_has_no_invalid_division(self):
        node = lower("aten.reshape.default", [[0, 3]], [3, 0], tail=[[3, 0]])
        self.assertNotIn("// 0", str(node["semantics"]["reads"]))

    def test_scalar_reshape(self):
        node = lower("aten.view.default", [[]], [1, 1], tail=[[1, 1]])
        self.assertEqual(node["semantics"]["reads"][0]["indices"], [])
        node = lower("aten.reshape.default", [[1, 1]], [], tail=[[]])
        self.assertEqual(node["semantics"]["reads"][0]["indices"], ["0", "0"])

    def test_expand_scalar_and_singleton(self):
        node = lower("aten.expand.default", [[]], [2, 3], tail=[[2, 3]])
        self.assertEqual(node["semantics"]["reads"][0]["indices"], [])
        node = lower("aten.expand.default", [[1, 3]], [2, 4, 3], tail=[[2, 4, -1]])
        self.assertEqual(node["semantics"]["reads"][0]["indices"], ["0", "i2"])
        with self.assertRaises(UnsupportedOperator):
            lower("aten.expand.default", [[3]], [2, 3], tail=[[-1, 3]])

    def test_squeeze_unsqueeze(self):
        node = lower("aten.unsqueeze.default", [[2, 3]], [2, 1, 3], tail=[1])
        self.assertEqual(node["semantics"]["reads"][0]["indices"], ["i0", "i2"])
        node = lower("aten.squeeze.default", [[1, 2, 1, 3]], [2, 3])
        self.assertEqual(node["semantics"]["reads"][0]["indices"], ["0", "i0", "0", "i1"])

    def test_clone_retains_layout_and_no_rounding(self):
        node = lower("aten.clone.default", [[2, 3]], [2, 3], kwargs={"memory_format": {"memory_format": "contiguous_format"}})
        self.assertEqual(node["op"], "clone")
        self.assertEqual(node["attributes"]["memory_format"], {"memory_format": "contiguous_format"})
        self.assertEqual(node["semantics"]["arithmetic"]["rounding"], "none")
        self.assertEqual(node["semantics"]["layout"]["alias"], "none")

    def test_cast_is_explicit_typed_boundary(self):
        node = lower("aten._to_copy.default", [[2, 3]], [2, 3], kwargs={"dtype": {"dtype": "float16"}}, out_dtype="float16")
        self.assertEqual(node["op"], "cast")
        self.assertEqual(node["attributes"]["source_dtype"], "float32")
        self.assertEqual(node["attributes"]["dtype"], "float16")
        self.assertEqual(node["semantics"]["arithmetic"]["rounding"], "dtype-conversion")
        self.assertEqual(node["semantics"]["arithmetic"]["conversion"]["float_narrowing"], "round-to-nearest-ties-to-even")

    def test_select_maps_and_layout(self):
        node = lower("aten.select.int", [[2, 3, 4]], [2, 4], tail=[1, -1])
        self.assertEqual(node["attributes"], {"dim": 1, "index": 2, "copy": False})
        self.assertEqual(node["semantics"]["reads"][0]["indices"], ["i0", "2", "i1"])

    def test_slice_bounds_and_step_match_cpu(self):
        a = torch.arange(14).reshape(2, 7)
        node = lower("aten.slice.Tensor", [[2, 7]], [2, 3], tail=[1, -6, 99, 2], dtype="int64")
        expected = a[:, -6:99:2]
        for coords in points(expected.shape):
            self.assertEqual(a[read_at(node["semantics"]["reads"][0], coords)].item(), expected[coords].item())
        self.assertEqual(node["attributes"], {"dim": 1, "start": 1, "end": 7, "step": 2, "copy": False})

    def test_functional_copy_reads_only_source_values(self):
        node = lower("aten.copy.default", [[2, 3], [3]], [2, 3])
        self.assertEqual(node["semantics"]["reads"], [{"value": "b", "indices": ["i1"], "role": "pointwise"}])
        self.assertEqual(node["semantics"]["metadata_reads"][0]["value"], "a")

    def test_slice_scatter_guards_match_cpu_and_preserve_old_values(self):
        a = torch.arange(14).reshape(2, 7)
        b = torch.arange(6).reshape(2, 3) + 100
        node = lower("aten.slice_scatter.default", [[2, 7], [2, 3]], [2, 7], tail=[1, 1, 7, 2], dtype="int64")
        expected = torch.slice_scatter(a, b, dim=1, start=1, end=7, step=2)
        for coords in points(expected.shape):
            active = [(tensor, read_at(read, coords)) for tensor, read in zip((a, b), node["semantics"]["reads"])]
            active = [(tensor, idx) for tensor, idx in active if idx is not None]
            self.assertEqual(len(active), 1, "must not create both old/new dependency at overwritten coordinates")
            tensor, idx = active[0]
            self.assertEqual(tensor[idx].item(), expected[coords].item())
        self.assertEqual(node["semantics"]["update"]["kind"], "functional-SSA-scatter")

    def test_select_scatter_guards_match_cpu(self):
        a = torch.arange(24).reshape(2, 3, 4)
        b = torch.arange(8).reshape(2, 4) + 100
        node = lower("aten.select_scatter.default", [[2, 3, 4], [2, 4]], [2, 3, 4], tail=[1, -2], dtype="int64")
        expected = torch.select_scatter(a, b, dim=1, index=-2)
        for coords in points(expected.shape):
            active = [(tensor, read_at(read, coords)) for tensor, read in zip((a, b), node["semantics"]["reads"])]
            active = [(tensor, idx) for tensor, idx in active if idx is not None]
            self.assertEqual(len(active), 1)
            tensor, idx = active[0]
            self.assertEqual(tensor[idx].item(), expected[coords].item())

    def test_zero_like_has_metadata_not_false_value_dependency(self):
        node = lower("aten.zeros_like.default", [[2, 3]], [2, 3])
        self.assertEqual(node["op"], "zeros_like")
        self.assertEqual(node["attributes"]["fill_value"], 0)
        self.assertEqual(node["semantics"]["reads"], [])
        self.assertEqual(node["semantics"]["metadata_reads"][0]["value"], "a")

    def test_factories(self):
        node = lower("aten.full.default", [], [2, 3], tail=[[2, 3], 7])
        self.assertEqual(node["attributes"]["fill_value"], 7)
        node = lower("aten.ones.default", [], [2], tail=[[2]])
        self.assertEqual(node["op"], "full")
        self.assertEqual(node["attributes"]["fill_value"], 1)
        node = lower("aten.new_zeros.default", [[5]], [2, 3], tail=[[2, 3]])
        self.assertEqual(node["attributes"]["shape"], [2, 3])

    def test_arange(self):
        node = lower("aten.arange.start_step", [], [4], tail=[1, 9, 2], dtype="int64")
        self.assertEqual(node["attributes"], {"start": 1, "end": 9, "step": 2, "dtype": "int64"})
        self.assertEqual([index(node["semantics"]["expression"], {"i0": i}) for i in range(4)], [1, 3, 5, 7])

    def test_unknown_and_mutation_targets_fail_closed(self):
        for target in ("aten.unknown.default", "aten.copy_.default", "aten.add_.Tensor", "aten.add.out", "custom.magic.default", "aten.empty.memory_format"):
            with self.subTest(target=target), self.assertRaises(UnsupportedOperator) as cm:
                lower(target, [[2]], [2])
            self.assertIn("z", str(cm.exception))
            self.assertIn(target, str(cm.exception))

    def test_computation_type_is_not_inferred_from_result_dtype(self):
        node = lower("aten.eq.Tensor", [[2], [2]], [2], out_dtype="bool")
        self.assertEqual(node["semantics"]["arithmetic"]["computation_dtype"], "ATen-opmath-of-promoted-inputs")
        node = lower("aten.abs.default", [[2]], [2], dtype="complex64", out_dtype="float32")
        self.assertEqual(node["semantics"]["arithmetic"]["computation_dtype"], "ATen-opmath-of-promoted-inputs")
        node = lower("aten.mul.Tensor", [[2]], [2], tail=[0.1234567], dtype="bfloat16")
        self.assertIn("not pre-rounded", node["semantics"]["arithmetic"]["literal_conversion"])

    def test_uninitialized_values_are_not_fabricated(self):
        for op in ("empty_like", "new_empty", "empty_strided"):
            with self.subTest(op=op), self.assertRaisesRegex(UnsupportedOperator, "uninitialized"):
                lower("aten." + op + ".default", [[2]], [2])

    def test_undefined_dependency_and_multioutput_fail_closed(self):
        graph = fixture("aten.neg.default", [[2]], [2])
        graph["nodes"][0]["args"] = [ref("missing")]
        with self.assertRaises(UnsupportedOperator):
            lower_capture(graph)
        graph = fixture("aten.neg.default", [[2]], [2])
        graph["nodes"][0]["outputs"].append(meta("z:1", [2]))
        with self.assertRaises(UnsupportedOperator):
            lower_capture(graph)

    def test_output_dictionary_reserved_keys_are_not_reference_tags(self):
        graph = fixture("aten.neg.default", [[2]], [2])
        for key in ("ref", "dtype", "device", "memory_format"):
            with self.subTest(key=key):
                graph["outputs"] = {key: ref("z")}
                lowered = lower_capture(graph)
                self.assertEqual(lowered["outputs"], {key: ref("z")})
        graph["outputs"] = {"ref": {"ref": ref("z")}}
        self.assertEqual(lower_capture(graph)["outputs"], graph["outputs"])
        graph["outputs"] = {"ref": ref("missing")}
        with self.assertRaisesRegex(ValueError, "undefined value"):
            lower_capture(graph)

    def test_json_roundtrip_structure_and_input_not_mutated(self):
        graph = fixture("aten.mul.Tensor", [[2], [2]], [2])
        graph["outputs"] = {"pair": [ref("z"), ref("a")], "answer": 42}
        original = json.dumps(graph, sort_keys=True)
        lowered = lower_capture(graph)
        self.assertEqual(lowered["outputs"], graph["outputs"])
        self.assertEqual(json.dumps(graph, sort_keys=True), original)
        self.assertEqual(json.loads(json.dumps(lowered, allow_nan=False)), lowered)
        self.assertEqual(lowered["format"], "semantic-graph-v1")


if __name__ == "__main__":
    unittest.main()
