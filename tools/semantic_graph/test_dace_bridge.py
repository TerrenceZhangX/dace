"""Public synthetic semantic fixtures only; no benchmark source or private data."""
from copy import deepcopy
import unittest

import dace
from tools.semantic_graph.dace_bridge import BridgeUnsupported, lower_to_sdfg, unroll_and_check


def ref(name): return {"ref": name}


class Fixture:
    def __init__(self, **inputs):
        self.graph = {"format": "semantic-graph-v1", "inputs": [], "nodes": [], "constants": {}, "outputs": None}
        for name, (shape, dtype) in inputs.items():
            self.graph["inputs"].append({"id": name, "name": name, "kind": "tensor", "shape": shape, "dtype": dtype})

    def node(self, op, args, shape, dtype, reads, attrs=None, reduction=None):
        name = "node" + str(len(self.graph["nodes"]))
        self.graph["nodes"].append({"id": name, "op": op, "args": args, "kwargs": {}, "attributes": attrs or {},
                                    "outputs": [{"id": name, "kind": "tensor", "shape": shape, "dtype": dtype}],
                                    "semantics": {"domain": [{"var": f"i{j}", "extent": n} for j, n in enumerate(shape)],
                                                  "order": "data-only", "reads": reads, "reduction": reduction}})
        self.graph["outputs"] = ref(name)
        return name


def read(name, indices, guard=None):
    item = {"value": name, "indices": indices}
    if guard is not None: item["guard"] = guard
    return item


def reduction(extent, op="sum", axes=None):
    return {"operator": op, "variables": [{"var": "r0", "extent": extent}],
            "axes": axes or [1], "group_size": extent, "tree": "unspecified"}


class BridgeTests(unittest.TestCase):
    def check(self, fixture, finite_only=False):
        sdfg, manifest = lower_to_sdfg(fixture.graph, "generic_bridge_test", finite_only=finite_only)
        sdfg.validate()
        code = "\n".join(unit.code for unit in sdfg.generate_code())
        expanded, result = unroll_and_check(fixture.graph, sdfg, manifest)
        expanded.validate()
        self.assertTrue(result["pass"])
        return code, result, manifest

    def test_broadcast_sum_mean(self):
        f = Fixture(x=([2, 1, 3], "float32"), bias=([1, 4, 1], "float32"))
        n = f.node("add", [ref("x"), ref("bias")], [2, 4, 3], "float32",
                   [read("x", ["i0", "0", "i2"]), read("bias", ["0", "i1", "0"])], {"alpha": 1})
        f.node("sum", [ref(n), [1]], [2, 3], "float32", [read(n, ["i0", "r0", "i1"])], reduction=reduction(4))
        f.node("mean", [ref(n), [1]], [2, 3], "float32", [read(n, ["i0", "r0", "i1"])], reduction=reduction(4))
        code, result, _ = self.check(f)
        self.assertIn("ReductionType::Sum", code)
        self.assertEqual(result["semantic_scalar_outputs"], 36)

    def test_matrix_and_layout(self):
        f = Fixture(a=([2, 3], "float32"), b=([3, 4], "float32"))
        m = f.node("mm", [ref("a"), ref("b")], [2, 4], "float32",
                   [read("a", ["i0", "r0"]), read("b", ["r0", "i1"])],
                   reduction=reduction(3, "sum-of-products"))
        t = f.node("transpose", [ref(m), 0, 1], [4, 2], "float32", [read(m, ["i1", "i0"])])
        f.node("reshape", [ref(t), [8]], [8], "float32", [read(t, ["i0 // 2", "i0 % 2"])])
        self.check(f)

    def test_bfloat16_rounding_and_float_accumulation(self):
        f = Fixture(x=([2, 3], "float32"))
        c = f.node("cast", [ref("x")], [2, 3], "bfloat16", [read("x", ["i0", "i1"])], {"dtype": "bfloat16"})
        a = f.node("add", [ref(c), 0.003], [2, 3], "bfloat16", [read(c, ["i0", "i1"])], {"alpha": 1})
        f.node("sum", [ref(a), [1]], [2], "bfloat16", [read(a, ["i0", "r0"])], reduction=reduction(3))
        code, _, _ = self.check(f)
        self.assertIn("float(0.003)", code)
        self.assertIn("dace::bfloat16", code)
        self.assertIn("ReductionType::Sum, float", code)

    def test_zeros_have_no_value_dependencies(self):
        f = Fixture(x=([2, 3], "float16"))
        f.node("zeros_like", [ref("x")], [2, 3], "float32", [], {"dtype": "float32"})
        _, result, _ = self.check(f)
        self.assertEqual(result["semantic_scalar_outputs"], 6)

    def test_finite_maximum_and_amax(self):
        f = Fixture(x=([2, 3], "float32"), y=([3], "float32"))
        m = f.node("maximum", [ref("x"), ref("y")], [2, 3], "float32",
                   [read("x", ["i0", "i1"]), read("y", ["i1"])])
        f.node("amax", [ref(m), [1]], [2], "float32", [read(m, ["i0", "r0"])], reduction=reduction(3, "max"))
        with self.assertRaisesRegex(BridgeUnsupported, "finite_only"):
            lower_to_sdfg(f.graph)
        _, _, manifest = self.check(f, finite_only=True)
        self.assertTrue(manifest["finite_only_precondition"])

    def test_softmax_row_dependencies_and_float_opmath(self):
        for shape, axis, dtype in (([2, 3], 1, "bfloat16"), ([2, 3], 0, "float32"), ([3], 0, "float64")):
            with self.subTest(shape=shape, axis=axis, dtype=dtype):
                f = Fixture(x=(shape, dtype))
                indices = [f"i{j}" if j != axis else "r0" for j in range(len(shape))]
                f.node("softmax", [ref("x"), axis, False], shape, dtype, [read("x", indices)],
                       {"dim": axis}, reduction=reduction(shape[axis], "softmax", [axis]))
                code, _, _ = self.check(f, finite_only=True)
                self.assertIn("ReductionType::Sum", code)
                self.assertIn("exp", code)

    def test_select_scatter_exclusive_old_source_reads(self):
        f = Fixture(base=([3, 4], "float32"), row=([4], "float32"))
        f.node("select_scatter", [ref("base"), ref("row"), 0, 1], [3, 4], "float32",
               [read("base", ["i0", "i1"], "not (i0 == 1)"), read("row", ["i1"], "i0 == 1")],
               {"dim": 0, "index": 1})
        self.check(f)

    def test_strided_slice_scatter_no_unselected_source_access(self):
        f = Fixture(base=([2, 7], "float32"), src=([2, 2], "float32"))
        guard = "i1 >= 1 and i1 < 6 and (i1 - 1) % 3 == 0"
        f.node("slice_scatter", [ref("base"), ref("src"), 1, 1, 6, 3], [2, 7], "float32",
               [read("base", ["i0", "i1"], f"not ({guard})"), read("src", ["i0", "(i1-1)//3"], guard)],
               {"dim": 1, "start": 1, "end": 6, "step": 3})
        self.check(f)

    def test_invalid_index_and_unknown_op_fail_closed(self):
        f = Fixture(x=([2], "float32"))
        f.node("clone", [ref("x")], [2], "float32", [read("x", ["__import__('os')"])])
        with self.assertRaises(BridgeUnsupported): lower_to_sdfg(f.graph)
        f.graph["nodes"][0]["op"] = "unknown"
        with self.assertRaises(BridgeUnsupported): lower_to_sdfg(f.graph)

    def test_tampered_scatter_guard_detected_by_native_cone_check(self):
        f = Fixture(base=([3], "float32"), src=([], "float32"))
        f.node("select_scatter", [ref("base"), ref("src"), 0, 1], [3], "float32",
               [read("base", ["i0"], "i0 != 2"), read("src", [], "i0 == 2")], {"dim": 0, "index": 1})
        sdfg, manifest = lower_to_sdfg(f.graph)
        with self.assertRaisesRegex(BridgeUnsupported, "cone mismatch"):
            unroll_and_check(f.graph, sdfg, manifest)

    def test_cat_repeated_segments_and_empty_skip(self):
        f = Fixture(a=([2, 2], "float32"), empty=([0], "float32"), b=([2, 1], "float32"))
        segments = [{"value": "a", "start": 0, "end": 2}, {"value": "b", "start": 2, "end": 3},
                    {"value": "a", "start": 3, "end": 5}]
        f.node("cat", [[ref("a"), ref("empty"), ref("b"), ref("a")], 1], [2, 5], "float32",
               [read("a", ["i0", "i1"], "i1 < 2"), read("b", ["i0", "i1-2"], "i1 >= 2 and i1 < 3"),
                read("a", ["i0", "i1-3"], "i1 >= 3")], {"dim": 1, "segments": segments})
        _, result, _ = self.check(f)
        counts = result["arithmetic_occurrences"]
        self.assertEqual(counts["tensor_read_occurrences"], {"op0:a": 8, "op0:b": 2})
        self.assertEqual(counts["totals"], {"cast:float32": 10})

    def test_padding_literal_regions_negative_crop(self):
        f = Fixture(x=([3, 4], "bfloat16"))
        guard = "i0 >= 1 and i0 < 4 and i1 < 3"
        f.node("constant_pad_nd", [ref("x"), [-1, 2, 1, 0], 0.003], [4, 5], "bfloat16",
               [read("x", ["i0-1", "i1+1"], guard)],
               {"pad_before": [1, -1], "pad_after": [0, 2], "fill_value": 0.003})
        f.graph["nodes"][-1]["semantics"]["constant_regions"] = [{"guard": f"not ({guard})", "value": 0.003}]
        code, result, _ = self.check(f)
        self.assertEqual(result["arithmetic_occurrences"]["tensor_read_occurrences"], {"op0:x": 9})
        self.assertEqual(result["arithmetic_occurrences"]["totals"], {"cast:bfloat16": 20})
        self.assertIn("dace::bfloat16(0.003)", code)

    def test_addmm_bias_once_and_alpha_beta_once(self):
        f = Fixture(bias=([3], "bfloat16"), a=([2, 4], "bfloat16"), b=([4, 3], "bfloat16"))
        f.node("addmm", [ref("bias"), ref("a"), ref("b")], [2, 3], "bfloat16",
               [read("a", ["i0", "r0"]), read("b", ["r0", "i1"]), read("bias", ["i1"])],
               {"alpha": 0.25, "beta": 2.0}, reduction=reduction(4, "sum-of-products"))
        _, result, _ = self.check(f)
        counts = result["arithmetic_occurrences"]
        self.assertEqual(counts["tensor_read_occurrences"], {"op0:a": 24, "op0:b": 24, "op0:bias": 6})
        self.assertEqual(counts["totals"]["mul"], 36)  # 24 products + 6 alpha + 6 beta
        self.assertEqual(counts["totals"]["wcr:sum"], 24)
        self.assertEqual(counts["totals"]["add"], 6)
        self.assertEqual(counts["per_tasklet"]["op0_finish"]["mul"], 12)

    def test_addmm_zero_beta_and_empty_contraction(self):
        for k, beta in ((3, 0), (0, 2), (0, 0)):
            with self.subTest(k=k, beta=beta):
                f = Fixture(bias=([2, 2], "float32"), a=([2, k], "float32"), b=([k, 2], "float32"))
                reads = [read("a", ["i0", "r0"]), read("b", ["r0", "i1"])] if k else []
                if beta: reads.append(read("bias", ["i0", "i1"]))
                f.node("addmm", [ref("bias"), ref("a"), ref("b")], [2, 2], "float32", reads,
                       {"alpha": 0.5 if k else float("inf"), "beta": beta},
                       reduction=reduction(k, "sum-of-products") if k else None)
                _, result, _ = self.check(f)
                counts = result["arithmetic_occurrences"]
                self.assertEqual(counts["tensor_read_occurrences"].get("op0:bias", 0), 4 if beta else 0)
                self.assertEqual(counts["totals"].get("wcr:sum", 0), 4*k)
                if not k:
                    self.assertEqual(counts["totals"].get("mul", 0), 4 if beta else 0)

    def test_gelu_erf_and_tanh_arithmetic_counts(self):
        for mode, special, multiplications, additions in (("none", "erf", 3, 1), ("tanh", "tanh", 6, 2)):
            with self.subTest(mode=mode):
                f = Fixture(x=([2, 3], "bfloat16"))
                f.node("gelu", [ref("x")], [2, 3], "bfloat16", [read("x", ["i0", "i1"])], {"approximate": mode})
                code, result, _ = self.check(f)
                counts = result["arithmetic_occurrences"]["totals"]
                self.assertEqual(counts["call:" + special], 6)
                self.assertEqual(counts["mul"], 6*multiplications)
                self.assertEqual(counts["add"], 6*additions)
                self.assertEqual(counts["cast:bfloat16"], 6)
                self.assertIn(special, code)

    def test_unsupported_gelu_and_nonfinite_padding(self):
        f = Fixture(x=([2], "float32"))
        f.node("gelu", [ref("x")], [2], "float32", [read("x", ["i0"])], {"approximate": "fast"})
        with self.assertRaisesRegex(BridgeUnsupported, "approximate"):
            lower_to_sdfg(f.graph)
        f.graph["nodes"][0].update(op="constant_pad_nd", attributes={"pad_before": [0], "pad_after": [0], "fill_value": float("inf")})
        with self.assertRaisesRegex(BridgeUnsupported, "finite"):
            lower_to_sdfg(f.graph)

    def test_unroll_budget(self):
        f = Fixture(x=([4], "float32"))
        f.node("clone", [ref("x")], [4], "float32", [read("x", ["i0"])])
        sdfg, manifest = lower_to_sdfg(f.graph)
        with self.assertRaisesRegex(BridgeUnsupported, "budget"):
            unroll_and_check(f.graph, sdfg, manifest, max_instances=3)


if __name__ == "__main__":
    unittest.main()
