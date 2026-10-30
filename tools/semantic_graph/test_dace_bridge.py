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

    def test_unroll_budget(self):
        f = Fixture(x=([4], "float32"))
        f.node("clone", [ref("x")], [4], "float32", [read("x", ["i0"])])
        sdfg, manifest = lower_to_sdfg(f.graph)
        with self.assertRaisesRegex(BridgeUnsupported, "budget"):
            unroll_and_check(f.graph, sdfg, manifest, max_instances=3)


if __name__ == "__main__":
    unittest.main()
