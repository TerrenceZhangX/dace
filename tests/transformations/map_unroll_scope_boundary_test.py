"""Regression tests: MapUnroll must replicate a scope, not its surroundings.

These structural/data-relation tests need no C++ compiler. Every transformation
and every result is validated; no isolated nodes or connectors are deleted to
make an invalid transformation appear valid.
"""
from collections import Counter
import copy

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import MapUnroll


def _counts(state):
    return Counter(n.label for n in state.nodes() if isinstance(n, nodes.Tasklet))


def _unroll(graph):
    graph.validate()
    count = graph.apply_transformations_repeated(MapUnroll, validate=True, validate_all=True)
    graph.validate()
    assert all(not isinstance(n, nodes.MapEntry) for s in graph.states() for n in s.nodes())
    return count


def _shared_input():
    graph = dace.SDFG("unroll_shared_input")
    for name in ("A", "B", "C"):
        graph.add_array(name, [3], dace.float64)
    state = graph.add_state()
    a = state.add_read("A")
    for dest, expr in (("B", "x+1"), ("C", "x*2")):
        state.add_mapped_tasklet("compute_" + dest, {"i": "0:3"},
            {"x": dace.Memlet("A[i]")}, f"y = {expr}", {"y": dace.Memlet(f"{dest}[i]")},
            input_nodes={"A": a}, external_edges=True)
    return graph, state, a


def test_map_unroll_retains_shared_input_and_sibling_map():
    graph, state, a = _shared_input()
    _unroll(graph)
    assert a in state.nodes()
    assert state.out_degree(a) == 6
    assert _counts(state) == {"compute_B": 3, "compute_C": 3}
    for target in ("compute_B", "compute_C"):
        reads = [str(e.data.subset) for e in state.out_edges(a) if e.dst.label == target]
        assert sorted(reads) == ["0", "1", "2"]


def _join(wcr=False):
    graph = dace.SDFG("unroll_downstream_join")
    for name in ("A", "B", "T"):
        graph.add_array(name, [3], dace.float64, transient=name == "T")
    graph.add_array("C", [1] if wcr else [3], dace.float64)
    state = graph.add_state()
    a, b, tmp, c = state.add_read("A"), state.add_read("B"), state.add_access("T"), state.add_write("C")
    state.add_mapped_tasklet("producer", {"i": "0:3"}, {"x": dace.Memlet("A[i]")}, "y = x+1",
        {"y": dace.Memlet("T[i]")}, input_nodes={"A": a}, output_nodes={"T": tmp}, external_edges=True)
    result = dace.Memlet("C[0]" if wcr else "C[j]", wcr="lambda a,b: a+b" if wcr else None)
    state.add_mapped_tasklet("consumer", {"j": "0:3"}, {"x": dace.Memlet("T[j]"), "b": dace.Memlet("B[j]")}, "y = x+b",
        {"y": result}, input_nodes={"T": tmp, "B": b}, output_nodes={"C": c}, external_edges=True)
    return graph, state, tmp, b, c


def test_map_unroll_preserves_downstream_join_and_external_input():
    graph, state, tmp, b, c = _join()
    _unroll(graph)
    assert all(n in state.nodes() for n in (tmp, b, c))
    assert _counts(state) == {"producer": 3, "consumer": 3}
    assert state.in_degree(tmp) == state.out_degree(tmp) == 3
    assert state.out_degree(b) == 3
    assert all(e.dst.label == "consumer" for e in state.out_edges(b))


def test_map_unroll_does_not_duplicate_downstream_wcr():
    graph, state, tmp, b, c = _join(wcr=True)
    _unroll(graph)
    assert _counts(state) == {"producer": 3, "consumer": 3}
    contributions = state.in_edges(c)
    assert len(contributions) == 3  # Not N*N downstream reductions.
    assert all(e.data.wcr and e.src.label == "consumer" for e in contributions)
    assert all(str(e.data.subset) == "0" for e in contributions)
    assert sorted(str(e.data.subset) for e in state.out_edges(b)) == ["0", "1", "2"]


def test_map_unroll_retains_sibling_direct_copy():
    graph, state, a = _shared_input()
    graph.add_array("D", [3], dace.float64)
    d = state.add_write("D")
    state.add_edge(a, None, d, None, dace.Memlet("A[0:3]", other_subset="0:3"))
    _unroll(graph)
    edges = state.edges_between(a, d)
    assert len(edges) == 1
    assert str(edges[0].data.subset) == "0:3"


def _local_transient_graph(collision=False):
    graph = dace.SDFG("unroll_local_temporary")
    graph.add_array("A", [3], dace.float64)
    graph.add_array("B", [3], dace.float64)
    graph.add_array("tmp", [1], dace.float64, transient=True)
    if collision:
        graph.add_array("tmp_0", [7], dace.float64, transient=False)
    state = graph.add_state()
    a, b = state.add_read("A"), state.add_write("B")
    entry, exit_ = state.add_map("outer", {"i": "0:3"})
    first = state.add_tasklet("first", {"x"}, {"y"}, "y = x+1")
    temp = state.add_access("tmp")
    second = state.add_tasklet("second", {"x"}, {"y"}, "y = x*2")
    state.add_memlet_path(a, entry, first, memlet=dace.Memlet("A[i]"), dst_conn="x")
    state.add_edge(first, "y", temp, None, dace.Memlet("tmp[0]"))
    state.add_edge(temp, None, second, "x", dace.Memlet("tmp[0]"))
    state.add_memlet_path(second, exit_, b, memlet=dace.Memlet("B[i]"), src_conn="y")
    return graph, state


def test_map_unroll_replicates_only_scope_local_temporaries():
    graph, state = _local_transient_graph()
    _unroll(graph)
    temps = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data.startswith("tmp")]
    assert len(temps) == 3
    assert len({n.data for n in temps}) == 3
    assert "tmp" not in graph.arrays
    for temp in temps:
        assert graph.arrays[temp.data].transient
        assert state.in_degree(temp) == state.out_degree(temp) == 1
        producer, consumer = state.in_edges(temp)[0].src, state.out_edges(temp)[0].dst
        assert str(state.in_edges(producer)[0].data.subset) == str(state.out_edges(consumer)[0].data.subset)


def test_map_unroll_nested_sdfgs_are_distinct_and_correctly_parented():
    graph = dace.SDFG("unroll_nested_parent")
    graph.add_array("A", [3], dace.float64)
    graph.add_array("B", [3], dace.float64)
    state = graph.add_state()
    a, b = state.add_read("A"), state.add_write("B")
    entry, exit_ = state.add_map("outer", {"i": "0:3"})
    inner = dace.SDFG("inner")
    inner.add_array("x", [1], dace.float64)
    inner.add_array("y", [1], dace.float64)
    s = inner.add_state()
    t = s.add_tasklet("inner_compute", {"a"}, {"b"}, "b=a+1")
    s.add_edge(s.add_read("x"), None, t, "a", dace.Memlet("x[0]"))
    s.add_edge(t, "b", s.add_write("y"), None, dace.Memlet("y[0]"))
    nested = state.add_nested_sdfg(inner, {"x"}, {"y"})
    state.add_memlet_path(a, entry, nested, memlet=dace.Memlet("A[i]"), dst_conn="x")
    state.add_memlet_path(nested, exit_, b, memlet=dace.Memlet("B[i]"), src_conn="y")
    nested.integrate_into_parent()
    _unroll(graph)
    copies = [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]
    assert len(copies) == len({id(n.sdfg) for n in copies}) == 3
    for n in copies:
        assert n.sdfg.parent is state
        assert n.sdfg.parent_sdfg is graph
        assert n.sdfg.parent_nsdfg_node is n
        n.sdfg.validate()


def test_map_unroll_nested_maps_keep_single_body_per_coordinate():
    graph = dace.SDFG("unroll_nested_maps")
    graph.add_array("A", [3, 2], dace.float64)
    graph.add_array("B", [3, 2], dace.float64)
    state = graph.add_state()
    a, b = state.add_read("A"), state.add_write("B")
    outer, outer_exit = state.add_map("outer", {"i": "0:3"})
    inner, inner_exit = state.add_map("inner", {"j": "0:2"})
    t = state.add_tasklet("copy", {"x"}, {"y"}, "y=x")
    state.add_memlet_path(a, outer, inner, t, memlet=dace.Memlet("A[i,j]"), dst_conn="x")
    state.add_memlet_path(t, inner_exit, outer_exit, b, memlet=dace.Memlet("B[i,j]"), src_conn="y")
    _unroll(graph)
    assert _counts(state) == {"copy": 6}
    expected = {f"{i}, {j}" for i in range(3) for j in range(2)}
    assert {str(e.data.subset) for e in state.out_edges(a)} == expected
    assert {str(e.data.subset) for e in state.in_edges(b)} == expected


def test_map_unroll_local_names_do_not_collide_with_external_arrays():
    graph, state = _local_transient_graph(collision=True)
    _unroll(graph)
    assert graph.arrays["tmp_0"].shape == (7,)
    assert not graph.arrays["tmp_0"].transient
    temps = [n.data for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data.startswith("tmp")]
    assert len(set(temps)) == 3
    assert "tmp_0" not in temps
    assert all(graph.arrays[name].shape == (1,) for name in temps)


def test_map_unroll_nondivisible_positive_strides():
    for bounds, expected in (((0, 5, 2), {"0", "2", "4"}), ((1, 5, 3), {"1", "4"})):
        graph = dace.SDFG("unroll_stride")
        graph.add_array("A", [6], dace.float64)
        graph.add_array("B", [6], dace.float64)
        state = graph.add_state()
        a, b = state.add_read("A"), state.add_write("B")
        entry, exit_ = state.add_map("stride", {"i": bounds})
        task = state.add_tasklet("copy", {"x"}, {"y"}, "y=x")
        state.add_memlet_path(a, entry, task, memlet=dace.Memlet("A[i]"), dst_conn="x")
        state.add_memlet_path(task, exit_, b, memlet=dace.Memlet("B[i]"), src_conn="y")
        _unroll(graph)
        assert _counts(state) == {"copy": len(expected)}
        assert {str(e.data.subset) for e in state.out_edges(a)} == expected


def test_map_unroll_refuses_empty_and_gpu_scopes_without_changes():
    for empty, gpu in ((True, False), (False, True)):
        graph, state, a = _shared_input()
        for n in state.nodes():
            if isinstance(n, nodes.MapEntry):
                if empty:
                    n.map.range = dace.subsets.Range([(0, -1, 1)])
                if gpu:
                    n.map.schedule = dace.ScheduleType.GPU_Device
        before = graph.to_json()
        assert graph.apply_transformations_repeated(MapUnroll, validate=True, validate_all=True) == 0
        assert graph.to_json() == before


def test_map_unroll_refuses_dynamic_map_inputs_without_changes():
    graph, state, a = _shared_input()
    graph.add_scalar("dynamic", dace.int64)
    source = state.add_read("dynamic")
    for n in list(state.nodes()):
        if isinstance(n, nodes.MapEntry):
            n.add_in_connector("bound")
            state.add_edge(source, None, n, "bound", dace.Memlet("dynamic[0]"))
    graph.validate()
    before = graph.to_json()
    assert graph.apply_transformations_repeated(MapUnroll, validate=True, validate_all=True) == 0
    assert graph.to_json() == before
