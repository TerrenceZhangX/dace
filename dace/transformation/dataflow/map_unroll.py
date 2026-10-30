# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import itertools

from dace import SDFG, __version__, symbolic
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation


class MapUnroll(transformation.SingleStateTransformation):
    """
    Unrolls a map with constant ranges in the top-level scope of an SDFG by
    replicating its subgraph for each iteration. If there are local data
    containers only used in this map, they will also be replicated, as will
    nested SDFGs found within.

    Only the selected scope is replicated; external access nodes, sibling
    branches and downstream consumers are retained. Maps with nonpositive or
    empty ranges, dynamic map inputs, GPU schedules, or scope-local streams and
    views/references are not supported by this transformation.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry
        # Must be top-level map
        if graph.scope_dict()[map_entry] is not None:
            return False
        # TrivialMapElimination cannot remove GPU or dynamically-fed scopes.
        from dace import dtypes
        if map_entry.map.schedule in dtypes.GPU_SCHEDULES:
            return False
        if any(not e.data.is_empty() and (e.dst_conn is None or not e.dst_conn.startswith("IN_"))
               for e in graph.in_edges(map_entry)):
            return False
        # Empty-map elimination needs separate boundary/lifetime handling.
        try:
            for begin, end, step in map_entry.map.range:
                values = [symbolic.evaluate(v, sdfg.constants) for v in (begin, end, step)]
                if any(int(v) != v for v in values):
                    return False
                begin, end, step = map(int, values)
                if step <= 0 or begin > end:
                    return False
        except (TypeError, ValueError, OverflowError):
            return False
        scope = graph.scope_subgraph(map_entry)
        for name in sdutil.local_transients(sdfg, scope, entry_node=map_entry, include_nested=True):
            if isinstance(sdfg.arrays[name], (dt.Stream, dt.View, dt.Reference)):
                return False
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        from dace.transformation.dataflow import TrivialMapElimination

        map_entry = self.map_entry
        if not self.can_be_applied(state, 0, sdfg):
            raise ValueError("MapUnroll requires a supported top-level constant nonempty map")

        # Copy only this scope. External access nodes and downstream consumers
        # can serve other branches and must neither be copied nor removed.
        subgraph = state.scope_subgraph(map_entry)
        map_exit = state.exit_node(map_entry)
        internal_edges = list(subgraph.edges())
        boundary_edges = list(state.in_edges(map_entry)) + list(state.out_edges(map_exit))

        # Save nested SDFGs to JSON, then deserialize them for every copy we
        # need to make
        nested_sdfgs = {}
        for node in subgraph:
            if isinstance(node, nodes.NestedSDFG):
                nested_sdfgs[node.sdfg] = node.sdfg.to_json()

        # Check for local memories that need to be replicated
        local_memories = [
            name
            for name in sdutil.local_transients(sdfg, subgraph, entry_node=map_entry, include_nested=True)
            if not isinstance(sdfg.arrays[name], dt.Stream) and not isinstance(sdfg.arrays[name], dt.View)
        ]

        ranges = map_entry.map.range.ranges
        constant_ranges = []
        for r in ranges:
            begin = symbolic.evaluate(r[0], sdfg.constants)
            end = symbolic.evaluate(r[1], sdfg.constants)
            step = symbolic.evaluate(r[2], sdfg.constants)
            # Inclusive bounds may be unaligned with the step.
            end += 1 if step > 0 else -1
            constant_ranges.append(range(int(begin), int(end), int(step)))
        index_tuples = itertools.product(*constant_ranges)
        for t in index_tuples:
            suffix = "_" + "_".join(map(str, t))
            node_to_unrolled = {}
            local_names = {name: sdfg.add_datadesc(name + suffix, copy.deepcopy(sdfg.arrays[name]),
                                                 find_new_name=True) for name in local_memories}
            # Copy all nodes
            for node in subgraph:
                if isinstance(node, nodes.NestedSDFG):
                    # Avoid deep-copying the nested SDFG
                    nsdfg = node.sdfg
                    # Don't copy the nested SDFG, as we will do this separately
                    node.sdfg = None
                    unrolled_node = copy.deepcopy(node)
                    node.sdfg = nsdfg
                    # Deserialize into a new SDFG specific to this copy
                    nsdfg_json = nested_sdfgs[nsdfg]
                    name = nsdfg_json["attributes"]["name"]
                    nsdfg_json["attributes"]["name"] += suffix
                    nsdfg_json["dace_version"] = __version__
                    unrolled_nsdfg = SDFG.from_json(nsdfg_json)
                    nsdfg_json["attributes"]["name"] = name  # Reinstate
                    # Set all the references
                    unrolled_nsdfg.parent = state
                    unrolled_nsdfg.parent_sdfg = sdfg
                    unrolled_nsdfg.update_cfg_list([])
                    unrolled_node.sdfg = unrolled_nsdfg
                    unrolled_nsdfg.parent_nsdfg_node = unrolled_node
                else:
                    unrolled_node = copy.deepcopy(node)
                    if node == map_entry:
                        # Fix the map bounds to only this iteration
                        unrolled_node.map.range = [(i, i, 1) for i in t]
                    if isinstance(node, nodes.AccessNode) and node.data in local_memories:
                        unrolled_node.data = local_names[node.data]
                state.add_node(unrolled_node)
                node_to_unrolled[node] = unrolled_node  # Remember mapping
            # Reconnect boundary edges to their original external endpoints.
            for src, src_conn, dst, dst_conn, memlet in internal_edges + boundary_edges:
                src = node_to_unrolled.get(src, src)
                dst = node_to_unrolled.get(dst, dst)
                memlet = copy.deepcopy(memlet)
                if memlet.data in local_memories:
                    memlet.data = local_names[memlet.data]
                state.add_edge(src, src_conn, dst, dst_conn, memlet)
            # Eliminate the now trivial map
            TrivialMapElimination.apply_to(
                sdfg, verify=False, annotate=False, save=False, map_entry=node_to_unrolled[map_entry]
            )

        # Now we can delete the original subgraph. This implicitly also remove
        # memlets between nodes
        state.remove_nodes_from(subgraph)

        # If we added a bunch of new nested SDFGs, reset the internal list
        if len(nested_sdfgs) > 0:
            sdfg.reset_cfg_list()

        # Remove local memories that were replicated
        for mem in local_memories:
            sdfg.remove_data(mem)
