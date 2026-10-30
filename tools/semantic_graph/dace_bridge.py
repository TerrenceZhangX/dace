"""Optional semantic-graph-v1 JSON -> native DaCe SDFG integration.

No PyTorch dependency, source execution, task-ID rules, or arbitrary ATen dispatch.
This baseline materializes packed logical SSA values; physical alias/layout ABI
and optimal schedules are deliberately not claimed. Numeric compilation/execution
must be validated separately from native SDFG validation and code generation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import itertools
import json
import math
import operator
from pathlib import Path

import dace
from dace.transformation.dataflow import MapUnroll


class BridgeUnsupported(ValueError):
    """Unsupported semantics, numeric policy, or malformed graph; fail closed."""


DTYPES = {"float32": dace.float32, "float64": dace.float64, "float16": dace.float16,
          "bfloat16": dace.bfloat16, "int32": dace.int32, "int64": dace.int64,
          "bool": dace.bool_}
FLOATS = {"float16", "bfloat16", "float32", "float64"}
POINTWISE = {"add", "sub", "rsub", "mul", "div", "pow", "tanh", "exp", "sigmoid",
             "sqrt", "rsqrt", "neg", "maximum", "gelu"}
LAYOUT = {"reshape", "view", "permute", "transpose", "expand", "slice", "select",
          "clone", "contiguous", "alias", "unsqueeze", "squeeze", "cast", "copy"}
REDUCE = {"sum", "mean", "mm", "bmm", "matmul", "amax"}
UPDATES = {"slice_scatter", "select_scatter"}
OPS = POINTWISE | LAYOUT | REDUCE | UPDATES | {"zeros_like", "softmax", "cat", "constant_pad_nd", "addmm"}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
        ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}


def refs(value):
    if isinstance(value, dict):
        if set(value) == {"ref"} and isinstance(value["ref"], str):
            yield value["ref"]
        else:
            for child in value.values():
                yield from refs(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from refs(child)


def index_eval(expression, env):
    """Evaluate only the documented integer coordinate/guard AST, never eval."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp):
            v = visit(node.operand)
            if isinstance(node.op, ast.USub): return -v
            if isinstance(node.op, ast.UAdd): return +v
            if isinstance(node.op, ast.Not): return not v
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [bool(visit(n)) for n in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            a = visit(node.left)
            for op, rhs in zip(node.ops, node.comparators):
                b = visit(rhs)
                if type(op) not in _CMP or not _CMP[type(op)](a, b): return False
                a = b
            return True
        if isinstance(node, ast.IfExp):
            return visit(node.body if visit(node.test) else node.orelse)
        raise BridgeUnsupported(f"unsupported index AST: {expression!r}")
    if type(expression) in (int, bool): return expression
    if not isinstance(expression, str) or len(expression) > 4096:
        raise BridgeUnsupported("invalid index expression")
    return visit(ast.parse(expression, mode="eval").body)


def _safe_index(expression, variables):
    # Validate syntax before symbolic parsing/code generation. Domain-specific
    # bounds are checked by native validation and the bounded exact cone check.
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
               ast.BoolOp, ast.Compare, ast.IfExp, *tuple(_BIN), *tuple(_CMP),
               ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or)
    tree = ast.parse(str(expression), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise BridgeUnsupported("unsupported coordinate/guard syntax")
        if isinstance(node, ast.Name) and node.id not in variables:
            raise BridgeUnsupported(f"undefined coordinate variable {node.id}")
        if isinstance(node, ast.Constant) and type(node.value) not in (int, bool):
            raise BridgeUnsupported("non-integer coordinate constant")


def validate_graph(graph):
    """Validate local structure; producer annotation authenticity is NOT proven.

    A producer may additionally pass its rule-regeneration verifier to
    lower_to_sdfg. This module intentionally does not depend on that producer.
    """
    if not isinstance(graph, dict) or graph.get("format") != "semantic-graph-v1":
        raise BridgeUnsupported("expected semantic-graph-v1")
    if not isinstance(graph.get("inputs"), list) or not isinstance(graph.get("nodes"), list):
        raise BridgeUnsupported("inputs/nodes must be lists")
    if graph.get("constants"):
        raise BridgeUnsupported("explicit tensor constant initialization is not implemented")
    table, node_ids = {}, set()
    def add(meta):
        if not isinstance(meta, dict) or not isinstance(meta.get("id"), str) or meta["id"] in table:
            raise BridgeUnsupported("invalid/duplicate value ID")
        if meta.get("kind", "tensor") == "scalar":
            if type(meta.get("specialized_value")) not in (bool, int, float):
                raise BridgeUnsupported("only specialized numeric Python scalars are supported")
        else:
            shape = meta.get("shape")
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise BridgeUnsupported("concrete nonnegative tensor shapes required")
            if meta.get("dtype") not in DTYPES:
                raise BridgeUnsupported("unsupported tensor dtype")
        table[meta["id"]] = meta
    for meta in graph["inputs"]: add(meta)
    for node in graph["nodes"]:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or node["id"] in node_ids:
            raise BridgeUnsupported("invalid/duplicate semantic node ID")
        node_ids.add(node["id"])
        if node.get("op") not in OPS:
            raise BridgeUnsupported(f"unsupported normalized op {node.get('op')!r}")
        if not isinstance(node.get("args"), list) or not isinstance(node.get("kwargs", {}), dict):
            raise BridgeUnsupported("invalid argument containers")
        for ref in refs([node["args"], node.get("kwargs", {})]):
            if ref not in table: raise BridgeUnsupported(f"unknown/forward reference {ref}")
        outputs, sem = node.get("outputs"), node.get("semantics")
        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(sem, dict):
            raise BridgeUnsupported("one typed tensor output and semantics required")
        if sem.get("order") != "data-only":
            raise BridgeUnsupported("semantic order must be data-only")
        domain = sem.get("domain")
        if not isinstance(domain, list) or [d.get("extent") for d in domain] != outputs[0].get("shape"):
            raise BridgeUnsupported("output domain disagrees with shape")
        variables = domain + (sem.get("reduction") or {}).get("variables", [])
        names = set()
        for var in variables:
            name = var.get("var")
            if not isinstance(name, str) or not name.isidentifier() or name.startswith("__") or name in names:
                raise BridgeUnsupported("invalid/duplicate iteration variable")
            if type(var.get("extent")) is not int or var["extent"] < 0:
                raise BridgeUnsupported("invalid iteration extent")
            names.add(name)
        for read in sem.get("reads", []):
            if read.get("value") not in table:
                raise BridgeUnsupported("unknown semantic read")
            if len(read.get("indices", [])) != len(table[read["value"]].get("shape", [])):
                raise BridgeUnsupported("semantic read rank mismatch")
            for expr in read["indices"]: _safe_index(expr, names)
            if "guard" in read: _safe_index(read["guard"], names)
        for region in sem.get("constant_regions", []):
            if not isinstance(region, dict) or "guard" not in region:
                raise BridgeUnsupported("invalid literal region")
            _safe_index(region["guard"], names)
        add(outputs[0])
    for ref in refs(graph.get("outputs")):
        if ref not in table: raise BridgeUnsupported("unknown output reference")
    return table


def _indices(indices):
    return ",".join(str(x) for x in indices) if indices else "0"


def _cast(expression, dtype):
    return f"dace.{dtype}({expression})"


def _expr(node, connector_by_ref, table, finite_only):
    op, attrs, args = node["op"], node.get("attributes", {}), node["args"]
    dtype = "float64" if node["outputs"][0]["dtype"] == "float64" else "float32"
    def value(arg):
        if isinstance(arg, dict) and set(arg) == {"ref"}:
            ref = arg["ref"]
            if ref in connector_by_ref: expression = connector_by_ref[ref]
            elif table[ref].get("kind") == "scalar": expression = repr(table[ref]["specialized_value"])
            else: raise BridgeUnsupported(f"missing tensor read {ref}")
        elif type(arg) in (int, float, bool) and (not isinstance(arg, float) or math.isfinite(arg)):
            expression = repr(arg)
        else: raise BridgeUnsupported(f"nonfinite/non-arithmetic operand {arg!r}")
        return _cast(expression, dtype)
    a = value(args[0])
    b = value(args[1]) if len(args) > 1 else None
    if op == "gelu":
        mode = attrs.get("approximate", node.get("kwargs", {}).get("approximate", "none"))
        if mode not in ("none", "tanh"):
            raise BridgeUnsupported("gelu approximate must be none or tanh")
        half, one = _cast("0.5", dtype), _cast("1", dtype)
        if mode == "none":
            return f"({half} * {a} * ({one} + erf({a} * {_cast(repr(math.sqrt(0.5)), dtype)})))"
        coefficient = _cast("0.044715", dtype)
        scale = _cast(repr(math.sqrt(2 / math.pi)), dtype)
        return f"({half} * {a} * ({one} + tanh({scale} * ({a} + {coefficient} * ({a} * {a} * {a})))))"
    if op in {"add", "sub", "rsub"}:
        alpha = value(attrs.get("alpha", 1))
        return {"add": f"({a} + {alpha} * {b})", "sub": f"({a} - {alpha} * {b})",
                "rsub": f"({b} - {alpha} * {a})"}[op]
    if op == "mul": return f"({a} * {b})"
    if op == "div":
        if attrs.get("rounding_mode") is not None: raise BridgeUnsupported("rounded division unsupported")
        return f"({a} / {b})"
    if op == "pow": return f"pow({a}, {b})"
    if op == "neg": return f"(-{a})"
    if op == "maximum":
        if not finite_only: raise BridgeUnsupported("maximum requires explicit finite_only policy")
        return f"({a} if {a} >= {b} else {b})"
    if op in {"tanh", "exp", "sqrt"}: return f"{op}({a})"
    if op == "sigmoid": return f"({_cast('1', dtype)} / ({_cast('1', dtype)} + exp(-{a})))"
    if op == "rsqrt": return f"({_cast('1', dtype)} / sqrt({a}))"
    raise BridgeUnsupported(f"unknown pointwise {op}")


def lower_to_sdfg(graph, name="semantic_bridge", *, finite_only=False, annotation_verifier=None):
    """Build native SDFG; finite_only is a caller-asserted execution precondition.

    Finite-only max/softmax excludes NaN/Inf and makes no signed-zero bit claim.
    It is recorded in the manifest, not silently assumed or enforced on runtime
    arrays. Physical inputs/outputs use packed logical tensors, not original views.
    """
    table = validate_graph(graph)
    if annotation_verifier is not None: annotation_verifier(graph)
    sdfg = dace.SDFG(name)
    names = {identifier: f"value{j}" for j, identifier in enumerate(table)}
    inputs_ids = {m["id"] for m in graph["inputs"]}
    outputs_ids = set(refs(graph.get("outputs")))
    manifest = {"array_to_value": {}, "logical_shapes": {}, "nodes": [],
                "scope": "packed logical-value ABI; explicit serial SSA-state schedule; CPU maps/WCR",
                "finite_only_precondition": finite_only, "native_numeric_execution": False,
                "producer_annotations_verified": annotation_verifier is not None}
    for identifier, meta in table.items():
        if meta.get("kind") == "scalar": continue
        sdfg.add_array(names[identifier], meta["shape"] or [1], DTYPES[meta["dtype"]],
                       transient=identifier not in inputs_ids | outputs_ids)
        manifest["array_to_value"][names[identifier]] = identifier
        manifest["logical_shapes"][names[identifier]] = meta["shape"]
    previous = None
    def state(label):
        nonlocal previous
        result = sdfg.add_state(label, is_start_state=previous is None)
        if previous is not None: sdfg.add_edge(previous, result, dace.InterstateEdge())
        previous = result
        return result
    def mem(array, indices, wcr=None):
        return dace.Memlet(f"{array}[{_indices(indices)}]", wcr=wcr)
    def emit(label, domain, inputs, expression, output_array, output_indices, dtype, wcr=None):
        st = state(label)
        if any(d["extent"] == 0 for d in domain): return
        ranges = {d["var"]: f"0:{d['extent']}" for d in domain} or {"scalar_iteration": "0:1"}
        st.add_mapped_tasklet(label, ranges, inputs, "out = " + _cast(expression, dtype),
                              {"out": mem(output_array, output_indices, wcr)},
                              schedule=dace.ScheduleType.Sequential, external_edges=True)
    def temporary(array, shape, dtype):
        sdfg.add_array(array, shape or [1], DTYPES[dtype], transient=True)

    for number, node in enumerate(graph["nodes"]):
        op, sem, attrs = node["op"], node["semantics"], node.get("attributes", {})
        out = node["outputs"][0]
        out_array, dtype = names[out["id"]], out["dtype"]
        domain = sem["domain"]
        indices = [d["var"] for d in domain]
        label = f"op{number}"
        manifest["nodes"].append({"node_id": node["id"], "op": op, "tasklet_prefix": label,
                                  "output": out_array, "output_value": out["id"]})
        if op == "cat":
            axis = attrs.get("dim")
            segments = attrs.get("segments")
            if type(axis) is not int or not 0 <= axis < len(domain) or not isinstance(segments, list):
                raise BridgeUnsupported("cat needs normalized dim and segments")
            offset = 0
            for j, segment in enumerate(segments):
                identifier = segment.get("value")
                if identifier not in table or table[identifier].get("kind") == "scalar":
                    raise BridgeUnsupported("cat segment requires tensor source")
                shape = table[identifier]["shape"]
                if len(shape) != len(domain) or table[identifier]["dtype"] != dtype:
                    raise BridgeUnsupported("cat requires same-rank/same-dtype segments")
                if any(n != out["shape"][k] for k, n in enumerate(shape) if k != axis):
                    raise BridgeUnsupported("cat non-concatenated extents disagree")
                if segment.get("start") != offset or segment.get("end") != offset + shape[axis]:
                    raise BridgeUnsupported("cat segment gap, overlap, or wrong extent")
                part = deepcopy(domain); part[axis]["extent"] = shape[axis]
                output_coords = list(indices); output_coords[axis] = f"({indices[axis]}+{offset})"
                emit(label + f"_segment{j}", part, {"x": mem(names[identifier], indices)}, "x",
                     out_array, output_coords, dtype)
                offset += shape[axis]
            if offset != out["shape"][axis]: raise BridgeUnsupported("cat segments do not cover output")
            continue
        if op == "constant_pad_nd":
            source_id = node["args"][0]["ref"]
            source_shape = table[source_id]["shape"]
            before, after = attrs.get("pad_before"), attrs.get("pad_after")
            fill = attrs.get("fill_value", 0)
            if (not isinstance(before, list) or not isinstance(after, list) or
                    len(before) != len(domain) or len(after) != len(domain) or
                    any(type(v) is not int for v in before + after)):
                raise BridgeUnsupported("padding needs normalized before/after integers")
            if len(source_shape) != len(domain) or table[source_id]["dtype"] != dtype:
                raise BridgeUnsupported("padding cannot change rank or dtype")
            if any(n + a + b != m or n + min(a, 0) + min(b, 0) < 0
                   for n, a, b, m in zip(source_shape, before, after, out["shape"])):
                raise BridgeUnsupported("invalid padding/cropping extents")
            if type(fill) not in (int, float, bool) or not math.isfinite(fill):
                raise BridgeUnsupported("padding currently requires finite numeric fill")
            if dtype in {"int32", "int64"} and not -(2 ** (int(dtype[3:]) - 1)) <= fill < 2 ** (int(dtype[3:]) - 1):
                raise BridgeUnsupported("padding fill outside integer range")
            # At most 2*rank exterior slabs and one interior rectangle. Prefix
            # dimensions are progressively restricted so regions never overlap.
            lower = [max(0, p) for p in before]
            upper = [min(m, p + n) for m, p, n in zip(out["shape"], before, source_shape)]
            lower = [min(m, x) for m, x in zip(out["shape"], lower)]
            upper = [max(l, u) for l, u in zip(lower, upper)]
            bounds = [[0, n] for n in out["shape"]]
            def pad_region(suffix, region, literal):
                part = deepcopy(domain)
                for j, (lo, hi) in enumerate(region): part[j]["extent"] = hi - lo
                coords = [f"({v}+{region[j][0]})" for j, v in enumerate(indices)]
                reads = {} if literal else {"x": mem(names[source_id], [f"({v}-({before[j]}))" for j, v in enumerate(coords)])}
                emit(label + suffix, part, reads, repr(fill) if literal else "x", out_array, coords, dtype)
            for axis in range(len(domain)):
                for side, lo, hi in (("low", bounds[axis][0], lower[axis]), ("high", upper[axis], bounds[axis][1])):
                    box = deepcopy(bounds); box[axis] = [lo, hi]
                    pad_region(f"_fill{axis}_{side}", box, True)
                bounds[axis] = [lower[axis], upper[axis]]
            pad_region("_interior", bounds, False)
            continue
        if op == "addmm":
            args = node["args"]
            if len(args) != 3 or dtype not in FLOATS or len(domain) != 2:
                raise BridgeUnsupported("addmm requires bias, rank-2 matrices, and floating output")
            bias_id, left_id, right_id = (arg["ref"] for arg in args)
            left, right, bias = (table[k]["shape"] for k in (left_id, right_id, bias_id))
            if len(left) != 2 or len(right) != 2 or left[1] != right[0] or out["shape"] != [left[0], right[1]]:
                raise BridgeUnsupported("addmm matrix shape mismatch")
            if any(table[k]["dtype"] != dtype for k in (bias_id, left_id, right_id)):
                raise BridgeUnsupported("addmm operand dtypes must match")
            if len(bias) > 2 or any(n != 1 and n != out["shape"][2-len(bias)+j] for j, n in enumerate(bias)):
                raise BridgeUnsupported("addmm bias cannot broadcast to output")
            alpha, beta, k = attrs.get("alpha", 1), attrs.get("beta", 1), left[1]
            if type(beta) not in (int, float) or not math.isfinite(beta):
                raise BridgeUnsupported("addmm beta must be finite")
            if k and (type(alpha) not in (int, float) or not math.isfinite(alpha)):
                raise BridgeUnsupported("nonempty addmm alpha must be finite")
            comp = "float64" if dtype == "float64" else "float32"
            final_inputs = {}
            final = _cast("0", comp)
            if k:
                accum = f"accumulator{number}"
                temporary(accum, out["shape"], comp)
                emit(label + "_init", domain, {}, "0", accum, indices, comp)
                reduction_domain = domain + [{"var": "r0", "extent": k}]
                emit(label + "_products", reduction_domain,
                     {"a": mem(names[left_id], [indices[0], "r0"]), "b": mem(names[right_id], ["r0", indices[1]])},
                     f"({_cast('a', comp)} * {_cast('b', comp)})", accum, indices, comp, "lambda x, y: x + y")
                final_inputs["acc"] = mem(accum, indices)
                final = f"({_cast(repr(alpha), comp)} * acc)"
            if beta != 0:
                bias_indices = ["0" if n == 1 else indices[2-len(bias)+j] for j, n in enumerate(bias)]
                final_inputs["bias"] = mem(names[bias_id], bias_indices)
                bias_term = f"({_cast(repr(beta), comp)} * {_cast('bias', comp)})"
                final = f"({final} + {bias_term})" if k else bias_term
            # Bias is loaded and multiplied once per output, never once per K.
            # beta=0 omits its read entirely; K=0 never multiplies alpha by zero.
            emit(label + "_finish", domain, final_inputs, final, out_array, indices, dtype)
            continue
        if op in UPDATES:
            # Disjoint rectangles implement the semantic guards. No unselected
            # source value is loaded, even at coordinates outside its bounds.
            if len(node["args"]) < 2: raise BridgeUnsupported("scatter needs base and source")
            base_id, source_id = node["args"][0]["ref"], node["args"][1]["ref"]
            base, src = names[base_id], names[source_id]
            source_shape = table[source_id]["shape"]
            axis = attrs["dim"]
            if type(axis) is not int or not 0 <= axis < len(domain): raise BridgeUnsupported("invalid scatter axis")
            if op == "select_scatter":
                start, count, step = attrs["index"], 1, 1
                if source_shape != out["shape"][:axis] + out["shape"][axis+1:]:
                    raise BridgeUnsupported("select scatter source shape mismatch")
            else:
                start, step, count = attrs["start"], attrs["step"], source_shape[axis]
                if type(step) is not int or step <= 0: raise BridgeUnsupported("invalid scatter step")
                if count != len(range(start, attrs["end"], step)): raise BridgeUnsupported("scatter source count mismatch")
            if count == 0:
                emit(label + "_unchanged", domain, {"old": mem(base, indices)}, "old", out_array, indices, dtype)
                continue
            stop = start + (count - 1) * step + 1
            if start < 0 or stop > out["shape"][axis]: raise BridgeUnsupported("scatter region outside output")
            for suffix, begin, end in (("before", 0, start), ("after", stop, out["shape"][axis])):
                part = deepcopy(domain); part[axis]["extent"] = end - begin
                coords = list(indices); coords[axis] = f"({begin}+{indices[axis]})"
                emit(label + "_" + suffix, part, {"old": mem(base, coords)}, "old", out_array, coords, dtype)
            if step > 1 and count > 1:
                part = deepcopy(domain); part[axis]["extent"] = count - 1
                part.append({"var": "gap", "extent": step - 1})
                coords = list(indices); coords[axis] = f"({start}+{indices[axis]}*{step}+gap+1)"
                emit(label + "_gaps", part, {"old": mem(base, coords)}, "old", out_array, coords, dtype)
            part = deepcopy(domain); part[axis]["extent"] = count
            coords = list(indices); coords[axis] = f"({start}+{indices[axis]}*{step})"
            source_coords = [v for j, v in enumerate(indices) if j != axis] if op == "select_scatter" else indices
            emit(label + "_updated", part, {"new": mem(src, source_coords)}, "new", out_array, coords, dtype)
            continue
        if op == "softmax":
            if not finite_only: raise BridgeUnsupported("softmax requires explicit finite_only policy")
            if dtype not in FLOATS or not domain: raise BridgeUnsupported("softmax needs floating rank>=1 tensor")
            axis = attrs["dim"]
            if type(axis) is not int or not 0 <= axis < len(domain) or out["shape"][axis] == 0:
                raise BridgeUnsupported("softmax requires nonempty valid axis")
            source = names[node["args"][0]["ref"]]
            comp = "float64" if dtype == "float64" else "float32"
            row_domain = [d for j, d in enumerate(domain) if j != axis]
            row_indices = [v for j, v in enumerate(indices) if j != axis]
            row_shape = [d for j, d in enumerate(out["shape"]) if j != axis]
            maximum, exponentials, total = f"max{number}", f"exp{number}", f"sum{number}"
            temporary(maximum, row_shape, comp); temporary(total, row_shape, comp)
            temporary(exponentials, out["shape"], comp)
            first = list(indices); first[axis] = "0"
            emit(label + "_maxinit", row_domain, {"x": mem(source, first)}, "x", maximum, row_indices, comp)
            emit(label + "_max", domain, {"x": mem(source, indices)}, "x", maximum, row_indices, comp,
                 "lambda x, y: max(x, y)")
            emit(label + "_exp", domain, {"x": mem(source, indices), "m": mem(maximum, row_indices)},
                 f"exp({_cast('x', comp)} - m)", exponentials, indices, comp)
            emit(label + "_suminit", row_domain, {}, "0", total, row_indices, comp)
            emit(label + "_sum", domain, {"e": mem(exponentials, indices)}, "e", total, row_indices, comp,
                 "lambda x, y: x + y")
            emit(label + "_finish", domain, {"e": mem(exponentials, indices), "s": mem(total, row_indices)},
                 "e / s", out_array, indices, dtype)
            continue
        inputs, by_ref = {}, {}
        for j, read in enumerate(sem.get("reads", [])):
            if "guard" in read: raise BridgeUnsupported("guard outside supported functional scatter")
            identifier = read["value"]
            if table[identifier].get("kind") == "scalar": continue
            connector = f"input{j}"
            inputs[connector] = mem(names[identifier], read["indices"])
            by_ref.setdefault(identifier, connector)
        if op == "zeros_like":
            if inputs: raise BridgeUnsupported("zeros_like must have metadata-only dependencies")
            emit(label, domain, {}, "0", out_array, indices, dtype)
        elif op in POINTWISE:
            if dtype not in FLOATS: raise BridgeUnsupported("integer pointwise promotion/overflow unsupported")
            emit(label, domain, inputs, _expr(node, by_ref, table, finite_only), out_array, indices, dtype)
        elif op in LAYOUT:
            if len(inputs) != 1: raise BridgeUnsupported(f"{op} needs exactly one value source")
            emit(label, domain, inputs, next(iter(inputs)), out_array, indices, dtype)
        else:
            red = sem.get("reduction")
            if not red or red["operator"] not in ("sum", "sum-of-products", "max"):
                raise BridgeUnsupported("explicit reduction relation required")
            comp = "float32" if dtype in ("float16", "bfloat16") else dtype
            if comp not in ("float32", "float64"): raise BridgeUnsupported("integer reduction unsupported")
            accum = f"accumulator{number}"
            temporary(accum, out["shape"], comp)
            if op == "amax":
                if not finite_only or red["group_size"] == 0:
                    raise BridgeUnsupported("amax needs finite_only and nonempty group")
                read = sem["reads"][0]
                replacements = {v["var"]: "0" for v in red["variables"]}
                # Replace AST variable names, not substrings (r1 vs r10).
                class Substitute(ast.NodeTransformer):
                    def visit_Name(self, n):
                        return ast.copy_location(ast.Constant(0), n) if n.id in replacements else n
                first = [ast.unparse(Substitute().visit(ast.parse(str(x), mode="eval"))) for x in read["indices"]]
                emit(label + "_init", domain, {"x": mem(names[read["value"]], first)}, "x", accum, indices, comp)
            else: emit(label + "_init", domain, {}, "0", accum, indices, comp)
            if op in {"mm", "bmm", "matmul"}:
                if len(inputs) != 2: raise BridgeUnsupported("contraction requires two scalar reads")
                a, b = inputs
                expression = f"({_cast(a, comp)} * {_cast(b, comp)})"
            else:
                if len(inputs) != 1: raise BridgeUnsupported("reduction requires one scalar read")
                expression = next(iter(inputs))
            wcr = "lambda x, y: max(x, y)" if op == "amax" else "lambda x, y: x + y"
            emit(label, domain + red["variables"], inputs, expression, accum, indices, comp, wcr)
            final = "acc"
            if op == "mean":
                if red["group_size"] == 0: raise BridgeUnsupported("empty mean NaN emission unsupported")
                final = f"(acc / {_cast(str(red['group_size']), comp)})"
            emit(label + "_finish", domain, {"acc": mem(accum, indices)}, final, out_array, indices, dtype)
    if previous is None: state("identity")
    sdfg.validate()
    return sdfg, manifest


def _coordinates(shape):
    return itertools.product(*(range(x) for x in shape))


def _tasklet_operation_counts(tasklet):
    """Static tasklet arithmetic occurrences, not hardware FLOPs or numeric proof.

    Count calls/casts separately and exclude coordinate expressions in memlets.
    These bridge tasklets have no arithmetic control-flow branches except finite
    maximum's conditional selection, whose arms contain casts only.
    """
    counts = Counter()
    names = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div",
             ast.Pow: "pow", ast.FloorDiv: "floor_div", ast.Mod: "mod"}
    for node in ast.walk(ast.parse(tasklet.code.as_string)):
        if isinstance(node, ast.BinOp): counts[names.get(type(node.op), type(node.op).__name__)] += 1
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): counts["neg"] += 1
        elif isinstance(node, ast.Compare): counts["compare"] += len(node.ops)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "dace":
                counts["cast:" + node.func.attr] += 1
            elif isinstance(node.func, ast.Name): counts["call:" + node.func.id] += 1
            else: raise BridgeUnsupported("unexpected tasklet call in arithmetic counter")
    return counts


def _native_operation_counts(sdfg, manifest, mapped):
    totals, per_node, per_tasklet, reads = Counter(), {}, {}, Counter()
    for state in sdfg.states():
        scope = state.scope_dict()
        for tasklet in state.nodes():
            if not isinstance(tasklet, dace.nodes.Tasklet): continue
            multiplicity = 1
            parent = scope[tasklet]
            while parent is not None:
                if not mapped: raise BridgeUnsupported("unrolled arithmetic counter found map")
                multiplicity *= math.prod(int((b-a)//c+1) for a,b,c in parent.map.range.ranges)
                parent = scope[parent]
            counts = _tasklet_operation_counts(tasklet)
            for edge in state.out_edges(tasklet):
                if edge.data.wcr is not None:
                    if "+" in edge.data.wcr: counts["wcr:sum"] += 1
                    elif "max" in edge.data.wcr: counts["wcr:max"] += 1
                    else: raise BridgeUnsupported("unknown WCR in arithmetic counter")
            prefix = tasklet.label.split("_")[0]
            per_node.setdefault(prefix, Counter())
            # MapUnroll may suffix tasklet labels with numeric coordinates.
            label = tasklet.label
            while label.rsplit("_", 1)[-1].lstrip("-").isdigit(): label = label.rsplit("_", 1)[0]
            per_tasklet.setdefault(label, Counter())
            for kind, number in counts.items():
                value = number * multiplicity
                totals[kind] += value; per_node[prefix][kind] += value; per_tasklet[label][kind] += value
            for edge in state.in_edges(tasklet):
                value = manifest["array_to_value"].get(edge.data.data)
                if value is not None: reads[(prefix, value)] += multiplicity
    return {"totals": dict(totals), "per_node": {k:dict(v) for k,v in per_node.items()},
            "per_tasklet": {k:dict(v) for k,v in per_tasklet.items()},
            "tensor_read_occurrences": {k+":"+v:n for (k,v),n in reads.items()}}


def unroll_and_check(graph, sdfg, manifest, *, max_instances=20000, max_read_checks=200000):
    """Native MapUnroll + exact immediate semantic input cone verification.

    Tracks native memlet dependencies through internal reduction/softmax buffers.
    Guarded scatter reads must select exactly one old/source region. This verifies
    value/index dependency SETS, not arithmetic occurrence counts or FP trees.
    """
    table = validate_graph(graph)
    estimate = 0
    for st in sdfg.states():
        for n in st.nodes():
            if isinstance(n, dace.nodes.MapEntry):
                estimate += math.prod(int((b - a) // c + 1) for a, b, c in n.map.range.ranges)
    if estimate > max_instances: raise BridgeUnsupported(f"unroll budget exceeded: {estimate} > {max_instances}")
    mapped_operations = _native_operation_counts(sdfg, manifest, mapped=True)
    expanded = deepcopy(sdfg)
    applications = expanded.apply_transformations_repeated(MapUnroll, validate=True, validate_all=True)
    expanded.validate()
    unrolled_operations = _native_operation_counts(expanded, manifest, mapped=False)
    if mapped_operations != unrolled_operations:
        raise BridgeUnsupported("native unroll changed arithmetic or tensor-read occurrence counts")
    expected, checks = {}, 0
    for node in graph["nodes"]:
        sem = node["semantics"]
        variables = sem["domain"] + (sem.get("reduction") or {}).get("variables", [])
        count = math.prod(v["extent"] for v in variables) * max(1, len(sem.get("reads", [])))
        if checks + count > max_read_checks: raise BridgeUnsupported("semantic read-check budget exceeded")
        checks += count
        for out_coord in _coordinates(node["outputs"][0]["shape"]):
            expected[(node["outputs"][0]["id"], out_coord)] = set()
        for coord in _coordinates([v["extent"] for v in variables]):
            env = dict(zip((v["var"] for v in variables), coord))
            key = (node["outputs"][0]["id"], tuple(env[v["var"]] for v in sem["domain"]))
            selected = [read for read in sem.get("reads", []) if "guard" not in read or index_eval(read["guard"], env)]
            if node["op"] == "constant_pad_nd":
                literals = [r for r in sem.get("constant_regions", []) if index_eval(r["guard"], env)]
                if len(selected) + len(literals) != 1:
                    raise BridgeUnsupported("padding guards must partition into exactly one read or literal")
                if literals and literals[0].get("value") != node["attributes"].get("fill_value"):
                    raise BridgeUnsupported("padding literal disagrees with fill_value")
            for read in selected:
                if table[read["value"]].get("kind") == "scalar": continue
                idx = tuple(index_eval(x, env) for x in read["indices"])
                shape = table[read["value"]]["shape"]
                if any(type(i) is not int or i < 0 or i >= n for i, n in zip(idx, shape)):
                    raise BridgeUnsupported("semantic read index out of bounds")
                expected[key].add((read["value"], idx))
    array_values = manifest["array_to_value"]
    deps, actual = {}, {}
    for array, value in array_values.items():
        if value not in {m["id"] for m in graph["inputs"]}: continue
        for coord in _coordinates(table[value]["shape"] or [1]):
            logical = () if not table[value]["shape"] else coord
            deps[(array, coord)] = {(value, logical)}
    def edge_key(edge):
        ranges = edge.data.subset.ranges
        if any(a != b or c != 1 for a, b, c in ranges):
            raise BridgeUnsupported("unrolled tasklet has non-scalar memlet")
        return edge.data.data, tuple(int(a) for a, b, c in ranges)
    tasklets = 0
    for st in expanded.states():
        for tasklet in st.nodes():
            if isinstance(tasklet, dace.nodes.MapEntry): raise BridgeUnsupported("native unroll left a map")
            if not isinstance(tasklet, dace.nodes.Tasklet): continue
            tasklets += 1
            cone = set()
            for edge in st.in_edges(tasklet):
                key = edge_key(edge)
                if key not in deps: raise BridgeUnsupported(f"read before initialized value: {key}")
                cone |= deps[key]
            for edge in st.out_edges(tasklet):
                key = edge_key(edge)
                result = set(cone)
                if edge.data.wcr is not None:
                    if key not in deps: raise BridgeUnsupported("WCR without initialization")
                    result |= deps[key]
                if key[0] in array_values:
                    value = array_values[key[0]]
                    logical = () if not table[value]["shape"] else key[1]
                    output = (value, logical)
                    if output in actual: raise BridgeUnsupported("semantic scalar output written more than once")
                    actual[output] = result
                    deps[key] = {output}  # next semantic node sees the SSA boundary
                else:
                    deps[key] = result
    if actual != expected:
        bad = [k for k in expected if actual.get(k) != expected[k]]
        raise BridgeUnsupported(f"native semantic input-cone mismatch at {bad[:3]}; actual={len(actual)} expected={len(expected)}")
    return expanded, {"pass": True, "map_unroll_applications": applications,
                      "tasklets_unrolled": tasklets, "semantic_scalar_outputs": len(expected),
                      "semantic_read_checks": checks,
                      "arithmetic_occurrences": unrolled_operations,
                      "arithmetic_count_preservation": True,
                      "scope": "native immediate input cone sets and static tasklet arithmetic/read occurrence preservation; not FP order, hardware FLOPs, or native numeric execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finite-only", action="store_true")
    parser.add_argument("--unroll", action="store_true")
    parser.add_argument("--max-instances", type=int, default=20000)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    sdfg, manifest = lower_to_sdfg(graph, finite_only=args.finite_only)
    args.output.mkdir(parents=True, exist_ok=True)
    sdfg.save(str(args.output / "parameterized.sdfg"))
    report = {"validation": "pass", "native_numeric_execution": False, "manifest": manifest}
    code = sdfg.generate_code()
    for j, unit in enumerate(code):
        (args.output / f"generated_{j}.{unit.language}").write_text(unit.code, encoding="utf-8")
    report["codegen_units"] = len(code)
    if args.unroll:
        expanded, check = unroll_and_check(graph, sdfg, manifest, max_instances=args.max_instances)
        expanded.save(str(args.output / "unrolled.sdfg"))
        report["unroll_check"] = check
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
