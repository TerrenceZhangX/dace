"""ATen capture -> explicit, typed tensor semantics (no task-specific rewrites).

Index expressions are a deliberately small Python-integer expression language.
They describe logical coordinates, not an execution schedule or a reduction tree.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

try:
    from .core import UnsupportedOperator
except ImportError:  # The module is also useful while other package pieces are built.
    class UnsupportedOperator(ValueError):
        def __init__(self, node_id, target, reason="unsupported operation"):
            self.node_id, self.target, self.reason = node_id, target, reason
            super().__init__(f"{node_id}: {target}: {reason}")


_DTYPE_ALIASES = {
    "float": "float32", "double": "float64", "half": "float16",
    "long": "int64", "int": "int32", "short": "int16", "char": "int8",
    "byte": "uint8", "bool": "bool", "bfloat16": "bfloat16",
}
_POINTWISE = {"add", "sub", "rsub", "mul", "div", "pow", "tanh", "exp", "sigmoid",
              "sqrt", "rsqrt", "maximum", "minimum", "neg", "abs", "relu",
              "log", "log1p", "expm1", "sin", "cos", "erf", "reciprocal",
              "eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or",
              "logical_not", "where", "clamp", "clamp_min", "clamp_max"}
_UNARY = {"tanh", "exp", "sigmoid", "sqrt", "rsqrt", "neg", "abs", "relu",
          "log", "log1p", "expm1", "sin", "cos", "erf", "reciprocal", "logical_not"}
_SUPPORTED_OVERLOADS = {
    "add": {"Tensor", "Scalar", "default"}, "sub": {"Tensor", "Scalar", "default"},
    "mul": {"Tensor", "Scalar", "default"}, "div": {"Tensor", "Scalar", "Tensor_mode", "Scalar_mode", "default"},
    "pow": {"Tensor_Tensor", "Tensor_Scalar", "Scalar", "default"},
    "sum": {"default", "dim_IntList"}, "mean": {"default", "dim"},
    "amax": {"default"}, "_softmax": {"default"}, "softmax": {"int"},
    "transpose": {"int"}, "transpose_copy": {"int"},
    "select": {"int"}, "select_copy": {"int"}, "slice": {"Tensor"}, "slice_copy": {"Tensor"},
    "copy": {"default"}, "copy_": {"default"},
    "to": {"dtype", "device", "dtype_layout", "other"},
    "squeeze": {"default", "dim", "dims"}, "squeeze_copy": {"default", "dim", "dims"},
    "full": {"default"}, "zeros": {"default"}, "ones": {"default"},
    "arange": {"default", "start", "start_step"},
}


def _dtype(value):
    if isinstance(value, dict):
        value = value.get("dtype")
    if value is None:
        return None
    value = str(value).removeprefix("torch.")
    return _DTYPE_ALIASES.get(value, value)


def _prod(shape):
    return math.prod(shape)


def _shape(meta):
    if meta.get("kind") == "scalar":
        return []
    shape = meta.get("shape")
    if not isinstance(shape, (list, tuple)) or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError(f"concrete nonnegative shape required: {shape!r}")
    return list(shape)


def _refs(value):
    if isinstance(value, dict):
        if set(value) == {"ref"} and isinstance(value["ref"], str):
            yield value["ref"]
        else:
            for sub in value.values():
                yield from _refs(sub)
    elif isinstance(value, (list, tuple)):
        for sub in value:
            yield from _refs(sub)


def _dim(dim, rank, insertion=False):
    if type(dim) is not int:
        raise ValueError(f"axis must be an integer, got {dim!r}")
    n = rank + int(insertion)
    if not -n <= dim < n:
        raise ValueError(f"axis {dim} outside rank {rank}")
    return dim % n


def _broadcast_index(source, dest, variables=None):
    variables = variables or [f"i{j}" for j in range(len(dest))]
    if len(source) > len(dest):
        raise ValueError(f"cannot broadcast {source} to {dest}")
    offset = len(dest) - len(source)
    result = []
    for j, extent in enumerate(source):
        out_extent = dest[offset + j]
        if extent != 1 and extent != out_extent:
            raise ValueError(f"cannot broadcast {source} to {dest}")
        result.append("0" if extent == 1 else variables[offset + j])
    return result


def _broadcast_shape(shapes):
    rank = max(map(len, shapes), default=0)
    out = [1] * rank
    for shape in shapes:
        for j, extent in enumerate(reversed(shape)):
            slot = rank - j - 1
            if out[slot] == 1:
                out[slot] = extent
            elif extent != 1 and extent != out[slot]:
                raise ValueError(f"incompatible broadcast dimensions {out[slot]} and {extent}")
    return out


def _reshape_indices(source, dest):
    if _prod(source) != _prod(dest):
        raise ValueError(f"reshape changes element count: {source} -> {dest}")
    terms = [f"(i{j} * {_prod(dest[j+1:])})" for j in range(len(dest))]
    flat = "(" + " + ".join(terms) + ")" if terms else "0"
    # Zero-sized domains are never instantiated; avoid syntactically undefined maps.
    return ["0" if extent <= 1 else f"(({flat} // {max(1, _prod(source[j+1:]))}) % {extent})"
            for j, extent in enumerate(source)]


class _Lowerer:
    def __init__(self, capture):
        self.capture = capture
        self.values = {}
        self.nodes = []
        for item in capture.get("inputs", []):
            self._register(item)
        for key, item in capture.get("constants", {}).items():
            value = deepcopy(item)
            value.setdefault("id", key)
            self._register(value)

    def _register(self, meta):
        key = meta["id"]
        if key in self.values:
            raise ValueError(f"duplicate SSA value id {key!r}")
        if meta.get("kind", "tensor") == "tensor":
            _shape(meta)
            if _dtype(meta.get("dtype")) is None:
                raise ValueError(f"tensor {key} is missing its dtype")
        self.values[key] = meta

    def lower(self, node):
        try:
            result = self._lower(node)
        except UnsupportedOperator:
            raise
        except (ValueError, KeyError, TypeError, IndexError, NotImplementedError) as exc:
            raise UnsupportedOperator(node.get("id", "<missing>"), node.get("target", "<missing>"), str(exc)) from exc
        for output in result["outputs"]:
            self._register(output)
        self.nodes.append(result)

    def _lower(self, node):
        # Each extension is a target-driven rule with the same strict contract.
        from .ops_layout import lower_layout_extension
        from .ops_reduction import lower_reduction_extension
        from .ops_convolution import lower_convolution_extension
        from .ops_indexing import lower_indexing_extension
        for extension in (lower_layout_extension, lower_reduction_extension, lower_convolution_extension, lower_indexing_extension):
            result = extension(node, self.values)
            if result is not None:
                return result
        target = node["target"]
        match = re.fullmatch(r"(?:torch\.ops\.)?aten\.([A-Za-z_][A-Za-z_0-9]*)(?:\.([A-Za-z_][A-Za-z_0-9]*))?", target)
        if not match:
            raise ValueError("only explicitly supported ATen targets can be lowered")
        base, overload = match.group(1), match.group(2) or "default"
        allowed = _SUPPORTED_OVERLOADS.get(base, {"default", "Tensor", "Scalar"})
        if overload not in allowed:
            raise ValueError(f"overload {overload!r} is not supported (out/in-place variants are not silently functionalized)")
        if base.endswith("_"):
            raise ValueError("in-place mutation requires capture functionalization with alias-aware SSA updates")
        outputs = deepcopy(node.get("outputs", []))
        if len(outputs) != 1:
            raise ValueError("only single-tensor-output operators are currently supported")
        out = outputs[0]
        shape = _shape(out)
        args, kwargs = deepcopy(node.get("args", [])), deepcopy(node.get("kwargs", {}))
        referenced = list(_refs([args, kwargs]))
        for ref in referenced:
            if ref not in self.values:
                raise ValueError(f"reference {ref!r} is undefined or not in SSA order")
        dtype = _dtype(out.get("dtype"))
        attrs = {}
        sem = {
            "domain": [{"var": f"i{j}", "extent": n} for j, n in enumerate(shape)],
            "reads": [], "reduction": None,
            "arithmetic": {
                "input_dtypes": [self.values[r].get("dtype") if self.values[r].get("kind") == "scalar" else _dtype(self.values[r].get("dtype")) for r in referenced],
                "output_dtype": dtype, "rounding": "at-output",
                "reassociation": "not-authorized-by-default",
            },
            "order": "data-only", "notes": [],
        }
        op = base

        def get(position, name, default=None):
            return kwargs[name] if name in kwargs else args[position] if len(args) > position else default

        def literal(value):
            from .literals import decode_float
            value = decode_float(value)
            if isinstance(value, dict) and set(value) == {"ref"}:
                meta = self.values[value["ref"]]
                if meta.get("kind") != "scalar" or "specialized_value" not in meta:
                    raise ValueError("structural arguments require concrete specialized scalars")
                return meta["specialized_value"]
            return value

        def tensor(position=0, name="self"):
            value = get(position, name)
            if not isinstance(value, dict) or set(value) != {"ref"}:
                raise ValueError(f"{name} requires a tensor reference")
            meta = self.values[value["ref"]]
            if meta.get("kind", "tensor") != "tensor":
                raise ValueError(f"{name} requires a tensor, not a Python scalar")
            return value["ref"], _shape(meta), meta

        def read(ref, indices, role="pointwise", **extra):
            sem["reads"].append({"value": ref, "indices": indices, "role": role, **extra})

        def expect(expected):
            if shape != list(expected):
                raise ValueError(f"captured output shape {shape} disagrees with semantic shape {expected}")

        def layout(kind, ref=None, mapping=None, copy=False):
            sem["layout"] = {
                "kind": kind, "input": ref, "output_strides": out.get("strides"),
                "input_strides": self.values[ref].get("strides") if ref else None,
                "logical_index_map": mapping,
                "alias": "none" if copy else "may-alias" if kind == "reshape" else "view",
                "requires_materialization": False,
            }
            if ref:
                source_meta = self.values[ref]
                if _dtype(source_meta.get("dtype")) != dtype:
                    raise ValueError("layout-only operation cannot change element dtype")
                source_strides = source_meta.get("strides")
                source_offset = source_meta.get("storage_offset", 0)
                sem["layout"]["input_storage_offset"] = source_offset
                sem["layout"]["output_storage_offset"] = out.get("storage_offset", 0)
                if source_strides is not None:
                    if len(source_strides) != len(mapping):
                        raise ValueError("source stride rank differs from source shape")
                    terms = [f"(({coord}) * {stride})" for coord, stride in zip(mapping, source_strides)]
                    sem["layout"]["input_storage_index"] = " + ".join([str(source_offset)] + terms)
                if out.get("strides") is not None:
                    if len(out["strides"]) != len(shape):
                        raise ValueError("output stride rank differs from output shape")
                    sem["layout"]["output_storage_index"] = " + ".join(
                        [str(out.get("storage_offset", 0))] + [f"(i{j} * {stride})" for j, stride in enumerate(out["strides"])])
            sem["arithmetic"]["rounding"] = "none"
            sem["notes"].append("Layout/alias obligations are preserved; functional data dependencies do not demand a materialized intermediate.")

        if base in _POINTWISE:
            if base in _UNARY:
                tensor()
            data_shapes = [_shape(self.values[r]) for r in referenced]
            expect(_broadcast_shape(data_shapes))
            for ref in referenced:
                read(ref, _broadcast_index(_shape(self.values[ref]), shape))
            if base in {"add", "sub", "rsub"}:
                attrs["alpha"] = literal(get(2, "alpha", 1))
            elif base == "div":
                attrs["rounding_mode"] = literal(get(2, "rounding_mode"))
                if attrs["rounding_mode"] not in (None, "trunc", "floor"):
                    raise ValueError("invalid division rounding_mode")
            elif base == "clamp":
                attrs.update(min=literal(get(1, "min")), max=literal(get(2, "max")))
            elif base in {"clamp_min", "clamp_max"}:
                attrs["min" if base == "clamp_min" else "max"] = literal(get(1, "min" if base == "clamp_min" else "max"))
            sem["arithmetic"]["literal_operands"] = [v for v in args if not isinstance(v, (dict, list))]
            sem["arithmetic"]["promotion"] = "ATen-wrapped-scalar-and-tensor-result-type"
            sem["arithmetic"]["literal_conversion"] = "wrapped scalar converted to operator computation type; not pre-rounded to low-precision output dtype"
            sem["arithmetic"]["computation_dtype"] = "ATen-opmath-of-promoted-inputs"
        elif base in {"sum", "mean", "amax"}:
            ref, source, meta = tensor()
            rank = len(source)
            raw_axes = literal(get(1, "dim"))
            if raw_axes is None or raw_axes == [] or raw_axes == ():
                axes = list(range(rank))
            else:
                raw_axes = raw_axes if isinstance(raw_axes, (list, tuple)) else [raw_axes]
                if rank == 0:
                    if any(a not in (-1, 0) for a in raw_axes):
                        raise ValueError("invalid axis for scalar reduction")
                    axes = []
                else:
                    axes = [_dim(literal(a), rank) for a in raw_axes]
                if len(axes) != len(set(axes)):
                    raise ValueError("duplicate reduction axes")
                axes.sort()
            keepdim = bool(literal(get(2, "keepdim", False)))
            if rank == 0:
                keepdim = bool(keepdim)
            expected = [1 if j in axes else d for j, d in enumerate(source)] if keepdim else [d for j, d in enumerate(source) if j not in axes]
            expect(expected)
            variables = [{"var": f"r{k}", "extent": source[axis]} for k, axis in enumerate(axes)]
            read_indices, out_axis = [], 0
            for j in range(rank):
                if j in axes:
                    read_indices.append(f"r{axes.index(j)}")
                else:
                    read_indices.append(f"i{j if keepdim else out_axis}")
                    out_axis += 1
            read(ref, read_indices, "reduction")
            count = _prod([source[a] for a in axes])
            identity = 0 if base in {"sum", "mean"} else "-infinity" if dtype and dtype.startswith(("float", "bfloat")) else "dtype-min"
            sem["reduction"] = {"axes": axes, "variables": variables, "operator": "max" if base == "amax" else "sum",
                                "identity": identity, "tree": "unspecified", "keepdim": keepdim, "group_size": count}
            if base == "mean":
                sem["reduction"]["finalize"] = {"op": "divide", "divisor": count, "empty": "nan"}
            attrs.update(axes=axes, keepdim=keepdim, dtype=_dtype(get(3, "dtype", dtype)))
            sem["arithmetic"]["accumulation_dtype"] = "float32" if dtype in ("float16", "bfloat16") else dtype
            sem["arithmetic"]["rounding"] = "reduction-implementation-dependent-and-at-output"
            sem["notes"].append("Reduction group is exact; accumulation order and reduction tree are intentionally unspecified.")
        elif base in {"_softmax", "softmax"}:
            op = "softmax"
            ref, source, meta = tensor()
            rank = len(source)
            axis = literal(get(1, "dim"))
            axis = _dim(axis, rank) if rank else 0 if axis in (0, -1) else _dim(axis, rank)
            expect(source)
            variables = [{"var": "r0", "extent": source[axis]}] if rank else []
            read(ref, ["r0" if j == axis else f"i{j}" for j in range(rank)], "row")
            sem["reduction"] = {"axes": [axis] if rank else [], "variables": variables, "operator": "softmax",
                                "identity": None, "tree": "unspecified", "functional": True,
                                "group_size": source[axis] if rank else 1}
            attrs.update(dim=axis, half_to_float=bool(literal(get(2, "half_to_float", False))) if base == "_softmax" else False,
                         dtype=dtype)
            sem["arithmetic"]["accumulation_dtype"] = "float32" if dtype in ("float16", "bfloat16") else dtype
            sem["arithmetic"]["rounding"] = "functional-kernel-and-at-output"
            sem["notes"].append("Each output depends on every input in its normalization row, including itself. Stable implementation and primitive arithmetic tree are not fixed.")
        elif base in {"mm", "bmm", "matmul", "mv", "dot"}:
            ref_a, a, _ = tensor(0, "self")
            ref_b, b, _ = tensor(1, "mat2" if base in {"mm", "bmm"} else "vec" if base in {"mv", "dot"} else "other")
            if not a or not b:
                raise ValueError("matmul does not accept zero-dimensional operands")
            if base == "mm" and (len(a) != 2 or len(b) != 2):
                raise ValueError("mm requires two matrices")
            if base == "bmm" and (len(a) != 3 or len(b) != 3 or a[0] != b[0]):
                raise ValueError("bmm requires equal batch-size rank-3 tensors")
            if base == "mv" and (len(a) != 2 or len(b) != 1):
                raise ValueError("mv requires matrix and vector")
            if base == "dot" and (len(a) != 1 or len(b) != 1):
                raise ValueError("dot requires two vectors")
            av, bv = len(a) == 1, len(b) == 1
            if a[-1] != (b[0] if bv else b[-2]):
                raise ValueError("matmul contraction extents disagree")
            batch = _broadcast_shape([a[:-2] if not av else [], b[:-2] if not bv else []])
            expected = batch + ([] if av else [a[-2]]) + ([] if bv else [b[-1]])
            expect(expected)
            n_batch = len(batch)
            ia = [] if av else _broadcast_index(a[:-2], batch)
            ib = [] if bv else _broadcast_index(b[:-2], batch)
            if not av:
                ia.append(f"i{n_batch}")
            ia.append("r0")
            ib.append("r0")
            if not bv:
                ib.append(f"i{n_batch + int(not av)}")
            read(ref_a, ia, "reduction", operand=0)
            read(ref_b, ib, "reduction", operand=1)
            sem["reduction"] = {"axes": [len(a)-1], "variables": [{"var": "r0", "extent": a[-1]}],
                                "operator": "sum-of-products", "identity": 0, "tree": "unspecified",
                                "group_size": a[-1], "contraction_axes": {ref_a: len(a)-1, ref_b: 0 if bv else len(b)-2}}
            attrs.update(left_vector=av, right_vector=bv, batch_shape=batch, contraction_extent=a[-1])
            sem["arithmetic"]["rounding"] = "contraction-implementation-dependent-and-at-output"
            sem["arithmetic"]["accumulation_dtype"] = "float32" if dtype in ("float16", "bfloat16") else dtype
        elif base in {"transpose", "transpose_copy", "permute", "permute_copy", "t", "t_copy"}:
            ref, source, _ = tensor()
            rank = len(source)
            if base in {"t", "t_copy"}:
                if rank > 2:
                    raise ValueError("t requires rank <= 2")
                permutation = list(reversed(range(rank)))
                attrs["dims"] = [0, 1] if rank == 2 else []
                op = "permute"
                attrs["dims"] = permutation
            elif base.startswith("transpose"):
                dim0, dim1 = literal(get(1, "dim0")), literal(get(2, "dim1"))
                if rank == 0:
                    if dim0 not in (0, -1) or dim1 not in (0, -1):
                        raise ValueError("invalid scalar transpose dimensions")
                    permutation = []
                    dim0 = dim1 = 0
                else:
                    dim0, dim1 = _dim(dim0, rank), _dim(dim1, rank)
                    permutation = list(range(rank))
                    permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
                attrs["dims"] = [dim0, dim1]
                op = "transpose"
            else:
                permutation = [_dim(literal(d), rank) for d in get(1, "dims")]
                if sorted(permutation) != list(range(rank)):
                    raise ValueError("invalid permutation")
                attrs["dims"] = permutation
                op = "permute"
            expect([source[d] for d in permutation])
            indices = [f"i{permutation.index(d)}" for d in range(rank)]
            read(ref, indices)
            attrs["copy"] = base.endswith("_copy")
            layout("permutation", ref, indices, copy=attrs["copy"])
        elif base in {"view", "reshape", "_unsafe_view", "view_copy", "reshape_copy", "_reshape_alias", "_reshape_alias_copy", "flatten"}:
            op = "reshape"
            ref, source, _ = tensor()
            if base == "flatten":
                start = _dim(literal(get(1, "start_dim", 0)), len(source)) if source else 0
                end = _dim(literal(get(2, "end_dim", -1)), len(source)) if source else 0
                if start > end:
                    raise ValueError("flatten start_dim exceeds end_dim")
                expected = source[:start] + [_prod(source[start:end+1])] + source[end+1:]
                expect(expected)
            else:
                requested = [literal(d) for d in get(1, "size", get(1, "shape"))]
                if requested.count(-1) > 1 or any(type(d) is not int or d < -1 for d in requested):
                    raise ValueError("invalid reshape shape")
                if -1 in requested:
                    known = _prod([d for d in requested if d != -1])
                    if not known or _prod(source) % known:
                        raise ValueError("ambiguous or indivisible inferred reshape dimension")
                    requested[requested.index(-1)] = _prod(source) // known
                expect(requested)
            indices = _reshape_indices(source, shape)
            read(ref, indices)
            attrs.update(shape=shape, source_shape=source, copy=base.endswith("_copy"))
            layout("reshape", ref, indices, copy=attrs["copy"])
            if base in {"view", "_unsafe_view", "_reshape_alias"}:
                sem["layout"]["alias"] = "view"
        elif base in {"unsqueeze", "unsqueeze_copy", "squeeze", "squeeze_copy"}:
            ref, source, _ = tensor()
            if base.startswith("unsqueeze"):
                axis = _dim(literal(get(1, "dim")), len(source), insertion=True)
                expected = source[:axis] + [1] + source[axis:]
                indices = [f"i{j if j < axis else j+1}" for j in range(len(source))]
                attrs.update(dim=axis, shape=expected)
                op = "unsqueeze"
            else:
                raw = literal(get(1, "dim"))
                if raw is None:
                    axes = [j for j, extent in enumerate(source) if extent == 1]
                elif not source:
                    if raw not in (0, -1, [], [0], [-1]):
                        raise ValueError("invalid squeeze axis for scalar")
                    axes = []
                else:
                    raw = raw if isinstance(raw, list) else [raw]
                    normalized = [_dim(literal(d), len(source)) for d in raw]
                    if len(set(normalized)) != len(normalized):
                        raise ValueError("duplicate squeeze axes")
                    axes = [d for d in normalized if source[d] == 1]
                expected = [d for j, d in enumerate(source) if j not in axes]
                indices = ["0" if j in axes else f"i{j-sum(k<j for k in axes)}" for j in range(len(source))]
                attrs.update(axes=axes, shape=expected)
                op = "squeeze"
            expect(expected)
            read(ref, indices)
            attrs["copy"] = base.endswith("_copy")
            layout("view", ref, indices, copy=attrs["copy"])
        elif base in {"expand", "expand_copy", "broadcast_to"}:
            op = "expand"
            ref, source, _ = tensor()
            requested = [literal(d) for d in get(1, "size")]
            if len(requested) < len(source):
                raise ValueError("expand cannot remove dimensions")
            offset = len(requested) - len(source)
            for j, extent in enumerate(requested):
                if extent == -1:
                    if j < offset:
                        raise ValueError("expand -1 cannot specify a new leading dimension")
                    requested[j] = source[j-offset]
                elif type(extent) is not int or extent < 0:
                    raise ValueError("expand size must be nonnegative or -1")
            expect(requested)
            indices = _broadcast_index(source, shape)
            read(ref, indices)
            attrs.update(shape=shape, copy=base.endswith("_copy"))
            layout("broadcast-view", ref, indices, copy=attrs["copy"])
        elif base in {"clone", "contiguous", "detach", "detach_copy", "alias", "alias_copy", "lift_fresh_copy", "lift_fresh"}:
            ref, source, _ = tensor()
            expect(source)
            indices = [f"i{j}" for j in range(len(source))]
            read(ref, indices)
            op = "clone" if base not in {"detach", "alias", "lift_fresh"} else "alias"
            attrs["memory_format"] = get(1, "memory_format")
            layout("copy" if op == "clone" else "view", ref, indices, copy=op == "clone")
        elif base in {"_to_copy", "to", "type_as"}:
            ref, source, meta = tensor()
            expect(source)
            op = "cast"
            indices = [f"i{j}" for j in range(len(source))]
            read(ref, indices)
            attrs.update(dtype=dtype, source_dtype=_dtype(meta["dtype"]), copy=base == "_to_copy",
                         device=get(1, "device") if base == "_to_copy" else kwargs.get("device"))
            sem["arithmetic"]["rounding"] = "dtype-conversion"
            sem["arithmetic"]["conversion"] = {"from": _dtype(meta["dtype"]), "to": dtype,
                                                 "float_to_integer": "truncate-toward-zero", "float_narrowing": "round-to-nearest-ties-to-even",
                                                 "out_of_range": "backend-defined-not-a-mathematical-identity"}
            sem["layout"] = {"kind": "cast", "input": ref, "alias": "none" if base == "_to_copy" else "may-alias",
                             "output_strides": out.get("strides"), "requires_materialization": False}
        elif base == "copy":
            dest_ref, dest, meta = tensor()
            src_ref, source, _ = tensor(1, "src")
            expect(dest)
            read(src_ref, _broadcast_index(source, shape))
            attrs.update(dtype=dtype, non_blocking=bool(literal(get(2, "non_blocking", False))))
            sem["metadata_reads"] = [{"value": dest_ref, "fields": ["shape", "dtype", "strides"]}]
            sem["arithmetic"]["rounding"] = "dtype-conversion"
            sem["notes"].append("Functional copy overwrites the complete destination; old destination elements are not value dependencies. In-place copy_ must be functionalized first.")
        elif base in {"select", "select_copy", "slice", "slice_copy"}:
            ref, source, _ = tensor()
            axis = _dim(literal(get(1, "dim", 0)), len(source))
            if base.startswith("select"):
                index = literal(get(2, "index"))
                if type(index) is not int or not -source[axis] <= index < source[axis]:
                    raise ValueError("select index outside source extent")
                index %= source[axis]
                expect(source[:axis] + source[axis+1:])
                indices = [str(index) if j == axis else f"i{j if j < axis else j-1}" for j in range(len(source))]
                attrs.update(dim=axis, index=index)
                op = "select"
            else:
                start, end, step = literal(get(2, "start")), literal(get(3, "end")), literal(get(4, "step", 1))
                if type(step) is not int or step <= 0:
                    raise ValueError("ATen slicing requires a positive integer step")
                start, end, step = slice(start, end, step).indices(source[axis])
                expected = list(source)
                expected[axis] = len(range(start, end, step))
                expect(expected)
                indices = [f"({start} + i{j} * {step})" if j == axis else f"i{j}" for j in range(len(source))]
                attrs.update(dim=axis, start=start, end=end, step=step)
                op = "slice"
            read(ref, indices)
            attrs["copy"] = base.endswith("_copy")
            layout("view", ref, indices, copy=attrs["copy"])
        elif base in {"slice_scatter", "select_scatter"}:
            ref, dest, _ = tensor()
            src_ref, source, _ = tensor(1, "src")
            axis = _dim(literal(get(2, "dim", 0)), len(dest))
            expect(dest)
            if base == "select_scatter":
                index = literal(get(3, "index"))
                if type(index) is not int or not -dest[axis] <= index < dest[axis]:
                    raise ValueError("select_scatter index outside destination extent")
                index %= dest[axis]
                if source != dest[:axis] + dest[axis+1:]:
                    raise ValueError("select_scatter source shape mismatch")
                guard = f"i{axis} == {index}"
                src_indices = [f"i{j}" for j in range(len(dest)) if j != axis]
                attrs.update(dim=axis, index=index)
            else:
                start, end, step = literal(get(3, "start")), literal(get(4, "end")), literal(get(5, "step", 1))
                if type(step) is not int or step <= 0:
                    raise ValueError("slice_scatter requires positive integer step")
                start, end, step = slice(start, end, step).indices(dest[axis])
                expected = list(dest)
                expected[axis] = len(range(start, end, step))
                if source != expected:
                    raise ValueError(f"slice_scatter source shape {source} != {expected}")
                guard = f"(i{axis} >= {start} and i{axis} < {end} and (i{axis} - {start}) % {step} == 0)"
                src_indices = [f"((i{j} - {start}) // {step})" if j == axis else f"i{j}" for j in range(len(dest))]
                attrs.update(dim=axis, start=start, end=end, step=step)
            read(ref, [f"i{j}" for j in range(len(dest))], guard=f"not ({guard})")
            read(src_ref, src_indices, guard=guard)
            sem["arithmetic"]["rounding"] = "none"
            sem["update"] = {"kind": "functional-SSA-scatter", "base": ref, "source": src_ref, "source_guard": guard}
            sem["notes"].append("Exactly one guarded read is active at each output coordinate; untouched elements retain the old SSA value.")
        elif base in {"zeros_like", "ones_like", "full_like", "zeros", "ones", "full", "new_zeros", "new_ones", "new_full"}:
            like = base.endswith("_like")
            new = base.startswith("new_")
            if like or new:
                ref, source, _ = tensor()
                sem["metadata_reads"] = [{"value": ref, "fields": ["shape", "dtype", "device", "strides"]}]
                expected = source if like else [literal(v) for v in get(1, "size")]
            else:
                expected = [literal(v) for v in get(0, "size")]
            expect(expected)
            fill = 0 if "zeros" in base else 1 if "ones" in base else literal(get(1 if not new else 2, "fill_value"))
            if not isinstance(fill, (int, float, bool, complex)):
                raise ValueError("full requires a literal or specialized scalar fill value")
            op = base if like else "full"
            attrs.update(shape=shape, dtype=dtype, fill_value=fill)
            sem["arithmetic"]["rounding"] = "dtype-conversion"
        elif base in {"empty", "empty_like", "new_empty", "empty_strided", "new_empty_strided"}:
            raise ValueError("uninitialized allocations have unknown values; functional capture must eliminate them or supply an explicit initialized SSA update")
        elif base == "arange":
            if overload == "default":
                start, end, step = 0, literal(get(0, "end")), 1
            else:
                start, end, step = literal(get(0, "start")), literal(get(1, "end")), literal(get(2, "step", 1))
            if not all(isinstance(v, (float, int)) for v in (start, end, step)) or step == 0:
                raise ValueError("arange requires finite concrete numeric bounds and nonzero step")
            if not all(math.isfinite(v) for v in (start, end, step)):
                raise ValueError("arange requires finite bounds")
            expect([max(0, math.ceil((end-start)/step))])
            attrs.update(start=start, end=end, step=step, dtype=dtype)
            sem["expression"] = f"({start!r} + i0 * {step!r})"
        else:
            raise ValueError("operator has no implemented semantic rule")

        # Casting and comparisons may change dtype. All metadata remains available
        # so consumers never erase dtype effects or storage/view obligations.
        result = deepcopy(node)
        result.update(op=op, args=args, kwargs=kwargs, outputs=outputs, attributes=attrs, semantics=sem)
        return result


def lower_capture(capture_graph):
    """Lower an acyclic ``aten-capture-v1`` dictionary, or fail explicitly.

    The result is an SSA functional *tensor* graph. Reads are exact integer index
    maps over output domains and optional reduction variables. Reductions retain
    their whole group and do not pick a fixed arithmetic tree. No raw target is
    admitted as an opaque node. Raw mutation is deliberately rejected: capture
    must first perform alias-aware functionalization, producing copy/scatter SSA.
    """
    if not isinstance(capture_graph, dict) or capture_graph.get("format") != "aten-capture-v1":
        raise ValueError("expected aten-capture-v1 graph")
    from .normalize import normalize_capture
    capture_graph = normalize_capture(capture_graph)
    lowerer = _Lowerer(capture_graph)
    for node in capture_graph.get("nodes", []):
        lowerer.lower(node)
    for ref in _refs(capture_graph.get("outputs")):
        if ref not in lowerer.values:
            raise ValueError(f"graph output references undefined value {ref!r}")
    result = deepcopy(capture_graph)
    result["format"] = "semantic-graph-v1"
    result["nodes"] = lowerer.nodes
    result.setdefault("metadata", {})["semantic_lowering"] = {
        "version": 1, "target_driven": True, "index_language": "python-integer-expressions-v1",
        "schedule": "none", "reduction_tree": "unspecified", "mutation": "functionalized-SSA-only",
        "rounding": "typed-ATen-operator-boundaries; reduction/kernel internals not uniquely specified",
    }
    from .literals import encode_floats
    return encode_floats(result)
