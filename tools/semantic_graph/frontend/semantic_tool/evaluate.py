"""Independent, fail-closed interpreter for semantic-graph-v1.

The numeric engine is explicitly selected PyTorch CPU primitives, NOT the source
function and NOT a DaCe/GPU execution or proof of all-algorithm equivalence.
Args/kwargs are the executable parameter encoding; normalized attributes supply
parameters only when args/kwargs omit them. Shape/dtype metadata is checked at
all tensor boundaries. Logical meta-device literals are mapped explicitly to CPU.
No operator is selected from an arbitrary ATen target.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import prod
import torch

from .core import UnsupportedOperator


class EvaluationError(ValueError):
    """Invalid semantic graph, binding, primitive arguments, or metadata."""


_DTYPES = {
    name: getattr(torch, name) for name in (
        "bool", "uint8", "int8", "int16", "int32", "int64", "float16",
        "bfloat16", "float32", "float64", "complex64", "complex128"
    )
}
_OPS = frozenset({
    "add", "sub", "mul", "div", "pow", "tanh", "exp", "sigmoid", "sqrt",
    "rsqrt", "maximum", "cast", "sum", "mean", "amax", "softmax", "matmul",
    "mm", "bmm", "permute", "transpose", "reshape", "view", "expand",
    "select", "slice", "clone", "contiguous", "slice_scatter",
    "select_scatter", "copy", "zeros_like", "neg", "rsub", "alias", "unsqueeze", "squeeze",
})


def _dtype(value):
    if isinstance(value, torch.dtype):
        if value not in _DTYPES.values():
            raise EvaluationError(f"Unsupported dtype: {value}")
        return value
    if isinstance(value, dict) and set(value) == {"dtype"}:
        value = value["dtype"]
    if isinstance(value, str):
        value = value.removeprefix("torch.")
        if value in _DTYPES:
            return _DTYPES[value]
    raise EvaluationError(f"Invalid or unsupported dtype: {value!r}")


def _resolve(value, env, where):
    if isinstance(value, dict):
        if set(value)=={'float'} and value['float'] in ('inf','-inf','nan'):
            from .literals import decode_float
            return decode_float(value)
        if "ref" in value:
            if set(value) != {"ref"} or not isinstance(value["ref"], str):
                raise EvaluationError(f"{where}: malformed reference {value!r}")
            if value["ref"] not in env:
                raise EvaluationError(f"{where}: unknown/forward reference {value['ref']!r}")
            return env[value["ref"]]
        if set(value) == {"dtype"}:
            return _dtype(value)
        if set(value) == {"device"}:
            if value["device"] not in ("cpu", "meta"):
                raise EvaluationError(f"{where}: interpreter only executes on CPU")
            return torch.device("cpu")
        if set(value) == {"layout"}:
            if value["layout"] != "strided":
                raise EvaluationError(f"{where}: unsupported tensor layout")
            return torch.strided
        if set(value) == {"memory_format"} and not isinstance(value["memory_format"], (dict, type(None))):
            formats = {"preserve_format": torch.preserve_format,
                       "contiguous_format": torch.contiguous_format,
                       "channels_last": torch.channels_last,
                       "channels_last_3d": torch.channels_last_3d}
            if value["memory_format"] not in formats:
                raise EvaluationError(f"{where}: unknown memory format")
            return formats[value["memory_format"]]
        return {k: _resolve(v, env, where) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve(v, env, where) for v in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise EvaluationError(f"{where}: non-encoded value {type(value).__name__}")


def _resolve_output(value, env):
    """Resolve references without treating ordinary return dictionary keys as types."""
    if isinstance(value, dict):
        if set(value)=={'float'} and value['float'] in ('inf','-inf','nan'):
            from .literals import decode_float
            return decode_float(value)
        if set(value) == {"ref"} and isinstance(value["ref"], str):
            if value["ref"] not in env:
                raise EvaluationError(f"graph outputs: unknown reference {value['ref']!r}")
            return env[value["ref"]]
        return {k: _resolve_output(v, env) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve_output(v, env) for v in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise EvaluationError("graph outputs: invalid encoded return")


def _metadata(value, meta, where):
    if not isinstance(meta, dict):
        raise EvaluationError(f"{where}: metadata must be an object")
    kind = meta.get("kind", "tensor" if "shape" in meta else "scalar")
    if kind == "tensor":
        if not isinstance(value, torch.Tensor):
            raise EvaluationError(f"{where}: expected tensor, got {type(value).__name__}")
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise EvaluationError(f"{where}: only strided CPU tensors are supported")
        shape = meta.get("shape")
        if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise EvaluationError(f"{where}: missing/invalid concrete shape")
        if list(value.shape) != shape:
            raise EvaluationError(f"{where}: shape mismatch: got {list(value.shape)}, expected {shape}")
        if "dtype" not in meta or value.dtype != _dtype(meta["dtype"]):
            raise EvaluationError(f"{where}: dtype mismatch: got {value.dtype}, expected {meta.get('dtype')}")
    elif kind == "scalar":
        if type(value) not in (bool, int, float, str) and value is not None:
            raise EvaluationError(f"{where}: expected a Python scalar")
        if "specialized_value" in meta:
            from .literals import decode_float, equal_tagged
            expected = decode_float(meta["specialized_value"])
            if type(value) is not type(expected) or not equal_tagged(value, expected):
                raise EvaluationError(f"{where}: specialized scalar must equal {expected!r}")
        scalar_types = {"int": int, "int64": int, "float": float, "float64": float,
                        "bool": bool, "str": str, "NoneType": type(None)}
        if meta.get("dtype") in scalar_types and type(value) is not scalar_types[meta["dtype"]]:
            raise EvaluationError(f"{where}: scalar dtype mismatch")
    else:
        raise EvaluationError(f"{where}: unsupported value kind {kind!r}")


def _bind(args, kwargs, attrs, names, defaults=None, aliases=None):
    """Bind a closed primitive signature; never silently drop extra parameters."""
    if len(args) > len(names):
        raise EvaluationError(f"arity: expected at most {len(names)} arguments, got {len(args)}")
    if any(k not in names for k in kwargs):
        raise EvaluationError(f"unknown keyword(s): {sorted(set(kwargs) - set(names))}")
    values = dict(zip(names, args))
    for key, value in kwargs.items():
        if key in values:
            raise EvaluationError(f"duplicate argument: {key}")
        values[key] = value
    for key in names:
        if key not in values:
            attr_key = (aliases or {}).get(key, key)
            if key in attrs:
                values[key] = attrs[key]
            elif attr_key in attrs:
                values[key] = attrs[attr_key]
            elif key in (defaults or {}):
                values[key] = defaults[key]
            else:
                raise EvaluationError(f"arity: missing required argument {key!r}")
    return values


def _tensor(value, name="input"):
    if not isinstance(value, torch.Tensor):
        raise EvaluationError(f"{name} must be a tensor")
    return value


def _dim(value, rank):
    if type(value) is not int or not -max(rank, 1) <= value < max(rank, 1):
        raise EvaluationError(f"invalid dimension {value!r} for rank {rank}")
    return value % max(rank, 1)


def _dims(value, rank):
    if value is None:
        return None
    values = [value] if type(value) is int else value
    if not isinstance(values, (tuple, list)):
        raise EvaluationError("reduction dimensions must be int/list/None")
    normalized = [_dim(v, rank) for v in values]
    if len(set(normalized)) != len(normalized):
        raise EvaluationError("duplicate reduction dimension")
    return tuple(normalized)


def _shape(value):
    if not isinstance(value, (list, tuple)) or any(type(x) is not int for x in value):
        raise EvaluationError("shape must be an integer list")
    return tuple(value)


def _boolean(value, name):
    if type(value) is not bool:
        raise EvaluationError(f"{name} must be bool")
    return value


def _primitive(op, args, kwargs, attrs):
    # ATen uses self/mat2/size keyword names. Normalize only these documented
    # signature aliases, never arbitrary target names or callbacks.
    kwargs = dict(kwargs)
    aliases = {"self": "input", "mat2": "other"}
    if op in {"reshape", "view"}:
        aliases["size"] = "shape"
    for old, new in aliases.items():
        if old in kwargs:
            if new in kwargs:
                raise EvaluationError(f"duplicate argument {new}")
            kwargs[new] = kwargs.pop(old)
    def bind(names, defaults=None, aliases=None):
        return _bind(args, kwargs, attrs, names, defaults, aliases)

    if op in {"add", "sub", "rsub"}:
        p = bind(["input", "other", "alpha"], {"alpha": 1})
        if op == "rsub":
            return torch.rsub(p["input"], p["other"], alpha=p["alpha"])
        return (torch.add if op == "add" else torch.sub)(p["input"], p["other"], alpha=p["alpha"])
    if op in {"mul", "pow", "maximum"}:
        p = bind(["input", "other"])
        fn = {"mul": torch.mul, "pow": torch.pow, "maximum": torch.maximum}[op]
        return fn(p["input"], p["other"])
    if op == "div":
        p = bind(["input", "other", "rounding_mode"], {"rounding_mode": None})
        if p["rounding_mode"] not in (None, "trunc", "floor"):
            raise EvaluationError("invalid division rounding_mode")
        return torch.div(p["input"], p["other"], rounding_mode=p["rounding_mode"])
    if op in {"tanh", "exp", "sigmoid", "sqrt", "rsqrt", "neg"}:
        p = bind(["input"])
        fn = {"tanh": torch.tanh, "exp": torch.exp, "sigmoid": torch.sigmoid,
              "sqrt": torch.sqrt, "rsqrt": torch.rsqrt, "neg": torch.neg}[op]
        return fn(_tensor(p["input"]))
    if op == "cast":
        # _to_copy kwargs remain meaningful; raw to.dtype positional flags are
        # deliberately not guessed. Lowering supplies normalized dtype attrs.
        p = bind(["input", "dtype", "layout", "device", "pin_memory", "non_blocking", "memory_format"],
                 {"dtype": None, "layout": None, "device": None, "pin_memory": None,
                  "non_blocking": False, "memory_format": torch.preserve_format})
        if p["dtype"] is None:
            raise EvaluationError("cast requires an explicit dtype")
        if p["layout"] not in (None, torch.strided, "strided") or p["pin_memory"] not in (None, False):
            raise EvaluationError("cast layout/pinned allocation is unsupported")
        if p["device"] not in (None, "cpu", torch.device("cpu")):
            raise EvaluationError("cast only supports CPU")
        _boolean(p["non_blocking"], "non_blocking")
        return _tensor(p["input"]).to(dtype=_dtype(p["dtype"]), device="cpu", copy=True,
                                     non_blocking=p["non_blocking"], memory_format=p["memory_format"])
    if op in {"sum", "mean", "amax"}:
        names = ["input", "dim", "keepdim"] + ([] if op == "amax" else ["dtype"])
        p = bind(names, {"dim": None, "keepdim": False, "dtype": None}, {"dim": "axes"})
        x = _tensor(p["input"])
        dim = _dims(p["dim"], x.ndim)
        keepdim = _boolean(p["keepdim"], "keepdim")
        if op == "amax":
            return torch.amax(x, dim=dim, keepdim=keepdim)
        dtype = None if p["dtype"] is None else _dtype(p["dtype"])
        return (torch.sum if op == "sum" else torch.mean)(x, dim=dim, keepdim=keepdim, dtype=dtype)
    if op == "softmax":
        # aten._softmax's third bool is half_to_float; softmax.int's third
        # argument is optional dtype. Both have explicit, disjoint handling.
        if len(args) >= 3 and type(args[2]) is bool:
            p = bind(["input", "dim", "half_to_float"], {"half_to_float": False}, {"dim": "axis"})
            x = _tensor(p["input"])
            if p["half_to_float"] and x.dtype != torch.float16:
                raise EvaluationError("half_to_float requires float16 input")
            return torch.softmax(x, dim=_dim(p["dim"], x.ndim),
                                 dtype=torch.float32 if p["half_to_float"] else None)
        if "half_to_float" in kwargs:
            p = bind(["input", "dim", "half_to_float"], {"half_to_float": False}, {"dim": "axis"})
            x = _tensor(p["input"])
            flag = _boolean(p["half_to_float"], "half_to_float")
            if flag and x.dtype != torch.float16:
                raise EvaluationError("half_to_float requires float16 input")
            return torch.softmax(x, dim=_dim(p["dim"], x.ndim), dtype=torch.float32 if flag else None)
        p = bind(["input", "dim", "dtype"], {"dtype": None}, {"dim": "axis"})
        x = _tensor(p["input"])
        return torch.softmax(x, dim=_dim(p["dim"], x.ndim),
                             dtype=None if p["dtype"] is None else _dtype(p["dtype"]))
    if op in {"matmul", "mm", "bmm"}:
        p = bind(["input", "other"])
        return {"matmul": torch.matmul, "mm": torch.mm, "bmm": torch.bmm}[op](
            _tensor(p["input"]), _tensor(p["other"], "other"))
    if op == "permute":
        p = bind(["input", "dims"])
        x = _tensor(p["input"])
        dims = _shape(p["dims"])
        if len(dims) != x.ndim or len({_dim(d, x.ndim) for d in dims}) != x.ndim:
            raise EvaluationError("permute requires each dimension exactly once")
        return x.permute(dims)
    if op == "transpose":
        normalized_attrs = dict(attrs)
        if "dims" in attrs:
            dims = _shape(attrs["dims"])
            if len(dims) != 2:
                raise EvaluationError("transpose requires two dims")
            normalized_attrs.update(dim0=dims[0], dim1=dims[1])
        p = _bind(args, kwargs, normalized_attrs, ["input", "dim0", "dim1"])
        x = _tensor(p["input"])
        return x.transpose(_dim(p["dim0"], x.ndim), _dim(p["dim1"], x.ndim))
    if op in {"reshape", "view"}:
        p = bind(["input", "shape"])
        x, shape = _tensor(p["input"]), _shape(p["shape"])
        return x.view(shape) if op == "view" else torch.reshape(x, shape)
    if op == "expand":
        p = bind(["input", "size", "implicit"], {"implicit": False}, {"size": "shape"})
        _boolean(p["implicit"], "implicit")
        return _tensor(p["input"]).expand(_shape(p["size"]))
    if op == "alias":
        p = bind(["input"])
        return _tensor(p["input"])
    if op == "unsqueeze":
        p = bind(["input", "dim"])
        x = _tensor(p["input"])
        return torch.unsqueeze(x, _dim(p["dim"], x.ndim + 1))
    if op == "squeeze":
        p = bind(["input", "dim"], {"dim": None}, {"dim": "axes"})
        x = _tensor(p["input"])
        dims = _dims(p["dim"], x.ndim)
        return torch.squeeze(x) if dims is None else torch.squeeze(x, dims)
    if op == "select":
        p = bind(["input", "dim", "index"])
        x = _tensor(p["input"])
        if type(p["index"]) is not int:
            raise EvaluationError("select index must be integer")
        return torch.select(x, _dim(p["dim"], x.ndim), p["index"])
    if op in {"slice", "slice_scatter"}:
        names = ["input"] + (["src"] if op == "slice_scatter" else []) + ["dim", "start", "end", "step"]
        p = bind(names, {"dim": 0, "start": None, "end": None, "step": 1})
        x = _tensor(p["input"])
        dim = _dim(p["dim"], x.ndim)
        if x.ndim == 0:
            raise EvaluationError("cannot slice a scalar tensor")
        if type(p["step"]) is not int or p["step"] <= 0:
            raise EvaluationError("slice step must be a positive integer")
        if any(v is not None and type(v) is not int for v in (p["start"], p["end"])):
            raise EvaluationError("slice bounds must be integers or null")
        if op == "slice":
            index = [slice(None)] * x.ndim
            index[dim] = slice(p["start"], p["end"], p["step"])
            return x[tuple(index)]
        src = _tensor(p["src"], "src")
        if src.dtype != x.dtype:
            raise EvaluationError("slice_scatter requires matching source/base dtypes")
        return torch.slice_scatter(x, src, dim=dim, start=p["start"], end=p["end"], step=p["step"])
    if op == "select_scatter":
        p = bind(["input", "src", "dim", "index"])
        x, src = _tensor(p["input"]), _tensor(p["src"], "src")
        if type(p["index"]) is not int or src.dtype != x.dtype:
            raise EvaluationError("select_scatter requires integer index and matching dtypes")
        return torch.select_scatter(x, src, dim=_dim(p["dim"], x.ndim), index=p["index"])
    if op == "copy":
        # Functional copy: destination shape/dtype and source broadcasting;
        # never perform copy_ on the input or discard the destination binding.
        p = bind(["input", "src", "non_blocking"], {"non_blocking": False})
        x, src = _tensor(p["input"]), _tensor(p["src"], "src")
        _boolean(p["non_blocking"], "non_blocking")
        return torch.broadcast_to(src, x.shape).to(dtype=x.dtype, device="cpu").clone(memory_format=torch.preserve_format)
    if op in {"clone", "contiguous"}:
        default = torch.preserve_format if op == "clone" else torch.contiguous_format
        p = bind(["input", "memory_format"], {"memory_format": default})
        x = _tensor(p["input"])
        memory_format = p["memory_format"] if p["memory_format"] is not None else default
        return x.clone(memory_format=memory_format) if op == "clone" else x.contiguous(memory_format=memory_format)
    if op == "zeros_like":
        p = bind(["input", "dtype", "layout", "device", "pin_memory", "memory_format"],
                 {"dtype": None, "layout": None, "device": None, "pin_memory": False,
                  "memory_format": torch.preserve_format})
        if p["layout"] not in (None, torch.strided, "strided") or p["device"] not in (None, "cpu", torch.device("cpu")):
            raise EvaluationError("zeros_like only supports strided CPU tensors")
        if p["pin_memory"] not in (False, None):
            raise EvaluationError("zeros_like pinned allocation unsupported")
        return torch.zeros_like(_tensor(p["input"]), dtype=None if p["dtype"] is None else _dtype(p["dtype"]),
                                memory_format=p["memory_format"])
    raise UnsupportedOperator("<primitive>", op, "unsupported normalized semantic op")


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # inference-mode tensors have no version counter
        return None


def _snapshot(tensor):
    return (tensor, tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
            tensor.device, tensor.storage_offset(), _version(tensor), tensor.detach().clone())


def _check_unmutated(snapshots):
    for tensor, shape, strides, dtype, device, offset, version, original in snapshots:
        if (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device,
            tensor.storage_offset(), _version(tensor)) != (shape, strides, dtype, device, offset, version):
            raise EvaluationError("input mutation detected (metadata or tensor version)")
        # Exact bytes preserve NaN payloads and signed zeros. reshape(-1) also
        # permits scalar byte views; contiguous handles noncontiguous inputs.
        before = original.detach().contiguous().reshape(-1).view(torch.uint8)
        after = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(before, after):
            raise EvaluationError("input mutation detected (tensor data)")


def evaluate(graph, inputs):
    """Evaluate normalized semantic operations and recursively resolve returns.

    ``inputs`` maps input ID or human-readable name to a concrete CPU tensor or
    specialized Python scalar. Unknown operations, references, argument keys,
    duplicate IDs, mismatched metadata, and attempted input mutation fail closed.
    Return containers are the graph's encoded lists/dicts, not flattened tensors.
    """
    if not isinstance(graph, Mapping) or graph.get("format") != "semantic-graph-v1":
        raise EvaluationError("expected semantic-graph-v1")
    if not isinstance(inputs, Mapping):
        raise EvaluationError("inputs must be a mapping of input names or IDs")
    if not isinstance(graph.get("inputs"), list) or not isinstance(graph.get("nodes"), list):
        raise EvaluationError("graph inputs/nodes must be lists")
    if "outputs" not in graph:
        raise EvaluationError("graph must encode its return structure")
    # Lowered ATen graphs carry checkable rule annotations. A graph containing
    # them must not mix unchecked hand-authored nodes into the same evaluation.
    annotated = [n for n in graph["nodes"] if isinstance(n, dict) and "semantics" in n]
    if annotated:
        if len(annotated) != len(graph["nodes"]):
            raise EvaluationError("cannot mix annotated and unannotated semantic nodes")
        from .verify import verify_semantics
        verify_semantics(graph)
    env, used, snapshots = {}, set(), []

    def put(identifier, value):
        if not isinstance(identifier, str) or not identifier or identifier in env:
            raise EvaluationError(f"invalid/duplicate value ID {identifier!r}")
        env[identifier] = value

    for meta in graph["inputs"]:
        if not isinstance(meta, dict):
            raise EvaluationError("input metadata must be an object")
        identifier = meta.get("id")
        keys = [key for key in dict.fromkeys([identifier, meta.get("name")]) if key is not None and key in inputs]
        if not keys:
            raise EvaluationError(f"missing input {identifier!r} ({meta.get('name')!r})")
        if len(keys) > 1 and inputs[keys[0]] is not inputs[keys[1]]:
            raise EvaluationError(f"ambiguous binding for input {identifier!r}")
        value = inputs[keys[0]]
        _metadata(value, meta, f"input {identifier}")
        put(identifier, value)
        used.update(keys)
        if isinstance(value, torch.Tensor):
            snapshots.append(_snapshot(value))
    if set(inputs) - used:
        raise EvaluationError(f"unexpected input keys: {sorted(set(inputs) - used, key=str)}")
    try:
        constants = graph.get("constants", {})
        if not isinstance(constants, dict):
            raise EvaluationError("constants must be an object")
        for identifier, meta in constants.items():
            if not isinstance(meta, dict) or "values" not in meta:
                raise EvaluationError(f"constant {identifier!r} requires explicit tensor values")
            shape = _shape(meta.get("shape"))
            if any(n < 0 for n in shape):
                raise EvaluationError("constant dimensions must be nonnegative")
            value = torch.tensor(meta["values"], dtype=_dtype(meta.get("dtype")), device="cpu")
            if value.numel() != prod(shape):
                raise EvaluationError(f"constant {identifier!r} value count mismatch")
            value = value.reshape(shape)
            _metadata(value, dict(meta, kind="tensor"), f"constant {identifier}")
            put(identifier, value)
        node_ids = set()
        with torch.no_grad():
            for node in graph["nodes"]:
                if not isinstance(node, dict):
                    raise EvaluationError("node must be an object")
                identifier, op = node.get("id"), node.get("op")
                where = f"node {identifier!r} op {op!r}"
                if not isinstance(identifier, str) or identifier in node_ids:
                    raise EvaluationError(f"{where}: invalid/duplicate node ID")
                node_ids.add(identifier)
                extension_ops = {'cat', 'constant_pad_nd', 'var', 'gelu', 'addmm', 'convolution', 'repeat', 'roll', 'stack', 'floor_divide', 'bitwise_not', 'bitwise_and', 'bitwise_or', 'bitwise_xor'}
                if not isinstance(op, str) or op not in (_OPS | extension_ops):
                    raise UnsupportedOperator(identifier, node.get("target", op), f"unsupported normalized operator {op!r}")
                if not isinstance(node.get("args", []), list) or not isinstance(node.get("kwargs", {}), dict) or not isinstance(node.get("attributes", {}), dict):
                    raise EvaluationError(f"{where}: invalid args/kwargs/attributes containers")
                args = _resolve(node.get("args", []), env, where)
                kwargs = {k: _resolve(v, env, where) for k, v in node.get("kwargs", {}).items()}
                attrs = {k: _resolve(v, env, where) for k, v in node.get("attributes", {}).items()}
                if not isinstance(args, list) or not isinstance(kwargs, dict) or not isinstance(attrs, dict):
                    raise EvaluationError(f"{where}: args/kwargs/attributes have invalid container types")
                outputs = node.get("outputs")
                if not isinstance(outputs, list) or len(outputs) != 1:
                    raise EvaluationError(f"{where}: arity: supported operators require exactly one output")
                try:
                    # Normalized reshape can still carry a mandatory view
                    # obligation; do not repair an illegal view with a copy.
                    layout = node.get("semantics", {}).get("layout", {})
                    execution_op = "view" if op == "reshape" and layout.get("alias") == "view" else op
                    if op in {'var', 'gelu', 'addmm'}:
                        from .ops_reduction import execute_reduction_extension
                        result = execute_reduction_extension(node, args, kwargs, attrs)
                    elif op == 'convolution':
                        from .ops_convolution import execute_convolution_extension
                        result = execute_convolution_extension(node, args, kwargs, attrs)
                    elif op in {'repeat', 'roll', 'stack', 'floor_divide', 'bitwise_not', 'bitwise_and', 'bitwise_or', 'bitwise_xor'}:
                        from .ops_indexing import execute_indexing_extension
                        result = execute_indexing_extension(node, args, kwargs, attrs)
                    elif op == 'cat':
                        bound = _bind(args, kwargs, attrs, ['tensors','dim'], {'dim':0})
                        result = torch.cat(bound['tensors'], dim=bound['dim'])
                    elif op == 'constant_pad_nd':
                        bound = _bind(args, kwargs, attrs, ['self','pad','value'], {'value':0}, {'value':'fill_value'})
                        result = torch.nn.functional.pad(bound['self'], bound['pad'], mode='constant', value=bound['value'])
                    else:
                        result = _primitive(execution_op, args, kwargs, attrs)
                    if attrs.get("copy") is True and op in {"permute", "transpose", "reshape", "view", "select", "slice", "expand", "unsqueeze", "squeeze", "alias"}:
                        result = result.clone(memory_format=torch.contiguous_format)
                    if isinstance(result, torch.Tensor):
                        if "dtype" in attrs and attrs["dtype"] is not None and result.dtype != _dtype(attrs["dtype"]):
                            raise EvaluationError("attribute dtype disagrees with executed result")
                        if "shape" in attrs and list(result.shape) != list(_shape(attrs["shape"])):
                            raise EvaluationError("attribute shape disagrees with executed result")
                    _metadata(result, outputs[0], where)
                    put(outputs[0].get("id"), result)
                except EvaluationError as exc:
                    raise EvaluationError(f"{where}: {exc}") from exc
                except (TypeError, ValueError, RuntimeError, IndexError, OverflowError) as exc:
                    raise EvaluationError(f"{where}: primitive rejected arguments: {exc}") from exc
        return _resolve_output(graph["outputs"], env)
    finally:
        _check_unmutated(snapshots)
