"""Closed ATen variance, GELU and addmm extensions for semantic-graph-v1.

The lowerer describes functional groups, not an execution schedule.  The CPU
helper is an explicitly selected primitive oracle, never a source-task replay.
Unknown targets return None; malformed supported calls raise ValueError.
"""
from __future__ import annotations

from copy import deepcopy
import math


_TARGETS = {
    "aten.var.correction": ("var", "correction"),
    "aten.var.dim": ("var", "dim"),
    "aten.var.default": ("var", "default"),
    "aten.gelu.default": ("gelu", "default"),
    "aten.addmm.default": ("addmm", "default"),
}
_FLOATS = {"float16", "bfloat16", "float32", "float64"}
_COMPLEX_REAL = {"complex64": "float32", "complex128": "float64"}
_NUMERIC = _FLOATS | set(_COMPLEX_REAL) | {"int8", "uint8", "int16", "int32", "int64"}
_OPMATH = "ATen-opmath-of-promoted-inputs"


def _selection(node):
    target = node.get("target")
    if isinstance(target, str):
        return _TARGETS.get(target.removeprefix("torch.ops."))
    # The interpreter may already have stripped the raw target.  Dispatch still
    # uses only these three explicit semantic primitive names.
    op = node.get("op")
    if op in {"gelu", "addmm"}:
        return op, "default"
    if op == "var":
        overload = node.get("attributes", {}).get("overload", "correction")
        if overload in {"correction", "dim", "default"}:
            return op, overload
    return None


def _bind(op, overload, args, kwargs, attrs=None):
    """Bind exact ATen signatures, rejecting extra/duplicate/positional kwargs."""
    attrs = attrs or {}
    if op == "var" and overload == "default":
        names, positional = ["self", "unbiased"], 2
        defaults = {"unbiased": True}
    elif op == "var" and overload == "dim":
        names, positional = ["self", "dim", "unbiased", "keepdim"], 4
        defaults = {"unbiased": True, "keepdim": False}
    elif op == "var":
        names, positional = ["self", "dim", "correction", "keepdim"], 2
        defaults = {"dim": None, "correction": None, "keepdim": False}
    elif op == "gelu":
        names, positional = ["self", "approximate"], 1
        defaults = {"approximate": "none"}
    else:
        names, positional = ["self", "mat1", "mat2", "beta", "alpha"], 3
        defaults = {"beta": 1, "alpha": 1}
    if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict):
        raise ValueError("args must be a sequence and kwargs must be an object")
    if len(args) > positional:
        raise ValueError(f"{op}.{overload} accepts at most {positional} positional arguments")
    if set(kwargs) - set(names):
        raise ValueError(f"unknown keyword(s): {sorted(set(kwargs) - set(names))}")
    result = dict(zip(names, args))
    for name, value in kwargs.items():
        if name in result:
            raise ValueError(f"duplicate argument: {name}")
        result[name] = value
    for name in names:
        if name not in result:
            attr_name = "axes" if name == "dim" else name
            if name in attrs:
                result[name] = attrs[name]
            elif attr_name in attrs:
                result[name] = attrs[attr_name]
            elif name in defaults:
                result[name] = defaults[name]
            else:
                raise ValueError(f"missing required argument: {name}")
    return result


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _number(value, name, finite=False):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a concrete real scalar")
    if finite and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _axes(raw, rank):
    if raw is None or raw == [] or raw == ():
        return list(range(rank))
    raw = [raw] if type(raw) is int else raw
    if not isinstance(raw, (list, tuple)):
        raise ValueError("reduction dim must be int, a list of ints, or None")
    normalized = []
    for dim in raw:
        if type(dim) is not int or not -max(rank, 1) <= dim < max(rank, 1):
            raise ValueError(f"invalid reduction axis {dim!r} for rank {rank}")
        axis = dim % max(rank, 1)
        if axis in normalized:
            raise ValueError("duplicate reduction axes")
        normalized.append(axis)
    return sorted(normalized) if rank else []


def lower_reduction_extension(node, values) -> dict | None:
    """Lower one single-output captured node using the caller's SSA value table.

    Call before the core overload allowlist.  This function does not register
    outputs or mutate node/values.  A recognized but invalid call fails closed.
    """
    target = node.get("target")
    selected = _TARGETS.get(target.removeprefix("torch.ops.")) if isinstance(target, str) else None
    if selected is None:
        return None
    # Lazy import avoids a cycle when semantics imports this extension.
    from .semantics import _broadcast_index, _dtype, _refs, _shape

    op, overload = selected
    args = deepcopy(node.get("args", []))
    kwargs = deepcopy(node.get("kwargs", {}))
    outputs = deepcopy(node.get("outputs", []))
    if len(outputs) != 1 or outputs[0].get("kind", "tensor") != "tensor":
        raise ValueError("reduction extension requires one tensor output")
    out = outputs[0]
    shape, dtype = _shape(out), _dtype(out)
    referenced = list(_refs([args, kwargs]))
    for ref in referenced:
        if ref not in values:
            raise ValueError(f"reference {ref!r} is undefined or not in SSA order")
    params = _bind(op, overload, args, kwargs)
    attrs = {"overload": overload}
    sem = {
        "domain": [{"var": f"i{j}", "extent": extent} for j, extent in enumerate(shape)],
        "reads": [], "reduction": None,
        "arithmetic": {
            "input_dtypes": [values[r].get("dtype") if values[r].get("kind") == "scalar" else _dtype(values[r]) for r in referenced],
            "output_dtype": dtype, "computation_dtype": _OPMATH,
            "promotion": "ATen-operator-input-dtype-rules",
            "rounding": "at-output", "reassociation": "not-authorized-by-default",
        },
        "order": "data-only", "notes": [],
    }

    def literal(value):
        if isinstance(value, dict) and set(value) == {"ref"}:
            meta = values[value["ref"]]
            if meta.get("kind") != "scalar" or "specialized_value" not in meta:
                raise ValueError("structural arguments require concrete specialized scalars")
            sem.setdefault("specializations", []).append({"value": value["ref"], "equals": meta["specialized_value"]})
            return meta["specialized_value"]
        if isinstance(value, (list, tuple)):
            return [literal(item) for item in value]
        return value

    def tensor(name):
        value = params[name]
        if not isinstance(value, dict) or set(value) != {"ref"}:
            raise ValueError(f"{name} requires a tensor reference")
        ref = value["ref"]
        meta = values[ref]
        if meta.get("kind", "tensor") != "tensor":
            raise ValueError(f"{name} requires a tensor, not a scalar")
        return ref, _shape(meta), _dtype(meta)

    def expect(expected, expected_dtype):
        if shape != expected:
            raise ValueError(f"captured output shape {shape} disagrees with semantic shape {expected}")
        if dtype != expected_dtype:
            raise ValueError(f"captured output dtype {dtype} disagrees with semantic dtype {expected_dtype}")

    if op == "var":
        ref, source, input_dtype = tensor("self")
        if input_dtype not in _FLOATS | set(_COMPLEX_REAL):
            raise ValueError("var requires floating or supported complex input dtype")
        axes = _axes(literal(params.get("dim")), len(source))
        keepdim = _boolean(literal(params.get("keepdim", False)), "keepdim")
        if overload in {"default", "dim"}:
            unbiased = _boolean(literal(params["unbiased"]), "unbiased")
            correction = int(unbiased)
            attrs["unbiased"] = unbiased
        else:
            correction = literal(params["correction"])
            correction = 1 if correction is None else _number(correction, "correction", finite=True)
        expected = ([1 if j in axes else n for j, n in enumerate(source)] if keepdim
                    else [n for j, n in enumerate(source) if j not in axes])
        expect(expected, _COMPLEX_REAL.get(input_dtype, input_dtype))
        indices, out_axis = [], 0
        for axis in range(len(source)):
            if axis in axes:
                indices.append(f"r{axes.index(axis)}")
            else:
                indices.append(f"i{axis if keepdim else out_axis}")
                out_axis += 1
        sem["reads"] = [{"value": ref, "indices": indices, "role": "reduction"}]
        count = math.prod(source[a] for a in axes)
        divisor = max(0, count - correction)
        stats = {
            "count": count, "correction": correction, "denominator": divisor,
            "definition": "sum(abs(x - mean(group)) ** 2) / max(0, N - correction)",
            "complex": "squared-complex-magnitude; real-valued-output",
            "empty_group": "nan",
            "nonpositive_degrees_of_freedom": {
                "condition": "N <= correction", "zero_centered_sum": "nan (0/0)",
                "positive_centered_sum": "+infinity (positive/0)",
                "nan_centered_sum": "nan",
            },
            "finite_output_guaranteed": False,
        }
        if input_dtype in _COMPLEX_REAL:
            stats["definition"] = "var(real(group), correction) + var(imag(group), correction)"
            stats["nonpositive_degrees_of_freedom"].update(
                positive_centered_sum="+infinity only when both real and imaginary centered sums are positive; otherwise NaN",
                complex_component_rule="apply real zero-divisor rule separately, then add; inf+nan is nan")
        sem["reduction"] = {
            "axes": axes,
            "variables": [{"var": f"r{j}", "extent": source[a]} for j, a in enumerate(axes)],
            "operator": "variance", "identity": None, "tree": "unspecified",
            "functional": True, "keepdim": keepdim, "group_size": count,
            "correction": correction, "statistics": stats,
            "finalize": {"op": "divide-centered-squared-magnitude", "divisor": divisor,
                         "empty": "nan", "zero_divisor": "IEEE-NaN-or-positive-infinity"},
        }
        attrs.update(axes=axes, keepdim=keepdim, correction=correction)
        sem["arithmetic"].update(accumulation_dtype=_OPMATH,
                                 rounding="functional-reduction-implementation-dependent-and-at-output")
        sem["notes"].append("The exact statistics group is specified, not Welford/two-pass or any reduction tree. Empty/singleton and nonpositive-DOF results are not assumed finite.")
    elif op == "gelu":
        ref, source, input_dtype = tensor("self")
        if input_dtype not in _FLOATS:
            raise ValueError("gelu requires real floating input dtype")
        expect(source, input_dtype)
        approximate = literal(params["approximate"])
        if approximate not in ("none", "tanh"):
            raise ValueError("gelu approximate must be 'none' or 'tanh'")
        attrs["approximate"] = approximate
        sem["reads"] = [{"value": ref, "indices": [f"i{j}" for j in range(len(source))], "role": "pointwise"}]
        sem["pointwise"] = {
            "operator": "gelu", "approximate": approximate,
            "definition": ("0.5*x*(1+erf(x/sqrt(2)))" if approximate == "none" else
                           "0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))"),
        }
        sem["arithmetic"]["rounding"] = "functional-pointwise-kernel-and-at-output"
        sem["notes"].append("Only the same-index input is a value dependency; approximate is preserved and constants are converted under ATen opmath, not pre-rounded to output dtype.")
    else:
        bias_ref, bias, bias_dtype = tensor("self")
        left_ref, left, left_dtype = tensor("mat1")
        right_ref, right, right_dtype = tensor("mat2")
        if len(left) != 2 or len(right) != 2 or left[1] != right[0]:
            raise ValueError("addmm requires rank-2 matrices with equal contraction extent")
        if left_dtype not in _NUMERIC or left_dtype != right_dtype or left_dtype != bias_dtype:
            raise ValueError("addmm requires matching supported numeric tensor dtypes")
        expect([left[0], right[1]], left_dtype)
        bias_indices = _broadcast_index(bias, shape)
        alpha = _number(literal(params["alpha"]), "alpha")
        beta = _number(literal(params["beta"]), "beta")
        if left_dtype not in _FLOATS | set(_COMPLEX_REAL) and (type(alpha) is not int or type(beta) is not int):
            raise ValueError("integer addmm requires integer alpha and beta")
        count = left[1]
        attrs.update(alpha=alpha, beta=beta, contraction_extent=count,
                     left_vector=False, right_vector=False, batch_shape=[])
        if count:
            sem["reads"].extend([
                {"value": left_ref, "indices": ["i0", "r0"], "role": "reduction", "operand": 0},
                {"value": right_ref, "indices": ["r0", "i1"], "role": "reduction", "operand": 1},
            ])
            sem["reduction"] = {
                "axes": [1], "variables": [{"var": "r0", "extent": count}],
                "operator": "sum-of-products", "identity": 0, "tree": "unspecified",
                "functional": True, "group_size": count,
                "contraction_axes": {left_ref: 1, right_ref: 0},
                "finalize": {"op": "scaled-contraction-plus-broadcast-bias", "alpha": alpha,
                             "beta": beta, "bias_ignored": beta == 0},
            }
        else:
            # No reduction assignments exist at K=0. Keeping bias outside a
            # nonexistent loop also preserves its cone in the basic scalarizer.
            sem["empty_contraction"] = {"identity": 0, "extent": 0, "alpha_ignored": True}
            sem.setdefault("metadata_reads", []).extend([
                {"value": left_ref, "fields": ["shape", "dtype"]},
                {"value": right_ref, "fields": ["shape", "dtype"]},
            ])
        if beta != 0:
            sem["reads"].append({"value": bias_ref, "indices": bias_indices, "role": "pointwise"})
        else:
            sem.setdefault("metadata_reads", []).append({"value": bias_ref, "fields": ["shape", "dtype"]})
        sem["pointwise"] = {"operator": "scaled-contraction-plus-broadcast-bias",
                            "alpha": alpha, "beta": beta, "bias_ignored": beta == 0,
                            "definition": "alpha*contraction + beta*bias; beta=0 omits bias entirely; K=0 uses contraction contribution zero without multiplying alpha"}
        sem["arithmetic"].update(accumulation_dtype=_OPMATH,
                                 literal_operands={"alpha": alpha, "beta": beta},
                                 literal_conversion="ATen scalar conversion to operation computation type, not output dtype",
                                 rounding="contraction-and-pointwise-implementation-dependent-and-at-output")
        sem["notes"].append("Functional contraction plus pointwise broadcast bias, not a chosen GEMM or reduction tree. beta=0 ignores bias values including NaN/Inf; alpha=0 does not authorize erasing matrix reads for all nonfinite inputs.")
    result = deepcopy(node)
    result.update(op=op, args=args, kwargs=kwargs, outputs=outputs, attributes=attrs, semantics=sem)
    return result


def execute_reduction_extension(node, args, kwargs, attrs):
    """Execute supported semantic primitives on CPU, returning None if unknown.

    Args/kwargs must already be resolved. Raw targets are matched against a
    closed table; no dynamic target lookup, source-task import, or GPU fallback.
    attrs supply only parameters omitted from args/kwargs.
    """
    selected = _selection(node)
    if selected is None:
        return None
    import torch

    op, overload = selected
    params = _bind(op, overload, args, kwargs, attrs)

    def tensor(name):
        value = params[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError(f"{name} requires a strided CPU tensor")
        return value

    if op == "gelu":
        approximate = params["approximate"]
        if approximate not in ("none", "tanh"):
            raise ValueError("gelu approximate must be 'none' or 'tanh'")
        return torch.nn.functional.gelu(tensor("self"), approximate=approximate)
    if op == "var":
        x = tensor("self")
        axes = _axes(params.get("dim"), x.ndim)
        keepdim = _boolean(params.get("keepdim", False), "keepdim")
        if overload in {"default", "dim"}:
            correction = int(_boolean(params["unbiased"], "unbiased"))
        else:
            correction = params["correction"]
            correction = 1 if correction is None else _number(correction, "correction", finite=True)
        return torch.var(x, dim=tuple(axes), correction=correction, keepdim=keepdim)
    alpha, beta = _number(params["alpha"], "alpha"), _number(params["beta"], "beta")
    return torch.addmm(tensor("self"), tensor("mat1"), tensor("mat2"), alpha=alpha, beta=beta)
