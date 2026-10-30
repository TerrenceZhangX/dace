"""Exact logical maps for explicitly whitelisted, forward real convolutions.

This is a sum-of-products relation, not an accumulation tree or a bitwise IEEE
algorithm. Padding contributes literal zero without reading an out-of-bounds
input (or an irrelevant weight). That simplification has a finite-real contract:
NaN/Inf inputs and nonfinite results are not supported; signed zero is not a
separate observable. Runtime evaluation uses only the three named CPU primitives.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .core import UnsupportedOperator


CONVOLUTION_TARGETS = frozenset({
    "aten.convolution.default", "torch.ops.aten.convolution.default",
})
CONVOLUTION_OPS = frozenset({"convolution"})
_SIGNATURE = ("input", "weight", "bias", "stride", "padding", "dilation",
              "transposed", "output_padding", "groups")
_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})
_DTYPE_ALIASES = {"half": "float16", "float": "float32", "double": "float64"}


def _dtype(value):
    if isinstance(value, dict) and set(value) == {"dtype"}:
        value = value["dtype"]
    value = str(value).removeprefix("torch.")
    return _DTYPE_ALIASES.get(value, value)


def _bind(args, kwargs, attrs=None):
    if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict):
        raise ValueError("convolution args/kwargs must be a sequence and object")
    if len(args) > len(_SIGNATURE):
        raise ValueError("convolution accepts at most nine arguments")
    if any(key not in _SIGNATURE for key in kwargs):
        raise ValueError("unknown convolution keyword(s): " + str(sorted(set(kwargs) - set(_SIGNATURE))))
    result = dict(zip(_SIGNATURE, args))
    for key, value in kwargs.items():
        if key in result:
            raise ValueError("duplicate convolution argument: " + key)
        result[key] = value
    attrs = attrs or {}
    for key in _SIGNATURE:
        if key not in result:
            if key in attrs and key not in ("input", "weight", "bias"):
                result[key] = attrs[key]
            elif key == "bias":
                result[key] = None
            else:
                raise ValueError("missing required convolution argument: " + key)
    return result


def _literal(value, values):
    if isinstance(value, dict) and set(value) == {"ref"}:
        meta = values[value["ref"]]
        if meta.get("kind") != "scalar" or "specialized_value" not in meta:
            raise ValueError("structural convolution arguments require specialized scalars")
        return meta["specialized_value"]
    if isinstance(value, (tuple, list)):
        return [_literal(item, values) for item in value]
    return value


def _vector(value, dimensions, name, minimum):
    # ATen accepts a one-element IntArrayRef and expands it over spatial axes.
    # A flat length-2*rank pad is NOT symmetric convolution padding.
    if not isinstance(value, (list, tuple)) or len(value) not in (1, dimensions):
        raise ValueError(f"{name} must contain one or {dimensions} spatial integers (symmetric per axis)")
    if any(type(item) is not int or item < minimum for item in value):
        raise ValueError(f"{name} entries must be integers >= {minimum}")
    return list(value) * dimensions if len(value) == 1 else list(value)


def _shape(meta, name):
    if not isinstance(meta, dict) or meta.get("kind", "tensor") != "tensor":
        raise ValueError(name + " requires tensor metadata")
    shape = meta.get("shape")
    if not isinstance(shape, (list, tuple)) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError(name + " shape must be concrete nonnegative integers")
    if _dtype(meta.get("dtype")) not in _FLOAT_DTYPES:
        raise ValueError(name + " requires a supported finite real floating dtype")
    # Literal constants may already carry data. Unknown runtime data is checked
    # by the CPU helper rather than pretending metadata proves finiteness.
    if "values" in meta:
        def finite(value):
            if isinstance(value, (list, tuple)):
                return all(finite(item) for item in value)
            return type(value) in (int, float) and math.isfinite(value)
        if not finite(meta["values"]):
            raise ValueError(name + " constant violates the finite floating contract")
    return list(shape)


def _tensor(value, values, name):
    if not isinstance(value, dict) or set(value) != {"ref"} or not isinstance(value["ref"], str):
        raise ValueError(name + " requires a tensor reference")
    ref = value["ref"]
    if ref not in values:
        raise ValueError("undefined convolution reference: " + ref)
    meta = values[ref]
    return ref, _shape(meta, name), meta


def _lower(node, values):
    bound = _bind(node.get("args", []), node.get("kwargs", {}))
    source_ref, source, source_meta = _tensor(bound["input"], values, "input")
    weight_ref, weight, weight_meta = _tensor(bound["weight"], values, "weight")
    dimensions = len(source) - 2
    if dimensions not in (1, 2, 3) or len(weight) != len(source):
        raise ValueError("convolution requires batched rank-3/4/5 input and same-rank weight")
    transposed = _literal(bound["transposed"], values)
    if type(transposed) is not bool:
        raise ValueError("transposed must be a boolean")
    if transposed:
        raise ValueError("transposed convolution is explicitly unsupported")
    stride = _vector(_literal(bound["stride"], values), dimensions, "stride", 1)
    padding = _vector(_literal(bound["padding"], values), dimensions, "padding", 0)
    dilation = _vector(_literal(bound["dilation"], values), dimensions, "dilation", 1)
    output_padding = _vector(_literal(bound["output_padding"], values), dimensions, "output_padding", 0)
    if any(output_padding):
        raise ValueError("nonzero output_padding is unsupported for nontransposed convolution")
    groups = _literal(bound["groups"], values)
    if type(groups) is not int or groups < 1:
        raise ValueError("groups must be a positive integer")
    if any(n == 0 for n in source[1:]) or any(n == 0 for n in weight):
        raise ValueError("empty channel, spatial, or kernel dimensions are unsupported; empty batch is allowed")
    if source[1] % groups or weight[0] % groups or weight[1] * groups != source[1]:
        raise ValueError("input/weight channels do not match groups")
    bias_ref, bias_meta = None, None
    if bound["bias"] is not None:
        bias_ref, bias, bias_meta = _tensor(bound["bias"], values, "bias")
        if bias != [weight[0]]:
            raise ValueError("bias must have exactly one entry per output channel")
    dtype = _dtype(source_meta.get("dtype"))
    if any(_dtype(meta.get("dtype")) != dtype for meta in (weight_meta, bias_meta) if meta is not None):
        raise ValueError("input, weight, and bias dtypes must agree")
    spatial = [(size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
               for size, kernel, pad, dil, step in zip(source[2:], weight[2:], padding, dilation, stride)]
    if any(n <= 0 for n in spatial):
        raise ValueError("effective kernel exceeds padded input or output spatial domain is empty")
    expected = [source[0], weight[0], *spatial]
    outputs = deepcopy(node.get("outputs", []))
    if len(outputs) != 1:
        raise ValueError("convolution requires exactly one tensor output")
    if _shape(outputs[0], "output") != expected:
        raise ValueError(f"captured output shape disagrees with convolution shape {expected}")
    if _dtype(outputs[0].get("dtype")) != dtype:
        raise ValueError("convolution output dtype must agree with input dtype")

    variables = [{"var": "r_inputchan", "extent": weight[1]}]
    variables += [{"var": f"r_kernel{axis}", "extent": size} for axis, size in enumerate(weight[2:])]
    coordinates = [f"(i{axis + 2} * {stride[axis]} - {padding[axis]} + r_kernel{axis} * {dilation[axis]})"
                   for axis in range(dimensions)]
    guard = " and ".join(f"(0 <= {coordinate} < {size})" for coordinate, size in zip(coordinates, source[2:]))
    input_channel = f"((i1 // {weight[0] // groups}) * {weight[1]} + r_inputchan)"
    reads = [
        {"value": source_ref, "indices": ["i0", input_channel, *coordinates],
         "role": "reduction", "operand": 0, "guard": guard, "scope": "reduction"},
        {"value": weight_ref, "indices": ["i1", "r_inputchan", *[v["var"] for v in variables[1:]]],
         "role": "reduction", "operand": 1, "guard": guard, "scope": "reduction"},
    ]
    reduction = {
        "axes": list(range(1, len(source))), "variables": variables,
        "operator": "sum-of-products", "identity": 0, "tree": "unspecified",
        "group_size": math.prod(v["extent"] for v in variables),
        "term_guard": guard, "guard_false_value": 0,
        "contraction_axes": {source_ref: list(range(1, len(source))),
                             weight_ref: list(range(1, len(weight)))},
    }
    if bias_ref is not None:
        # scalarize currently visits all reads at every reduction assignment.
        # The anchor guard preserves precisely one bias read, even if the first
        # kernel point is padding, and avoids a misleading repeated-add count.
        reads.append({"value": bias_ref, "indices": ["i1"], "role": "bias", "operand": 2,
                      "scope": "output", "frequency": "once-per-output",
                      "guard": " and ".join(f"({v['var']} == 0)" for v in variables)})
        reduction["finalize"] = {"op": "add-bias", "value": bias_ref, "indices": ["i1"],
                                 "read_index": 2, "frequency": "once-per-output"}
    attrs = {
        "spatial_rank": dimensions, "stride": stride, "padding": padding,
        "dilation": dilation, "transposed": False, "output_padding": output_padding,
        "groups": groups, "kernel_shape": weight[2:], "input_channels_per_group": weight[1],
        "output_channels_per_group": weight[0] // groups, "has_bias": bias_ref is not None,
        "padding_mode": "zeros", "padding_value": 0,
        "finite_float_contract": True,
    }
    sem = {
        "domain": [{"var": f"i{axis}", "extent": extent} for axis, extent in enumerate(expected)],
        "reads": reads, "reduction": reduction,
        "arithmetic": {
            "input_dtypes": [dtype] * (2 + int(bias_ref is not None)), "output_dtype": dtype,
            "accumulation_dtype": "float32" if dtype in ("float16", "bfloat16") else dtype,
            "rounding": "contraction-implementation-dependent-and-at-output",
            "reassociation": "not-authorized-by-default",
            "finite_inputs_required": True, "finite_output_required": True,
            "signed_zero": "not-distinguished", "multiply_accumulate": "implementation-dependent",
        },
        "order": "data-only",
        "notes": [
            "Forward cross-correlation over batch/output-channel/spatial coordinates; no kernel reversal.",
            "Padding is symmetric per spatial axis; invalid terms are literal zero and read neither input nor weight.",
            "Finite real arithmetic contract; NaN/Inf and signed-zero-sensitive equivalence are not claimed.",
            "Bias is added once per output after the reduction, never once per product.",
            "Exact scalar reduction group, not a selected reduction tree, schedule, or bitwise-equivalence claim.",
        ],
    }
    result = deepcopy(node)
    result.update(op="convolution", outputs=outputs, attributes=attrs, semantics=sem)
    return result


def lower_convolution_extension(node, values) -> dict | None:
    """Return a normalized node for our exact target, None for other extensions.

    Recognized but invalid/unsupported convolutions fail with node ID and target.
    ``values`` is the already-defined SSA metadata table, not runtime tensors.
    """
    if node.get("target") not in CONVOLUTION_TARGETS:
        return None
    try:
        return _lower(node, values)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise UnsupportedOperator(node.get("id", "<missing>"), node.get("target"), str(exc)) from exc


def execute_convolution_extension(node, args, kwargs, attrs):
    """Explicit F.conv1d/2d/3d CPU numeric helper; never arbitrary ATen dispatch.

    ``args``/``kwargs`` contain resolved runtime operands. Structural parameters
    may fall back to normalized attrs, and captured output shape/dtype is checked.
    Unknown ops/targets, nonfinite data, or unsupported convolution cases fail.
    """
    import torch
    import torch.nn.functional as functional

    target = node.get("target")
    if node.get("op") not in CONVOLUTION_OPS or (target is not None and target not in CONVOLUTION_TARGETS):
        raise UnsupportedOperator(node.get("id", "<missing>"), target or node.get("op"),
                                  "not an explicitly whitelisted convolution")
    try:
        bound = _bind(args, kwargs, attrs)
        metadata, encoded = {}, dict(bound)
        for key in ("input", "weight", "bias"):
            value = bound[key]
            if key == "bias" and value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided:
                raise ValueError(key + " requires a strided CPU tensor")
            if _dtype(value.dtype) not in _FLOAT_DTYPES or value.is_quantized:
                raise ValueError(key + " requires a supported real floating dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(key + " violates the finite floating contract")
            metadata[key] = {"id": key, "kind": "tensor", "shape": list(value.shape), "dtype": _dtype(value.dtype)}
            encoded[key] = {"ref": key}
        runtime_node = {"id": node.get("id", "<missing>"), "target": "aten.convolution.default",
                        "args": [], "kwargs": encoded, "outputs": node.get("outputs", [])}
        canonical = _lower(runtime_node, metadata)["attributes"]
        for key in canonical:
            if key in (attrs or {}) and attrs[key] != canonical[key]:
                raise ValueError("convolution attribute disagrees with runtime parameters: " + key)
        primitives = {1: functional.conv1d, 2: functional.conv2d, 3: functional.conv3d}
        primitive = primitives[canonical["spatial_rank"]]
        result = primitive(bound["input"], bound["weight"], bound["bias"],
                           tuple(canonical["stride"]), tuple(canonical["padding"]),
                           tuple(canonical["dilation"]), canonical["groups"])
        out = node["outputs"][0]
        if list(result.shape) != out["shape"] or _dtype(result.dtype) != _dtype(out["dtype"]):
            raise ValueError("CPU convolution output metadata mismatch")
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError("CPU convolution result violates the finite floating contract")
        return result
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OverflowError) as exc:
        raise UnsupportedOperator(node.get("id", "<missing>"), target or "convolution", str(exc)) from exc
