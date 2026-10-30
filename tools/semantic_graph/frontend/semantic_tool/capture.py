"""Trusted Python -> functional, concrete-shape ATen capture (no GPU execution).

IMPORTANT: fake tracing executes Python. It is NOT a Python security sandbox.
An unknown callable is rejected before execution. Inspect the callable, decorators,
closures and dependencies, then explicitly pass trusted=True or register it with
trust_callable(). Never use this entry point on untrusted uploaded Python code.
ATen capture does not imply semantic support: lower_capture rejects unknown ops.
"""
from __future__ import annotations

import inspect
import json
import math
import operator
import weakref
from typing import Any, Callable, Sequence

import torch
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx


class CaptureError(RuntimeError):
    """Capture is incomplete or cannot be represented faithfully."""


class UntrustedCallableError(CaptureError):
    """Python execution has not been explicitly authorized by the caller."""


_TRUSTED: weakref.WeakSet = weakref.WeakSet()


def trust_callable(function: Callable) -> Callable:
    """Acknowledge that this callable and its dependencies were inspected.

    This is an authorization boundary, NOT a static purity checker. Registration
    is per-process and per-callable; no module/path is implicitly trusted.
    """
    if not callable(function):
        raise TypeError("expected a callable")
    _TRUSTED.add(function)
    return function


def _dtype(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def capture(function: Callable, example_args: Sequence[Any], input_names=None,
            *, trusted: bool = False, max_constant_elements: int = 100000) -> dict:
    """Capture original ``function(*example_args)`` as ``aten-capture-v1``.

    Inputs are flat positional tensors (real, fake or meta) or JSON scalars.
    Python scalars and tensor shapes are specialized; 0-d tensors stay tensors.
    Functionalization removes mutations but retains views. Inputs mutated by the
    original callable are rejected (copy-back is outside returned-value semantics).
    Unknown Python FX callbacks, opaque literals, unavailable tensor constants,
    and non-ATen namespaces fail closed. Non-finite Python float literals are
    encoded strictly as {"float": "inf" | "-inf" | "nan"}, never JSON NaN/Infinity.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    if not trusted and function not in _TRUSTED:
        raise UntrustedCallableError(
            "Fake tracing executes Python: inspect callable/dependencies first, "
            "then pass trusted=True or call trust_callable(function).")
    args = tuple(example_args)
    for i, arg in enumerate(args):
        if not isinstance(arg, (torch.Tensor, bool, int, float, str, type(None))):
            raise CaptureError(f"input {i}: unsupported argument type {type(arg).__name__}")
    if input_names is None:
        try:
            names = list(inspect.signature(function).parameters)
            input_names = names if len(names) == len(args) else None
        except (TypeError, ValueError):
            input_names = None
    names = list(input_names) if input_names is not None else [f"arg{i}" for i in range(len(args))]
    if len(names) != len(args) or len(set(names)) != len(names) or not all(isinstance(n, str) for n in names):
        raise CaptureError("input_names must be unique strings matching example_args")
    notes = ["Concrete tensor shapes/strides specialized; Python shape loops are unrolled.",
             "Trusted Python executes during tracing; fake mode is not a security sandbox.",
             "Functionalization removes mutations, preserving view relations; introduced functional copies do not mandate physical storage."]
    env: dict[Node, Any] = {}
    constants = {}
    nonfinite_tokens = set()
    graph = {"format": "aten-capture-v1", "inputs": [], "nodes": [],
             "outputs": None, "constants": constants,
             "metadata": {"tracing_mode": "fake", "functionalization": "mutations",
                          "trusted_callable": True, "shape_specialization": "concrete",
                          "python_scalars_specialized": True,
                          "device_execution": "fake/meta (no GPU kernel execution)", "notes": notes}}

    def encode(value):
        if isinstance(value, Node):
            if value not in env:
                raise CaptureError(f"unresolved or forward reference {value.name}")
            return env[value]
        if isinstance(value, torch.dtype):
            return {"dtype": _dtype(value)}
        if isinstance(value, torch.device):
            return {"device": str(value)}
        if isinstance(value, torch.memory_format):
            return {"memory_format": str(value).removeprefix("torch.")}
        if isinstance(value, torch.layout):
            return {"layout": str(value).removeprefix("torch.")}
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            notes.append(f"Specialized symbolic scalar {value}")
            value = int(value) if isinstance(value, torch.SymInt) else (float(value) if isinstance(value, torch.SymFloat) else bool(value))
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                token = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
                nonfinite_tokens.add(token)
                return {"float": token}
            return value
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise CaptureError("JSON mapping keys must be strings")
            return {key: encode(item) for key, item in value.items()}
        raise CaptureError(f"unsupported literal/callback type {type(value).__name__}")

    def descriptor(value, identifier):
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise CaptureError(f"{identifier}: unsupported tensor layout {value.layout}")
            return {"id": identifier, "kind": "tensor", "shape": [int(n) for n in value.shape],
                    "dtype": _dtype(value.dtype), "strides": [int(s) for s in value.stride()],
                    "device": str(value.device), "requires_grad": bool(value.requires_grad),
                    "storage_offset": int(value.storage_offset())}
        return {"id": identifier, "kind": "scalar", "dtype": type(value).__name__,
                "specialized_value": encode(value)}

    try:
        # No real-mode fallback: a failed fake capture is not permission to run GPU.
        gm = make_fx(torch.func.functionalize(function, remove="mutations"),
                     tracing_mode="fake", _allow_non_fake_inputs=True)(*args)
    except Exception as exc:
        raise CaptureError(f"fake ATen tracing failed: {type(exc).__name__}: {exc}") from exc

    input_index = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if input_index >= len(args):
                raise CaptureError("unexpected flattened placeholder structure")
            original = args[input_index]
            desc = descriptor(node.meta.get("val", original) if isinstance(original, torch.Tensor) else original, node.name)
            desc["name"] = names[input_index]
            graph["inputs"].append(desc)
            env[node] = {"ref": node.name}
            input_index += 1
        elif node.op == "get_attr":
            value = gm
            for component in str(node.target).split("."):
                value = getattr(value, component)
            if not isinstance(value, torch.Tensor):
                env[node] = encode(value)
                continue
            if value.numel() > max_constant_elements:
                raise CaptureError(f"{node.name}: tensor constant exceeds max_constant_elements={max_constant_elements}")
            if value.device.type != "cpu" or isinstance(value, torch._subclasses.fake_tensor.FakeTensor):
                raise CaptureError(f"{node.name}: tensor constant values unavailable on CPU")
            desc = descriptor(value, node.name)
            desc["values"] = encode(value.detach().tolist())
            constants[node.name] = desc
            env[node] = {"ref": node.name}
        elif node.op == "call_function":
            if node.target is operator.getitem:
                parent, index = node.args
                if not isinstance(parent, Node) or parent not in env or not isinstance(env[parent], (list, dict)):
                    raise CaptureError(f"{node.name}: getitem is only valid for structured ATen outputs")
                try:
                    env[node] = env[parent][index]
                except (IndexError, KeyError, TypeError) as exc:
                    raise CaptureError(f"{node.name}: invalid structured getitem") from exc
                continue
            if not isinstance(node.target, torch._ops.OpOverload) or node.target.namespace != "aten":
                raise CaptureError(f"{node.name}: unsupported callback/target {node.target}")
            target = str(node.target)
            schema = node.target._schema
            if schema.is_mutable:
                raise CaptureError(f"{node.name}: unresolved mutation/input copy-back {target}")
            val = node.meta.get("val")
            if "val" not in node.meta:
                raise CaptureError(f"{node.name}: output metadata unavailable for {target}")
            outputs = []

            def add_output(v, identifier):
                if isinstance(v, (list, tuple)):
                    return [add_output(item, f"{identifier}:{i}") for i, item in enumerate(v)]
                if isinstance(v, dict):
                    return {key: add_output(item, f"{identifier}:{key}") for key, item in v.items()}
                outputs.append(descriptor(v, identifier))
                return {"ref": identifier}

            refs = add_output(val, node.name)
            encoded_args, encoded_kwargs = encode(node.args), encode(node.kwargs)
            graph["nodes"].append({"id": node.name, "target": target, "args": encoded_args,
                                   "kwargs": encoded_kwargs, "outputs": outputs})
            env[node] = refs
        elif node.op == "output":
            def encode_return(value):
                # In v1 {'ref': STRING} is reserved for graph references. A user
                # mapping with this exact shape cannot be distinguished after
                # JSON encoding, so reject it rather than reinterpret user data.
                if isinstance(value, Node):
                    return encode(value)
                if isinstance(value, dict):
                    if set(value) == {"ref"} and isinstance(value["ref"], str):
                        raise CaptureError("user output {'ref': string} collides with reserved v1 reference encoding")
                    if set(value) == {"float"} and isinstance(value["float"], str) and value["float"] in ("inf", "-inf", "nan"):
                        raise CaptureError("user output {'float': nonfinite-token} collides with reserved v1 float encoding")
                    if not all(isinstance(key, str) for key in value):
                        raise CaptureError("JSON mapping keys must be strings")
                    return {key: encode_return(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [encode_return(item) for item in value]
                return encode(value)
            graph["outputs"] = encode_return(node.args[0])
        else:
            raise CaptureError(f"{node.name}: unsupported FX node kind {node.op}")
    if input_index != len(args):
        raise CaptureError("placeholder count does not match inputs")
    graph["metadata"]["targets"] = sorted({node["target"] for node in graph["nodes"]})
    graph["metadata"]["nonfinite_literal_encoding"] = {"tag": "float", "tokens": ["inf", "-inf", "nan"]}
    graph["metadata"]["nonfinite_literals"] = sorted(nonfinite_tokens)
    json.dumps(graph, allow_nan=False)
    return graph


__all__ = ["capture", "trust_callable", "CaptureError", "UntrustedCallableError"]
