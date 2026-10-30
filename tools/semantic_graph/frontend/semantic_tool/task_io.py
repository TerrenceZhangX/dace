"""Definition/workload loading and CPU/meta examples; no task-ID formulas.

Python import and custom get_inputs are an explicit trusted-source boundary.
Official axes are always retained separately from reduced probes. Meta examples
allocate no tensor payload and preserve original dtypes, including 0-d tensors.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import operator
import os
from pathlib import Path
import types
from typing import Any

import torch

from .capture import CaptureError, UntrustedCallableError, trust_callable

# Optional external data root; the portable frontend does not bundle benchmarks.
DEFAULT_TASK_ROOT = Path(os.environ.get("SOL_EXECBENCH_L1_DIR") or
                         (Path(__file__).resolve().parents[1] / "research" / "minimal_tasks" / "L1"))
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.FloorDiv: operator.floordiv, ast.Div: operator.truediv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}


def resolve_expression(expression, axes: dict[str, int]) -> int:
    """Evaluate only integer shape arithmetic, never Python eval or callbacks."""
    if isinstance(expression, int) and not isinstance(expression, bool):
        return expression
    if not isinstance(expression, str):
        raise ValueError(f"invalid dimension {expression!r}")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return axes[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 32:
                raise ValueError("shape exponent exceeds safety bound")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError(f"unsupported shape expression {expression!r}")

    value = visit(ast.parse(expression, mode="eval").body)
    if int(value) != value or value < 0:
        raise ValueError(f"dimension must be a nonnegative integer: {expression!r}")
    return int(value)


def resolve_axes(definition: dict, workload: dict, overrides=None) -> dict[str, int]:
    axes = dict(workload.get("axes", {}))
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(definition["axes"])
    if unknown:
        raise ValueError(f"unknown axes overrides: {sorted(unknown)}")
    for name, spec in definition["axes"].items():
        if spec["type"] == "const":
            axes[name] = spec["value"]
    axes.update(overrides)
    pending = {name: spec["expression"] for name, spec in definition["axes"].items()
               if spec["type"] == "expr" and name not in overrides}
    while pending:
        progress = False
        for name, expr in list(pending.items()):
            try:
                axes[name] = resolve_expression(expr, axes)
            except KeyError:
                continue
            del pending[name]
            progress = True
        if not progress:
            raise ValueError(f"unresolved/cyclic axes expressions: {pending}")
    for name in definition["axes"]:
        if name not in axes or type(axes[name]) is not int or axes[name] < 0:
            raise ValueError(f"missing/invalid concrete axis {name}")
    return axes


def discover_tasks(root=DEFAULT_TASK_ROOT) -> list[Path]:
    return sorted(path.parent for path in Path(root).glob("*/definition.json"))


def load_definition(task_dir) -> dict:
    with (Path(task_dir) / "definition.json").open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_workloads(task_dir) -> list[dict]:
    with (Path(task_dir) / "workload.jsonl").open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_reference(source_path, *, trusted=False):
    if not trusted:
        raise UntrustedCallableError("Import executes Python: inspect reference and dependencies, then pass trusted=True")
    source_path = Path(source_path).resolve()
    # In-memory compilation avoids modifying upstream __pycache__ artifacts.
    source = source_path.read_text(encoding="utf-8-sig")
    module = types.ModuleType("semantic_reference_" + hashlib.sha256(str(source_path).encode()).hexdigest()[:16])
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    if not callable(getattr(module, "run", None)):
        raise ValueError("reference must define callable run")
    trust_callable(module.run)
    return module


def _probe_overrides(definition, official_axes, source, cap):
    # Literal dimensions assigned in run cannot be shrunk without rewriting the
    # original source. Conservatively retain matching axis-name/value assignments.
    fixed = set()
    tree = ast.parse(source)
    for fn in tree.body:
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == "run":
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in official_axes and node.value.value == official_axes[target.id]:
                            fixed.add(target.id)
    # Only shape-relevant axes shrink: non-shape scalar constants such as a
    # softcapping threshold must not acquire a misleading changed value.
    shape_axes = set()
    for spec in list(definition["inputs"].values()) + list(definition.get("outputs", {}).values()):
        for dim in spec.get("shape") or []:
            if isinstance(dim, str):
                shape_axes.update(node.id for node in ast.walk(ast.parse(dim, mode="eval")) if isinstance(node, ast.Name))
    pending = list(shape_axes)
    while pending:
        name = pending.pop()
        spec = definition["axes"].get(name, {})
        if spec.get("type") == "expr":
            dependencies = {node.id for node in ast.walk(ast.parse(spec["expression"], mode="eval")) if isinstance(node, ast.Name)}
            pending.extend(dependencies - shape_axes)
            shape_axes.update(dependencies)
    return {name: min(value, cap) for name, value in official_axes.items()
            if name in shape_axes and definition["axes"][name]["type"] != "expr" and name not in fixed}


@dataclass
class Task:
    task_dir: Path
    definition: dict
    workloads: list[dict]
    workload_index: int
    function: Any
    input_names: list[str]
    example_args: tuple
    source_path: Path
    official_axes: dict
    probe_axes: dict
    metadata: dict
    module: Any

    @property
    def name(self):
        return self.definition.get("name", self.task_dir.name)

    @property
    def axes(self):
        return self.probe_axes

    @property
    def workload(self):
        return self.workloads[self.workload_index]

    @property
    def input_meta(self):
        return input_metadata(self.definition, self.workload, self.probe_axes,
                              scalar_values=dict(zip(self.input_names, self.example_args)))


def _validate_scalar(value, dtype, name):
    declared = getattr(torch, dtype, None)
    if not isinstance(declared, torch.dtype):
        raise ValueError(f"{name}: unknown scalar dtype {dtype}")
    if declared == torch.bool:
        valid = type(value) is bool
    elif declared.is_floating_point:
        valid = type(value) in (int, float) and math.isfinite(value)
    else:
        valid = type(value) is int
        if valid:
            limits = torch.iinfo(declared)
            valid = limits.min <= value <= limits.max
    if not valid:
        raise ValueError(f"{name}: Python scalar {value!r} violates declared dtype {dtype}; tensors are not scalar placeholders")
    return value


def input_metadata(definition, workload, axes, *, scalar_values=None) -> list[dict]:
    result = []
    for name, spec in definition["inputs"].items():
        item = {"name": name, "dtype": spec["dtype"]}
        shape = spec.get("shape")
        if shape is None:
            init = workload.get("inputs", {}).get(name, {})
            item["kind"] = "scalar"
            if init.get("type") == "scalar" and "value" in init:
                item["specialized_value"] = _validate_scalar(init["value"], spec["dtype"], name)
            elif init.get("type") == "custom" and definition.get("custom_inputs_entrypoint"):
                item["initialization"] = "custom"
                if scalar_values is not None and name in scalar_values:
                    item["specialized_value"] = _validate_scalar(scalar_values[name], spec["dtype"], name)
                else:
                    item["pending_custom_initialization"] = True
            else:
                raise ValueError(f"{name}: scalar input requires explicit workload scalar value or a custom generator")
        else:
            item.update(kind="tensor", shape=[resolve_expression(dim, axes) for dim in shape])
        result.append(item)
    return result


def make_example_args(definition, workload, axes, *, device="meta", module=None,
                      trusted=False, seed=0, max_elements=10_000_000):
    """Generate flat inputs in definition order, meta or bounded CPU only.

    Custom input generators run unchanged on the requested CPU/meta device under
    fork_rng; this preserves e.g. initialized recurrent states without task rules.
    Unknown initialization types fail closed rather than silently using random.
    """
    device = torch.device(device)
    if device.type not in ("cpu", "meta"):
        raise ValueError("task_io permits only CPU or meta examples; GPU execution is forbidden")
    metas = input_metadata(definition, workload, axes)
    elements = sum(math.prod(item["shape"]) for item in metas if item["kind"] == "tensor")
    if device.type == "cpu" and elements > max_elements:
        raise ValueError(f"CPU example requires {elements} elements, exceeds cap {max_elements}; use meta or smaller probe axes")
    custom_name = definition.get("custom_inputs_entrypoint")
    custom_inputs = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if custom_name:
            if not trusted or module is None:
                raise UntrustedCallableError("custom input generator needs inspected module and trusted=True")
            generator = getattr(module, custom_name, None)
            if not callable(generator):
                raise ValueError(f"custom generator {custom_name!r} missing")
            axes_and_scalars = dict(axes)
            axes_and_scalars.update({m["name"]: m["specialized_value"] for m in metas if m["kind"] == "scalar" and "specialized_value" in m})
            custom_inputs = generator(axes_and_scalars, device)
            if not isinstance(custom_inputs, dict):
                raise ValueError("custom generator must return an input-name dictionary")
        args = []
        for item in metas:
            name = item["name"]
            if item["kind"] == "scalar":
                if item.get("initialization") == "custom":
                    if name not in custom_inputs:
                        raise ValueError(f"{name}: missing custom scalar input")
                    value = custom_inputs[name]
                else:
                    value = item["specialized_value"]
                args.append(_validate_scalar(value, item["dtype"], name))
                continue
            dtype = getattr(torch, item["dtype"], None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"{name}: unknown dtype {item['dtype']}")
            init = workload.get("inputs", {}).get(name, {"type": "random"})
            kind = init.get("type")
            if kind == "custom":
                if name not in custom_inputs:
                    raise ValueError(f"{name}: missing custom input")
                value = custom_inputs[name]
            elif kind == "random":
                if device.type == "meta":
                    value = torch.empty(item["shape"], dtype=dtype, device=device)
                elif dtype.is_floating_point:
                    value = torch.randn(item["shape"], dtype=dtype, device=device)
                elif dtype == torch.bool:
                    value = torch.randint(0, 2, item["shape"], device=device).bool()
                else:
                    value = torch.randint(0, 5, item["shape"], dtype=dtype, device=device)
            elif kind in ("zeros", "ones", "full"):
                fill = 0 if kind == "zeros" else (1 if kind == "ones" else init["value"])
                value = torch.full(item["shape"], fill, dtype=dtype, device=device)
            else:
                raise ValueError(f"{name}: unsupported initialization {kind!r}")
            if not isinstance(value, torch.Tensor) or list(value.shape) != item["shape"] or value.dtype != dtype or value.device != device:
                raise ValueError(f"{name}: generated input violates declared shape/dtype/device contract")
            args.append(value)
    return tuple(args)


def load_task(task_dir, workload_index=0, *, probe=True, probe_cap=4,
              axes_overrides=None, device="meta", trusted=False, seed=0,
              max_elements=10_000_000) -> Task:
    """Load metadata plus unchanged reference.run and original-dtype examples.

    probe=True caps independent axes (including constants) except source-literal
    fixed axes; derived axes are recomputed. This is a SHAPE PROBE, not an official
    workload. probe=False uses official axes unless explicitly overridden. Metadata
    always records both axis sets and every difference. Arbitrary shape-dependent
    functions may reject probes: no task-ID rewrite/fallback is attempted.
    """
    task_dir = Path(task_dir).resolve()
    definition, workloads = load_definition(task_dir), load_workloads(task_dir)
    workload = workloads[workload_index]
    official = resolve_axes(definition, workload)
    source_path = task_dir / "reference.py"
    overrides = _probe_overrides(definition, official, source_path.read_text(encoding="utf-8-sig"), probe_cap) if probe else {}
    overrides.update(axes_overrides or {})
    axes = resolve_axes(definition, workload, overrides)
    module = load_reference(source_path, trusted=trusted)
    names = list(definition["inputs"])
    signature_names = list(inspect.signature(module.run).parameters)
    if signature_names != names:
        raise ValueError(f"reference signature {signature_names} does not match definition input order {names}")
    args = make_example_args(definition, workload, axes, device=device, module=module,
                             trusted=trusted, seed=seed, max_elements=max_elements)
    changes = {key: {"official": official[key], "example": axes[key]} for key in axes if official[key] != axes[key]}
    metadata = {"task": definition.get("name", task_dir.name), "workload_index": workload_index,
                "workload_uuid": workload.get("uuid"), "official_axes": official,
                "example_axes": axes, "probe_axes": axes, "axis_changes": changes,
                "shape_scope": "reduced-probe-not-official" if changes else "official-workload",
                "original_dtypes": True, "example_device": str(device),
                "custom_initialization": definition.get("custom_inputs_entrypoint"),
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
    return Task(task_dir, definition, workloads, workload_index, module.run, names, args,
                source_path, official, axes, metadata, module)


__all__ = ["Task", "DEFAULT_TASK_ROOT", "discover_tasks", "load_definition", "load_workloads",
           "load_reference", "load_task", "resolve_axes", "resolve_expression",
           "input_metadata", "make_example_args"]
"""No existing source, environment, or prior audit evidence is modified."""
