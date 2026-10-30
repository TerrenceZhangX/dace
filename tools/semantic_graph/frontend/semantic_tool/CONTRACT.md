# Typed semantic graph contracts

This optional frontend automatically captures an explicitly trusted PyTorch callable with concrete input metadata and lowers supported operators into typed, parameterized relations. It is not a universal functional-spec inference system or an all-algorithm equivalence proof. No task identifier selects an implementation rule.

## Capture

`capture(function, example_args, input_names=None, trusted=False)` produces `aten-capture-v1`. Fake tracing executes Python and is not a sandbox. Trust requires inspection of the callable and dependencies. Functionalization removes intermediate mutations; unresolved caller-input mutation is rejected.

Each input contains id/name/kind/shape/dtype/strides/device/storage_offset as applicable. Nodes have id, exact ATen target overload, encoded args/kwargs, and output descriptors. Values reference SSA identifiers with `{"ref":"id"}`; typed dtype/device/layout/memory_format markers are explicit. Nonfinite floats are strict JSON tags `{"float":"inf"}`, `{"float":"-inf"}`, `{"float":"nan"}`. Colliding user return dictionaries are rejected rather than ambiguously decoded. Concrete shape and Python scalar specialization are recorded; unsupported data-dependent capture fails explicitly.

## Semantic relations

`lower_capture(captured)` produces `semantic-graph-v1`. It preserves the source capture parameters while adding:

- `op`: normalized, explicitly supported operator.
- `attributes`: axes, dimensions, shape, scaling, dtype or other operator parameters.
- `semantics.domain`: output iteration coordinates and extents.
- `semantics.reads`: exact source value identifiers and logical input indices, with optional index guards.
- `semantics.reduction`: reduction variables/extents, combiner, identity and unspecified tree.
- `semantics.arithmetic`: typed rounding/accumulation contract without inventing backend association.
- `semantics.layout`: stride/offset/alias metadata distinct from required value dependencies.
- `semantics.constant_regions`: padding/literal regions without fake tensor reads.
- `semantics.order="data-only"`: no execution scheduling edges are introduced.

Partition operators such as split and unbind are normalized to exact single-output views. Unknown operations never pass through as opaque analyzed nodes. Semantic annotations can be recomputed using `verify_semantics`; consistency does not authenticate source provenance.

## Evaluation and expansion

`evaluate(graph, inputs)` independently interprets explicitly supported normalized CPU primitives without calling the original function. It validates value metadata and mutations. This is not native DaCe/GPU execution.

`instantiate(graph, max_nodes, max_edges)` creates small scalar-value dependency instances. Reductions remain explicit group relations rather than claiming a particular primitive addition tree. Large workloads remain parameterized; expansion budgets fail explicitly. Where data guards cannot be resolved from indices, a conservative may-dependence relation must be labelled rather than claimed exact.

## DaCe bridge

The sibling bridge consumes this JSON to create native SDFGs and verify native memlet dependencies. Its supported operators, packed logical ABI, numerical preconditions, conservative state schedule and compilation limitations are documented separately. SDFG state ordering is not automatically part of the source functional specification.

External benchmark datasets are not bundled. Set `SOL_EXECBENCH_L1_DIR` to a separately obtained, licensed dataset when running optional integration tests.
