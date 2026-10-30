# Semantic frontend for the DaCe development fork

## Architecture and status

This is a **frontend and independent verification layer for DaCe**, not a replacement compiler. It automatically consumes an inspected original PyTorch callable plus concrete input metadata, captures functional ATen, builds typed/indexed semantic relations, optionally expands small scalar dependencies, and exports JSON for the DaCe SDFG bridge.

```text
original trusted PyTorch function + shape/dtype/scalar contract
 → FakeTensor ATen capture + functional SSA mutation removal
 → target-driven semantic rules (no task-ID formulas)
 → typed parameterized semantic-graph-v1
 → optional scalar value dependencies / independent CPU interpreter
 → DaCe tools/semantic_graph/dace_bridge.py
 → native SDFG, MapUnroll, native memlet dependency verification
```

The DaCe fork is https://github.com/TerrenceZhangX/dace/tree/semantic-l1-bootstrap . Native transformations and code generation remain DaCe's job. SDFG execution-state order is never claimed to be a necessary source semantic dependency.

## Current measured coverage

- Full L1 inventory: **94 tasks / 1,480 workloads**. All94 reference source hashes statically reviewed; two hardcoded-CUDA sources blocked in this CPU/fake execution setup.
- First official-shape per-task sweep: 79 captures, 36 semantic graph constructions.
- Updated sweep after generic repairs: **83 captures, 67 complete parameterized semantic graph constructions out of 94**; 16 lowering gaps,9 capture gaps,2 source-device blocks remain.
- Those counts mean **capture/lowering/structure validation**, not correct complete scalar expansion or GPU execution.
- Earlier nine-task small-shape set: all9 automatic capture→lower→scalarize→CPU interpretation passed8 seeds each, exact input-coordinate ancestor sets verified for6,310 output elements.
- Nine-task original official workload configurations: all144 fake capture/lowering checks passed before all-L1 expansion. No numerical execution of those144.
- DaCe bridge:9/9 small parameterized graphs+CPU code generation;8/9 native unroll+immediate input-coordinate cone checks. Task010 expansion58,386 instances exceeded120-second retry budget; no successful unroll claim for it.
- Frontend tests currently **199 unittest methods** (108 in semantic_tool/tests,91 in sibling tests); all passed in recorded final run. Includes generic end-to-end convolution, statistics, layout, dtype, alias, SSA mutation, strict failure and annotation tampering cases.
- DaCe fork tests:23 MapUnroll/scope-related tests and11 generic bridge tests independently rerun and passed. Native compiled numerical execution remains unverified because no C++ compiler is available.

These counts are a checkpoint, not completion of the all-L1 objective. Unhandled dynamic-index/control operators are explicitly rejected, never hidden as opaque successful nodes.

## API

```python
from semantic_tool import build
from semantic_tool.scalarize import instantiate
from semantic_tool.evaluate import evaluate

def fn(x, bias):
    return (x + bias).sum(dim=-1)

# trusted=True explicitly acknowledges that Python/dependencies were inspected.
graph = build(fn, (x, bias), input_names=['x', 'bias'], trusted=True)
small_dependencies = instantiate(graph, max_nodes=20000)
result = evaluate(graph, {'x': x, 'bias': bias})
```

Capture executes Python, even in fake mode. It is **not a security sandbox**. Unknown callables are rejected unless explicitly trusted/registered. A data-dependent branch that cannot be captured is not silently specialized using sample tensor values. Existing input mutation is rejected until an explicit effect contract is available. CPU/meta device support does not authorize CUDA execution.

### Benchmark task directory

```powershell
python -m semantic_tool.cli <task-dir> --trusted-source --output semantic.json
python -m semantic_tool.cli <task-dir> --trusted-source --probe-cap 3 --output small.json --scalar-output scalar.json
```

The task directory provides definition, original reference and workload files. Official-shape mode is default for CLI. Reduced probes change shape contracts and are labelled as such; generic cap heuristics may be invalid when source dimensions are hardcoded, so use official meta capture as the coverage baseline.

Input source data is **not distributed in the DaCe fork**. Obtain SOL-ExecBench directly from its official dataset subject to its license; the evaluator being open source does not make dataset artifacts BSD-licensed.

## What the graph represents

- Pointwise/broadcast relations specify exact output-domain to input-index maps, including zero-dimensional tensors and Python scalars separately.
- Reductions, softmax, contractions, variance and convolution specify dependency groups without prematurely choosing a tree/algorithm.
- Layout operations retain logical maps and physical stride/offset/alias metadata. The separate DaCe bridge currently uses a packed logical-value ABI rather than promising physical alias preservation.
- Functional scatters preserve old destination dependencies exactly outside the written region. Padding introduces literal regions with no fake input reads.
- Dtype boundaries, BF16/FP16 output rounding, wrapped scalar semantics and unspecified backend accumulation details remain explicit.
- `verify_semantics` regenerates annotations from target/args/metadata, detecting stale or changed index rules; this is not cryptographic source authentication.
- Scalar dependency graph groups are not all reduced to unique primitive add trees. No universal minimum-span or complete-algorithm-space proof is claimed.

## Remaining work toward complete L1

1. Data-dependent indexing/scatter/gather, sort/topk and dynamic output shape/control flow.
2. Multiple-output native normalization, FFT and vector-norm functional rules.
3. Input mutation effects and source-hardcoded CUDA handling without modifying the benchmark's functional contract.
4. Literal tensor creation on meta that currently loses constant payload.
5. Numerical tests for every real task, not just parameterized structure; robust shape probes and runtime memory budgets.
6. Bridge all newly implemented frontend rules into native DaCe and verify its expanded tasklets/memlets.
7. Compiled CPU/GPU numerical acceptance with explicit reduction/NaN/Inf/alias policy.

Fail-closed rejection and resource-limited expansion are recorded separately from semantic error. They are not considered successful expansion.
