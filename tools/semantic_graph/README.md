# Optional semantic JSON → DaCe bridge

This small, standalone integration consumes `semantic-graph-v1` JSON and builds native DaCe SDFGs. It uses explicit normalized operations, typed tasklets, maps, exact read-index memlets, initialized WCR accumulators, and functional SSA updates. It does **not** call an original Python function, import PyTorch, dispatch arbitrary ATen targets, or branch on benchmark/task identifiers. Tests contain only generic hand-authored tensor graphs; no benchmark source/data is included.

## Usage

From a checkout with its existing DaCe environment:

```sh
python -m unittest tools.semantic_graph.test_dace_bridge -v
python -m tools.semantic_graph.dace_bridge input_graph.json --output output_dir --unroll
# Explicit restricted numeric policy for maximum/amax/softmax:
python -m tools.semantic_graph.dace_bridge input_graph.json --output output_dir --finite-only --unroll
```

The CLI writes a compact mapped SDFG, native generated code, and a report. With `--unroll`, it additionally applies native `MapUnroll`, validates the result, checks exact semantic input-coordinate dependency sets, and saves the unrolled SDFG. Expansion defaults to a 20,000-map-instance cap; the API additionally caps semantic read checks. Do not unroll large official workloads indiscriminately.

```python
from tools.semantic_graph.dace_bridge import lower_to_sdfg, unroll_and_check

sdfg, manifest = lower_to_sdfg(graph, finite_only=False)
expanded, check = unroll_and_check(graph, sdfg, manifest)
```

A trusted producer may pass `annotation_verifier=producer_verify_function` to `lower_to_sdfg`. The local validator checks schema, SSA ordering, shapes, dtype support, and a closed coordinate-expression grammar, but **does not authenticate producer annotations or prove they match the original source**. The optional verifier should regenerate semantic rules from the producer's captured operations before building the SDFG.

## Supported baseline

- Floating pointwise add/sub/rsub/mul/div/pow, neg, exp/tanh/sigmoid/sqrt/rsqrt.
- GELU with explicit `approximate="none"` (erf) or `"tanh"` expressions, FP32 opmath for low-precision storage, and one output rounding boundary. Unknown approximation modes are rejected.
- Floating maximum and amax under the explicit `finite_only=True` policy.
- Casts, packed logical layout maps (reshape/view/transpose/permute/expand/select/slice), clone/alias/copy, squeeze/unsqueeze.
- Sum/mean and matrix/batched contractions from explicit reduction index maps.
- Floating addmm: matrix-product WCR followed by one alpha scaling and one broadcast bias beta scaling/add per output. `beta=0` omits bias reads entirely. Empty K omits all matrix reads and does not compute `alpha*0`, preserving the producer’s zero-contraction convention.
- Stable functional softmax under `finite_only=True`: row maximum → subtract/exp → sum → divide.
- `zeros_like` with metadata-only input dependencies.
- Cat with normalized disjoint segments, including repeated tensor sources and empty segments.
- Constant padding/cropping with at most `2*rank+1` nonoverlapping rectangles. Literal padding has no tensor value dependency; negative padding crops exact source coordinates. Fill must currently be finite, with checked integer range.
- Functional select/scatter and positive-step slice/scatter. Disjoint prefix/suffix/gap/update maps ensure an unselected source index is never loaded. Old destination elements remain dependencies exactly outside the updated region.
- FP16 and BF16 storage with FP32 pointwise opmath and reduction accumulators, explicit casts at semantic node output boundaries. Wrapped scalar literals are not pre-rounded to the low-precision storage type.

All unsupported operations fail closed. Examples include variance and convolution (not added by this incremental bridge change), tensor-constant initialization, generic factories other than zeros_like, integer arithmetic/reduction promotion and overflow, rounded integer division, empty mean, scalar/empty-axis softmax, complex arithmetic, and unknown operators.

## Important semantic boundaries

### Numeric policy

`finite_only` is an explicit **caller-asserted runtime precondition** for maximum/amax/softmax. It is recorded in the manifest, not dynamically verified on arrays. Inputs to these operators must be finite; the bridge does not claim PyTorch NaN/Inf propagation or signed-zero bit equivalence. A production runner must verify the precondition or supply and test a stronger numeric implementation.

FP32 accumulators avoid rounding BF16/FP16 after every WCR contribution. This does not fix an FP reduction tree or prove identical results to a particular library kernel. WCR specifies a combiner, not a unique association. Special-function ulps, subnormals, overflow, FMA contraction, reassociation, device math, and native rounding require independent compiled numerical tests. Do not enable unauthorized fast-math transformations.

### ABI and layout

Arrays use **packed logical tensor values**; source physical strides/storage offsets and caller-visible aliases are not reproduced. Layout operators are value-map materializations. Input packing and output unpacking/return-structure resolution belong in an execution adapter. A shape-[] tensor remains a one-element array, distinct from a specialized Python scalar. This baseline must not be advertised as a physical-view/alias-preserving ABI.

### Schedule and graph meaning

The bridge deliberately selects a conservative CPU execution schedule: sequential maps in a chain of SSA-operation states, with extra reduction initialization/finalization states. Those state barriers are not claimed to exist in the source semantic graph and do not establish optimal span. Later DaCe optimization/scheduling is separate from frontend dependency semantics.

`unroll_and_check` tracks actual native tasklet memlet dependencies through temporary reduction/softmax arrays, stopping at each semantic SSA output boundary. It compares each semantic output's immediate tensor-input-coordinate **set** with the supplied `reads`/reduction/guard rules. This includes whole-row softmax, exclusive SSA scatter dependencies, cat segments, literal padding, and addmm final bias dependencies.

The checker also counts static tasklet AST arithmetic occurrences (`add`, `mul`, etc.), casts, special-function calls, WCR contributions, and per-semantic-node tensor read occurrences before and after native MapUnroll. Mapped counts are weighted by exact map cardinality and must match the unrolled graph. Generic tests separately assert hand-calculated counts: addmm bias must be read once per output rather than once per K, and each GELU output must execute exactly one erf/tanh call. These are explicit implementation-operation counts, **not hardware FLOPs, optimal arithmetic work, numeric equivalence, or source authenticity**. A count-preserving incorrect formula still requires separate numerical validation.

### Verification status

The expanded suite executed 17 generic unittest methods successfully against DaCe 2.0.0b1 (including subcases for GELU modes, empty K, zero beta, and different softmax axes). Successful cases exercise native SDFG validation, CPU code generation, native `MapUnroll`, exact input-coordinate dependency checking, and arithmetic/read-occurrence preservation; negative cases exercise unsupported operations, hostile index syntax, guard tampering, and expansion caps. Softmax tests include different axes/ranks and BF16/FP64 storage.

No native compilation or native numerical execution is implied by these tests. Code generation is not compilation. Run a separate authorized C++/GPU-toolchain numerical acceptance suite before making execution-equivalence or performance claims.
