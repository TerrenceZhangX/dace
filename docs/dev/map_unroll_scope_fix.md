# MapUnroll: preserve the selected scope's boundary

## Defect

`MapUnroll` previously selected `weakly_connected_component(state, map_entry)`, replicated the selected nodes, and removed the original selection. That helper independently follows successors and predecessors; it does not repeatedly traverse both directions. As a result, the selected graph can have edges crossing its boundary.

Two minimal generic examples demonstrate the problem:

```
A -> map1 -> B       A -> producer -> T -> consumer -> C
 \
  -> map2 -> C                           ^
                                        B
```

* **Shared source:** selecting map1 also selects A, but not map2. Removing the original selection deletes map2's required input edge, leaving a dangling `IN_A` connector.
* **Downstream join:** selecting producer includes T and consumer, but not the independent B input. Copies of consumer lose their B read; removing the original selection leaves B isolated.

Neither deleting isolated nodes nor disabling validation is a valid repair: B is a real input to the returned result. Replacing the helper with a complete weak-component traversal is also insufficient, because that would replicate unrelated sibling/downstream computations and can multiply downstream write-conflict-resolution (WCR) contributions.

## Repair

The transformation now:

1. Selects `state.scope_subgraph(map_entry)` rather than a weak component.
2. Snapshots scope-internal edges and entry/exit boundary edges before adding copies.
3. Replicates only nodes inside the selected scope.
4. Reconnects every boundary edge to its **original external endpoint**.
5. Eliminates the singleton map in each copy with the existing `TrivialMapElimination`.
6. Deletes only the original scope.

External AccessNodes preserve shared-source fan-out and producer/consumer joins. A downstream consumer is transformed independently, so its WCR contribution count remains N, not N squared. WCR functions and memlet metadata are copied without selecting an arithmetic reduction tree. There is no isolated-node cleanup and no validation bypass in the public regression runs.

Two adjacent correctness defects were also repaired:

* **Nondivisible steps:** DaCe map endpoints are inclusive. Adding an entire step to convert to a Python range can add an out-of-domain iteration. For example, inclusive `(0, 5, 2)` must produce `{0, 2, 4}`, not `{0, 2, 4, 6}`. Endpoint conversion now adds only the direction of iteration; supported maps have positive steps.
* **Temporary-name collisions:** replicated scope-local descriptors use `find_new_name=True` and a per-iteration name map. An unrelated existing descriptor named `tmp_0` must not become the temporary for iteration zero.

## Explicit applicability boundary

The transformation rejects unsupported cases rather than changing their meaning:

* nonconstant, noninteger, nonpositive-step, or empty ranges;
* non-top-level maps (nested maps become eligible after their parent is unrolled);
* GPU schedules or dynamic map input connectors;
* scope-local Stream, View, or Reference replication, whose lifetime/alias rules need separate handling.

Direct `apply` checks applicability as well. This does not implement sequential-loop unrolling or silently authorize reordered reductions.

## Regression evidence

The standalone generic test file is:

```
tests/transformations/map_unroll_scope_boundary_test.py
```

It contains 11 tests covering shared-source fan-out, a downstream join's independent input, exact downstream WCR contribution count, preservation of a sibling direct copy, scope-local temporary independence, distinct nested SDFGs with correct parents, nested map coordinate coverage, temporary-name collision avoidance, nondivisible positive steps, empty/GPU rejection without changes, and dynamic-input rejection without changes.

Combined with the 12 existing `TrivialMapElimination` tests, the following run passed **23/23 tests** after the final patch:

```sh
python -m pytest tests/transformations/map_unroll_scope_boundary_test.py \
    tests/trivial_map_elimination_test.py -q
```

All new tests use native SDFG validation and exact structural/indexed assertions. They require no external benchmark checkout, private source, compiler, or machine-specific path.

An independent validation-only regression run also exercised seven graphs. Five were previously captured native graphs; their private definitions are not part of these tests or this document.

| Case | MapUnroll applications | Final validation |
|---|---:|---|
| Generic shared-source example | 2 | pass |
| Generic downstream-join example | 2 | pass |
| Elementwise activation graph | 9 | pass |
| Elementwise backward graph | 5 | pass |
| Gated backward/reduction graph | 6 | pass |
| Transpose/projection graph | 4 | pass |
| Previously working matrix projection | 3 | pass |

The original unmodified transformation reproduced dangling input connectors or isolated required inputs in the affected cases. All seven patched runs used `validate=True, validate_all=True`; no hand-edited or sanitized failure graph was substituted.

## Numerical execution limitation

The original native test `tests/transformations/map_unroll_test.py` reached its expected four unroll transformations and generated C++ successfully. In the verification environment, native compilation then failed during compiler configuration: the selected NMake generator did not accept a platform argument, and a suitable configured C/C++ compiler was unavailable. Therefore **compiled CPU numerical execution is not claimed** by this report. No compiler or dependency was installed to hide that gap.

On an environment with a configured native compiler, run:

```sh
python -m pytest tests/transformations/map_unroll_test.py -q
```

An unrelated legacy Windows `pyreadline` installation can fail during pytest's capture-plugin startup (`collections.Callable` is absent). In that situation, `-p no:capture -p no:cacheprovider` lets these graph tests run without changing installed packages. This option affects pytest output capture, not SDFG validation.
