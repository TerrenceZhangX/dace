# Optional PyTorch semantic frontend

The `semantic_tool` package is an optional producer for the sibling DaCe semantic JSON bridge. It is not imported by core DaCe and does not add PyTorch to DaCe's mandatory dependencies.

## Run

Use a Python environment with PyTorch and put this directory on `PYTHONPATH`:

```sh
export PYTHONPATH="$PWD/tools/semantic_graph/frontend:$PYTHONPATH"
python -m unittest discover -s tools/semantic_graph/frontend/semantic_tool/tests -v
python -m semantic_tool.cli /path/to/inspected/task --trusted-source --output semantic.json
python -m tools.semantic_graph.dace_bridge semantic.json --output sdfg-result --finite-only
```

On PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\tools\semantic_graph\frontend"
python -m unittest discover -s tools/semantic_graph/frontend/semantic_tool/tests -v
```

The frontend may run in a PyTorch environment separate from the environment used for DaCe. JSON is the explicit interchange format; no import-time model execution or GPU execution is required.

For direct callables:

```python
from semantic_tool import build

def fn(x, bias):
    return (x + bias).sum(dim=-1)

graph = build(fn, (x, bias), input_names=['x', 'bias'], trusted=True)
```

`trusted=True` acknowledges inspected Python and dependencies. Fake tracing executes Python; it is not a security sandbox. Data-dependent control-flow limitations and unsupported operators are reported explicitly.

## Data and tests

No benchmark definitions, reference sources, captured graphs, model weights or datasets are distributed here. Tests use generic synthetic computations. Six optional integration tests require an externally supplied L1 directory and otherwise skip:

```sh
export SOL_EXECBENCH_L1_DIR=/path/to/authorized/SOL-ExecBench/L1
```

Obtain external data from the official publisher under its own license. Do not infer the dataset license from this repository's license.

The initial exported checkpoint discovers199 tests; without the external dataset,193 execute successfully and6 skip. Passing CPU interpreter or dependency tests is not native compiled DaCe/GPU correctness.

## Development policy

- Target-driven rules, never benchmark-ID formula replacements.
- Exact output domains and input-index maps; reduction groups and numerical contracts remain explicit.
- Unsupported/data-dependent cases are not opaque successful graph nodes.
- Native DaCe transformations and code generation remain the downstream implementation path.
- Keep structural validation, dependency validation, numeric interpretation, compiled execution and hardware performance separate in coverage reports.
