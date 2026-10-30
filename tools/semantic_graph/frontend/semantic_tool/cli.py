"""CLI: python -m semantic_tool.cli <task-dir> --trusted-source --output result.json."""
from pathlib import Path
import argparse,json
from .task_io import load_task
from .capture import capture
from .semantics import lower_capture
from .core import validate_graph
from .scalarize import instantiate

def main():
    p=argparse.ArgumentParser(description='Construct typed reduction-aware semantic graph from an inspected original PyTorch task.')
    p.add_argument('task_dir');p.add_argument('--output',required=True);p.add_argument('--workload-index',type=int,default=0)
    p.add_argument('--probe-cap',type=int,default=None,help='Shrink shape axes explicitly; otherwise use real workload dimensions.')
    p.add_argument('--trusted-source',action='store_true',help='Acknowledge inspected Python source and dependencies; tracing executes Python, not a sandbox.')
    p.add_argument('--scalar-output');p.add_argument('--max-nodes',type=int,default=20000)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():p.error('Output already exists; choose a new path to preserve prior graph evidence')
    scalar=Path(args.scalar_output) if args.scalar_output else None
    if scalar and scalar.exists():p.error('Scalar output already exists')
    if not args.trusted_source:p.error('Inspect reference.py and its dependencies, then explicitly use --trusted-source')
    task=load_task(args.task_dir,args.workload_index,probe=args.probe_cap is not None,probe_cap=args.probe_cap or 4,device='meta',trusted=True)
    captured=capture(task.function,task.example_args,task.input_names)
    validate_graph(captured)
    graph=lower_capture(captured);validate_graph(graph);graph['metadata']['task']=task.metadata
    expanded=instantiate(graph,max_nodes=args.max_nodes) if scalar else None
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(graph,indent=2,ensure_ascii=False),encoding='utf-8')
    if scalar:
        scalar.parent.mkdir(parents=True,exist_ok=True);scalar.write_text(json.dumps(expanded,indent=2),encoding='utf-8')
    print(json.dumps({'semantic_graph':str(out),'nodes':len(graph['nodes']),'shape_scope':task.metadata['shape_scope'],
      'scalar_graph':str(scalar) if scalar else None,'source':'original trusted PyTorch source; no task-specific adapter','gpu_execution':False},ensure_ascii=False))
if __name__=='__main__':main()
