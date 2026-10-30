"""Shared graph validation: structural checks are necessary, not semantic proof."""
from __future__ import annotations
import hashlib,json,math

class UnsupportedOperator(NotImplementedError):
    def __init__(self,node_id,target,reason='unsupported semantic operator'):
        self.node_id,self.target,self.reason=node_id,target,reason
        super().__init__(f'{node_id}: {target}: {reason}')

class GraphValidationError(ValueError): pass

def refs(value):
    if isinstance(value,dict):
        if set(value)=={'ref'} and isinstance(value['ref'],str):yield value['ref']
        else:
            for v in value.values():yield from refs(v)
    elif isinstance(value,(tuple,list)):
        for v in value:yield from refs(v)

def metadata_table(graph):
    result={}
    for item in graph.get('inputs',[]):result[item['id']]=item
    for key,item in graph.get('constants',{}).items():result[key]=dict(item,id=key)
    for node in graph.get('nodes',[]):
        for out in node.get('outputs',[]):result[out['id']]=out
    return result

def validate_graph(graph):
    if graph.get('format') not in ('aten-capture-v1','semantic-graph-v1'):
        raise GraphValidationError('Unknown graph format')
    known=set();nids=set()
    def add(item):
        i=item['id']
        if i in known:raise GraphValidationError(f'Duplicate value {i}')
        if item.get('kind','tensor')=='tensor':
            if not isinstance(item.get('shape'),list) or any(not isinstance(x,int) or x<0 for x in item['shape']):
                raise GraphValidationError(f'{i}: shape must be concrete nonnegative integers')
            if not item.get('dtype'):raise GraphValidationError(f'{i}: missing dtype')
        known.add(i)
    for x in graph.get('inputs',[]):add(x)
    for k,v in graph.get('constants',{}).items():add(dict(v,id=k))
    for n in graph.get('nodes',[]):
        if n['id'] in nids:raise GraphValidationError('Duplicate node '+n['id'])
        nids.add(n['id'])
        for r in refs([n.get('args',[]),n.get('kwargs',{})]):
            if r not in known:raise GraphValidationError(f'{n["id"]}: unavailable dependency {r}')
        if graph['format']=='semantic-graph-v1':
            if not n.get('op') or not isinstance(n.get('semantics'),dict):raise GraphValidationError(f'{n["id"]}: missing semantics')
            if n['semantics'].get('order')!='data-only':raise GraphValidationError(f'{n["id"]}: unexpected schedule ordering')
            for r in n['semantics'].get('reads',[]):
                if r.get('value') not in known:raise GraphValidationError(f'{n["id"]}: unknown read value')
        for out in n.get('outputs',[]):add(out)
    for r in refs(graph.get('outputs')):
        if r not in known:raise GraphValidationError('Unknown graph output '+r)
    return {'nodes':len(nids),'values':len(known),'status':'valid-structure'}

def graph_hash(graph):
    return hashlib.sha256(json.dumps(graph,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def build(function,example_args,input_names=None,*,trusted=False):
    from .capture import capture
    from .semantics import lower_capture
    captured=capture(function,example_args,input_names=input_names,trusted=trusted)
    validate_graph(captured)
    graph=lower_capture(captured)
    validate_graph(graph)
    from .verify import verify_semantics
    verify_semantics(graph)
    return graph
