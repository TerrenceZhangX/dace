"""Instantiate exact scalar value dependences, keeping reduction trees unspecified.
This is NOT a minimum-depth primitive DAG: a group remains one semantic relation.
Only small concrete domains are instantiated; parameterized graph remains primary.
"""
from __future__ import annotations
import ast,itertools,math,operator
from .core import validate_graph,metadata_table,UnsupportedOperator,GraphValidationError,refs

class ExpansionLimit(ValueError):pass
_BIN={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod}
_CMP={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}
_FUN={'min':min,'max':max,'floor':math.floor,'ceil':math.ceil,'int':int,'abs':abs}

def index_eval(text,env):
    if isinstance(text,(int,float,bool)):return text
    node=ast.parse(str(text),mode='eval').body
    def visit(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float,bool)):return n.value
        if isinstance(n,ast.Name) and n.id in env:return env[n.id]
        if isinstance(n,ast.BinOp) and type(n.op) in _BIN:return _BIN[type(n.op)](visit(n.left),visit(n.right))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd,ast.Not)):
            v=visit(n.operand);return -v if isinstance(n.op,ast.USub) else +v if isinstance(n.op,ast.UAdd) else not v
        if isinstance(n,ast.BoolOp):
            values=(bool(visit(x)) for x in n.values);return all(values) if isinstance(n.op,ast.And) else any(values)
        if isinstance(n,ast.Compare):
            l=visit(n.left)
            for op,rhs in zip(n.ops,n.comparators):
                r=visit(rhs)
                if type(op) not in _CMP or not _CMP[type(op)](l,r):return False
                l=r
            return True
        if isinstance(n,ast.IfExp):return visit(n.body if visit(n.test) else n.orelse)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in _FUN and not n.keywords:return _FUN[n.func.id](*map(visit,n.args))
        raise GraphValidationError('Unsupported index expression: '+str(text))
    return visit(node)

def _coords(shape):return itertools.product(*(range(int(x)) for x in shape))
def value_key(value,coord):return value+'['+','.join(map(str,coord))+']'

def instantiate(graph,max_nodes=20000,max_edges=200000):
    if type(max_nodes) is not int or type(max_edges) is not int or max_nodes<=0 or max_edges<=0:raise ValueError('Expansion budgets must be positive integers')
    validate_graph(graph)
    if graph['nodes'] and all('target' in n for n in graph['nodes']):
        from .verify import verify_semantics
        verify_semantics(graph)
    table=metadata_table(graph)
    estimate=sum(math.prod(m.get('shape',[])) for m in table.values())
    if estimate>max_nodes:raise ExpansionLimit(f'{estimate} scalar values exceed limit {max_nodes}; use parameterized graph')
    nodes=[];edges=set();known=set();reductions=0
    def add(value,coord,kind,**extra):
        key=value_key(value,coord)
        if key in known:raise GraphValidationError('Duplicate scalar value '+key)
        known.add(key);nodes.append(dict(id=key,value=value,index=list(coord),kind=kind,**extra));return key
    for m in graph.get('inputs',[]):
        for c in _coords(m.get('shape',[])):add(m['id'],c,'input',dtype=m.get('dtype'))
    for key,m in graph.get('constants',{}).items():
        for c in _coords(m.get('shape',[])):add(key,c,'constant',dtype=m.get('dtype'))
    for node in graph['nodes']:
        sem=node['semantics'];reads=sem.get('reads',[]);reduction=sem.get('reduction')
        if len(node.get('outputs',[]))!=1:raise UnsupportedOperator(node['id'],node['op'],'scalarization requires one semantic output per node')
        out=node['outputs'][0];shape=out.get('shape',[])
        for coord in _coords(shape):
            env={f'i{i}':c for i,c in enumerate(coord)}
            domain=sem.get('domain',[])
            if len(domain)==len(coord):env.update({d['var']:c for d,c in zip(domain,coord)})
            dst=add(out['id'],coord,'reduction-group' if reduction else 'operation',op=node['op'],origin=node['id'],dtype=out.get('dtype'))
            if reduction:reductions+=1
            variables=(reduction or {}).get('variables',[])
            assignments=[{}]
            if variables:
                assignments=({v['var']:c for v,c in zip(variables,cs)} for cs in _coords([v['extent'] for v in variables]))
            for extra in assignments:
                local={**env,**extra}
                for read in reads:
                    guard=read.get('guard',read.get('condition'))
                    if guard is not None and not index_eval(guard,local):continue
                    value=read['value'];meta=table[value]
                    indices=read.get('indices')
                    if indices is None:raise UnsupportedOperator(node['id'],node['op'],'missing exact input index map')
                    idx=tuple(int(index_eval(i,local)) for i in indices)
                    dims=meta.get('shape',[])
                    if len(idx)!=len(dims) or any(v<0 or v>=s for v,s in zip(idx,dims)):
                        raise GraphValidationError(f'{node["id"]}: out of bounds {value}{idx} shape={dims}')
                    src=value_key(value,idx)
                    if src not in known:raise GraphValidationError(f'{node["id"]}: missing scalar producer {src}')
                    edges.add((src,dst))
                    if len(edges)>max_edges:raise ExpansionLimit(f'Edge budget {max_edges} exceeded')
    outputs=[]
    for value in refs(graph['outputs']):
        for c in _coords(table[value].get('shape',[])):outputs.append(value_key(value,c))
    may_dependence = any(n['semantics'].get('dependence_kind') == 'may-dependence' for n in graph['nodes'])
    return {'format':'scalar-dependence-v1','nodes':nodes,'edges':[list(e) for e in sorted(edges)],'outputs':outputs,
      'metadata':{'source_format':graph['format'],'reduction_groups':reductions,'tree_choice':'unspecified',
       'dependence_kind':'may-dependence' if may_dependence else 'exact-index-dependence',
       'scope':('conservative possible dependencies; data guards not resolved' if may_dependence else 'exact value/index dependences for selected operator decomposition') + '; not all-algorithm graph or optimal span proof',
       'node_count':len(nodes),'edge_count':len(edges)}}

def ancestors(expanded,output):
    pred={}
    for a,b in expanded['edges']:pred.setdefault(b,[]).append(a)
    seen=set();todo=[output]
    while todo:
        cur=todo.pop()
        for p in pred.get(cur,[]):
            if p not in seen:seen.add(p);todo.append(p)
    return seen
