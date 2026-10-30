"""Normalize explicit composite/multi-output ATen operations into single-output rules.
Only lossless layout partition operations here; no arithmetic reassociation.
"""
from copy import deepcopy
from .core import UnsupportedOperator

def normalize_capture(graph):
    g=deepcopy(graph);out=[];metas={m['id']:m for m in g.get('inputs',[])}
    metas.update({k:dict(v,id=k) for k,v in g.get('constants',{}).items()})
    for n in g.get('nodes',[]):
        target=n.get('target');args=n.get('args',[]);kw=n.get('kwargs',{})
        if target in ('aten.split.Tensor','aten.split_with_sizes.default','aten.unbind.int'):
            if not args or not isinstance(args[0],dict) or set(args[0])!={'ref'}:raise UnsupportedOperator(n['id'],target,'partition input must be reference')
            m=metas[args[0]['ref']];shape=m['shape'];rank=len(shape)
            dim=kw.get('dim',args[1] if target=='aten.unbind.int' and len(args)>1 else args[2] if len(args)>2 else 0)
            if type(dim) is not int or not -rank<=dim<rank:raise UnsupportedOperator(n['id'],target,'invalid partition dimension')
            dim%=rank
            if target=='aten.unbind.int':sizes=[1]*shape[dim]
            elif target=='aten.split_with_sizes.default':sizes=kw.get('split_sizes',args[1])
            else:
                size=kw.get('split_size',args[1])
                if type(size) is not int or size<=0:raise UnsupportedOperator(n['id'],target,'invalid split size')
                sizes=[min(size,shape[dim]-i) for i in range(0,shape[dim],size)] or [0]
            if any(type(x) is not int or x<0 for x in sizes) or sum(sizes)!=shape[dim] or len(sizes)!=len(n['outputs']):raise UnsupportedOperator(n['id'],target,'partition/output shape mismatch')
            start=0
            for i,(size,desc) in enumerate(zip(sizes,n['outputs'])):
                expected=list(shape)
                if target=='aten.unbind.int':expected.pop(dim)
                else:expected[dim]=size
                if desc['shape']!=expected:raise UnsupportedOperator(n['id'],target,'incorrect partition output shape')
                replacement={'id':n['id']+'__part_'+str(i),'target':'aten.select.int' if target=='aten.unbind.int' else 'aten.slice.Tensor',
                    'args':[args[0],dim,i] if target=='aten.unbind.int' else [args[0],dim,start,start+size,1],
                    'kwargs':{},'outputs':[desc],'normalization_origin':{'id':n['id'],'target':target,'output_index':i,'rule':'exact-partition-index-map'}}
                out.append(replacement);metas[desc['id']]=desc;start+=size
        else:
            out.append(n)
            for m in n.get('outputs',[]):metas[m['id']]=m
    g['nodes']=out;g.setdefault('metadata',{})['capture_normalization']='exact single-output partition views only'
    return g
