"""Exact, target-driven cat and constant padding semantics; no task identifiers.

``lower_layout_extension(node, values)`` returns a fresh semantic node for the
supported ATen targets, None for unrelated targets, and raises UnsupportedOperator
for malformed/unsupported variants. It never executes ATen or imports torch.

Padding literal regions are represented separately from tensor reads. Consumers
must interpret ``constant_regions`` or the normalized operator; padding does NOT
read a convenient input element. All maps are logical, with storage metadata
retained separately. Mixed-dtype cat is conservatively rejected, not silently cast.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .core import UnsupportedOperator, refs

_ALIASES = {'float':'float32','double':'float64','half':'float16','long':'int64',
            'int':'int32','short':'int16','char':'int8','byte':'uint8',
            'cfloat':'complex64','cdouble':'complex128'}
_DTYPES = {'bool','uint8','int8','int16','int32','int64','float16','bfloat16',
           'float32','float64','complex64','complex128'}


def _dtype(meta):
    raw = str(meta.get('dtype', '')).removeprefix('torch.')
    dtype = _ALIASES.get(raw, raw)
    if dtype not in _DTYPES:
        raise ValueError(f'unsupported or missing dtype {raw!r}')
    return dtype


def _shape(meta):
    if meta.get('kind','tensor') != 'tensor':
        raise ValueError('tensor metadata required')
    shape = meta.get('shape')
    if not isinstance(shape,(tuple,list)) or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError('concrete nonnegative tensor shape required')
    strides = meta.get('strides')
    if strides is not None and (not isinstance(strides,(tuple,list)) or len(strides) != len(shape)
                               or any(type(s) is not int or s < 0 for s in strides)):
        raise ValueError('strides must match tensor rank and be nonnegative integers')
    offset = meta.get('storage_offset',0)
    if type(offset) is not int or offset < 0:
        raise ValueError('storage_offset must be a nonnegative integer')
    layout = meta.get('layout')
    if layout not in (None,'strided','torch.strided'):
        raise ValueError('only strided tensor layouts are supported')
    _dtype(meta)
    return list(shape)


def _address(meta, indices):
    strides = meta.get('strides')
    if strides is None:
        return None
    terms = [str(meta.get('storage_offset',0))]
    terms.extend(f'(({index}) * {stride})' for index,stride in zip(indices,strides))
    return '('+' + '.join(terms)+')'


def lower_layout_extension(node, values):
    """Lower cat.default / constant_pad_nd.default or reject without approximation."""
    target = node.get('target','')
    match = re.fullmatch(r'(?:torch\.ops\.)?aten\.(cat|constant_pad_nd)(?:\.([A-Za-z_][A-Za-z_0-9]*))?',target)
    if not match:
        return None
    try:
        base,overload = match.group(1),match.group(2) or 'default'
        if overload != 'default':
            raise ValueError('only default functional overload is supported; no out mutation')
        return _lower(node,values,base)
    except UnsupportedOperator:
        raise
    except (ValueError,TypeError,KeyError,IndexError) as exc:
        raise UnsupportedOperator(node.get('id','<missing>'),target,str(exc)) from exc


def _lower(node,values,base):
    outputs = deepcopy(node.get('outputs',[]))
    if len(outputs) != 1:
        raise ValueError('exactly one tensor output required')
    out = outputs[0]
    shape = _shape(out)
    dtype = _dtype(out)
    args,kwargs = deepcopy(node.get('args',[])),deepcopy(node.get('kwargs',{}))
    if not isinstance(args,(list,tuple)) or not isinstance(kwargs,dict):
        raise ValueError('args/kwargs must be list/dict')
    names = ('tensors','dim') if base == 'cat' else ('self','pad','value')
    if len(args) > len(names) or set(kwargs)-set(names):
        raise ValueError('unknown or excess arguments')
    for i in range(len(args)):
        if names[i] in kwargs:
            raise ValueError(f'duplicate argument {names[i]}')
    referenced = list(refs([args,kwargs]))
    for ref in referenced:
        if ref not in values:
            raise ValueError(f'undefined SSA reference {ref!r}')
    def get(i,default=None):
        return kwargs.get(names[i],args[i] if i < len(args) else default)
    def literal(value):
        if isinstance(value,dict) and set(value)=={'ref'}:
            meta = values[value['ref']]
            if meta.get('kind') != 'scalar' or 'specialized_value' not in meta:
                raise ValueError('structural/fill arguments require specialized Python scalar')
            value = meta['specialized_value']
        # Strict JSON supports IEEE nonfinite literals only via this closed tag.
        # Decode the reviewed data representation, never arbitrary objects/code.
        if isinstance(value,dict) and set(value)=={'float'}:
            tag = value['float']
            if type(tag) is not str or tag not in ('inf','-inf','nan'):
                raise ValueError('invalid nonfinite float literal tag')
            return {'inf':math.inf,'-inf':-math.inf,'nan':math.nan}[tag]
        return value
    def tensor(value):
        if not isinstance(value,dict) or set(value)!={'ref'} or not isinstance(value['ref'],str):
            raise ValueError('tensor operand must be an SSA ref')
        ref = value['ref']; meta = values[ref]
        source = _shape(meta)
        if _dtype(meta) != dtype:
            raise ValueError('same dtype required; mixed dtype promotion needs an explicit semantic rule')
        if meta.get('device') is not None and out.get('device') is not None and meta['device'] != out['device']:
            raise ValueError('layout extension cannot change tensor device')
        return ref,source,meta
    sem = {'domain':[{'var':f'i{j}','extent':n} for j,n in enumerate(shape)],
           'reads':[],'reduction':None,
           'arithmetic':{'input_dtypes':[values[r].get('dtype') if values[r].get('kind')=='scalar' else _dtype(values[r]) for r in referenced],
                         'output_dtype':dtype,'rounding':'none','reassociation':'not-authorized-by-default'},
           'order':'data-only','notes':[]}
    input_layouts = []
    def add_read(ref,source,meta,indices,guard):
        sem['reads'].append({'value':ref,'indices':indices,'role':'layout-select','guard':guard})
        input_layouts.append({'value':ref,'shape':source,'strides':deepcopy(meta.get('strides')),
                              'storage_offset':meta.get('storage_offset',0),'guard':guard,
                              'logical_index_map':indices,'storage_index_map':_address(meta,indices)})
    attrs = {}
    if base == 'cat':
        operands = get(0)
        if not isinstance(operands,(list,tuple)) or not operands:
            raise ValueError('cat requires a nonempty sequence of tensors')
        sources = [tensor(v) for v in operands]
        # ATen permits one-dimensional empty tensors as rank-independent skips.
        active = [s for s in sources if s[1] != [0]]
        template = active[0][1] if active else [0]
        rank = len(template)
        if rank == 0:
            raise ValueError('cat cannot concatenate rank-zero tensors')
        dim = literal(get(1,0))
        if type(dim) is not int or not -rank <= dim < rank:
            raise ValueError('cat dim outside tensor rank')
        dim %= rank
        expected = list(template); expected[dim] = 0
        offset = 0; segments = []; skipped = []
        for ref,source,meta in sources:
            if source == [0]:
                skipped.append(ref)
                continue
            if len(source)!=rank or any(source[j]!=template[j] for j in range(rank) if j!=dim):
                raise ValueError('cat input shapes differ outside concatenation dimension')
            end = offset + source[dim]
            indices = [f'(i{j} - {offset})' if j==dim else f'i{j}' for j in range(rank)]
            guard = f'(i{dim} >= {offset} and i{dim} < {end})'
            add_read(ref,source,meta,indices,guard)
            segments.append({'value':ref,'start':offset,'end':end})
            expected[dim] += source[dim]; offset = end
        if shape != expected:
            raise ValueError(f'captured output shape {shape} != cat shape {expected}')
        attrs.update(dim=dim,segments=segments,skipped_empty_1d=skipped,shape=shape)
        if skipped:
            sem['metadata_reads']=[{'value':r,'fields':['shape','dtype','device']} for r in skipped]
        sem['notes'].append('Exactly one segment read is active per output coordinate; repeated input references retain separate coordinate segments. Empty domains have no value reads.')
    else:
        ref,source,meta = tensor(get(0))
        raw_pad = get(1)
        if not isinstance(raw_pad,(tuple,list)) or len(raw_pad)%2 or len(raw_pad)>2*len(source):
            raise ValueError('pad must have even length no greater than twice input rank')
        pad = [literal(p) for p in raw_pad]
        if any(type(p) is not int for p in pad):
            raise ValueError('padding must be concrete integers')
        fill = literal(get(2,0))
        if type(fill) not in (bool,int,float):
            raise ValueError('padding fill requires a JSON numeric literal or specialized scalar')
        # Conservative checked conversion: never invent integer wraparound.
        if any(p > 0 for p in pad) and dtype in {'uint8','int8','int16','int32','int64'}:
            bits = int(re.search(r'\d+',dtype).group())
            lo,hi = (0,2**bits-1) if dtype=='uint8' else (-2**(bits-1),2**(bits-1)-1)
            if (isinstance(fill,float) and not math.isfinite(fill)) or not lo <= fill <= hi:
                raise ValueError('padding scalar outside checked integer dtype range')
        before,after = [0]*len(source),[0]*len(source)
        for pair in range(len(pad)//2):
            axis = len(source)-1-pair
            before[axis],after[axis] = pad[2*pair:2*pair+2]
        expected = [n+l+r for n,l,r in zip(source,before,after)]
        if any(n+min(l,0)+min(r,0)<0 for n,l,r in zip(source,before,after)):
            raise ValueError('negative padding crops beyond source extent')
        if any(n<0 for n in expected) or shape != expected:
            raise ValueError(f'captured output shape {shape} != padded shape {expected}')
        indices = [f'(i{j} - ({p}))' for j,p in enumerate(before)]
        inside = '('+' and '.join(f'({index} >= 0 and {index} < {n})' for index,n in zip(indices,source))+')' if source else 'True'
        add_read(ref,source,meta,indices,inside)
        sem['constant_regions']=[{'guard':f'not ({inside})','value':fill,'dtype':dtype,
                                  'conversion':'ATen-scalar-to-output-dtype'}]
        sem['arithmetic']['literal_conversion']={'value':fill,'to':dtype,'rule':'ATen-scalar-to-output-dtype'}
        attrs.update(pad=pad,pad_before=before,pad_after=after,fill_value=fill,shape=shape)
        sem['notes'].append('Inside region copies exactly one source element; outside region is a typed scalar literal with zero input value dependencies. Negative padding crops logical coordinates.')
    sem['layout']={'kind':'concatenate-copy' if base=='cat' else 'constant-pad-copy',
                   'inputs':input_layouts,'output_strides':deepcopy(out.get('strides')),
                   'output_storage_offset':out.get('storage_offset',0),
                   'alias':'none','requires_materialization':False,
                   'storage_semantics':'fresh-ATen-output; schedule/allocation-not-prescribed'}
    result = deepcopy(node)
    result.update(op=base,args=args,kwargs=kwargs,outputs=outputs,attributes=attrs,semantics=sem)
    return result
