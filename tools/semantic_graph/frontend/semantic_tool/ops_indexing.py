"""Exact repeat/roll/stack maps and checked floor/bitwise pointwise rules.

Drop-in extension: no source imports, task IDs, dynamic ATen lookup or schedules.
The CPU executor accepts already-resolved arguments and returns None for unrelated
operators. Recognized malformed or mutating overloads fail closed.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .core import UnsupportedOperator, refs

_ALLOWED = {'repeat': {'default'}, 'roll': {'default'}, 'stack': {'default'},
            'floor_divide': {'default', 'Scalar'},
            'bitwise_not': {'default'},
            'bitwise_and': {'Tensor', 'Scalar', 'Scalar_Tensor'},
            'bitwise_or': {'Tensor', 'Scalar', 'Scalar_Tensor'},
            'bitwise_xor': {'Tensor', 'Scalar', 'Scalar_Tensor'}}
_DTYPES = {'bool', 'uint8', 'int8', 'int16', 'int32', 'int64',
           'float16', 'bfloat16', 'float32', 'float64', 'complex64', 'complex128'}
_INT = {'bool', 'uint8', 'int8', 'int16', 'int32', 'int64'}
_ALIASES = {'float':'float32', 'double':'float64', 'half':'float16', 'long':'int64',
            'int':'int32', 'short':'int16', 'char':'int8', 'byte':'uint8'}


def _selection(node):
    target = node.get('target', '')
    match = re.fullmatch(r'(?:torch\.ops\.)?aten\.([A-Za-z_][A-Za-z_0-9]*)(?:\.([A-Za-z_][A-Za-z_0-9]*))?', target)
    if target:
        if not match or match[1] not in _ALLOWED:
            return None
        base, overload = match[1], match[2] or 'default'
        if overload not in _ALLOWED[base]:
            raise UnsupportedOperator(node.get('id', '?'), target, 'unsupported functional overload; no out/in-place execution')
        if node.get('op', base) != base:
            raise UnsupportedOperator(node.get('id', '?'), target, 'normalized op conflicts with target')
        return base
    return node.get('op') if node.get('op') in _ALLOWED else None


def _dtype(meta):
    name = str(meta.get('dtype', '')).removeprefix('torch.')
    name = _ALIASES.get(name, name)
    if name not in _DTYPES:
        raise ValueError(f'unsupported dtype {name!r}')
    return name


def _shape(meta):
    shape = meta.get('shape')
    if meta.get('kind', 'tensor') != 'tensor' or not isinstance(shape, (list, tuple)) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('concrete nonnegative tensor shape required')
    _dtype(meta)
    return list(shape)


def _bind(base, args, kwargs, attrs=None):
    names = {'repeat': ('self', 'repeats'), 'roll': ('self', 'shifts', 'dims'),
             'stack': ('tensors', 'dim'), 'bitwise_not': ('self',)}.get(base, ('self', 'other'))
    defaults = {'dim': 0, 'dims': []}
    attrs = attrs or {}
    if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict) or len(args) > len(names) or set(kwargs) - set(names):
        raise ValueError('unknown/excess arguments')
    result = {}
    for i, name in enumerate(names):
        if i < len(args) and name in kwargs:
            raise ValueError(f'duplicate argument {name}')
        if i < len(args):
            result[name] = args[i]
        elif name in kwargs:
            result[name] = kwargs[name]
        elif name in attrs:
            result[name] = attrs[name]
        elif name in defaults:
            result[name] = defaults[name]
        else:
            raise ValueError(f'missing argument {name}')
    return result


def _integers(value, name):
    if type(value) is int:
        return [value]
    if not isinstance(value, (list, tuple)) or any(type(v) is not int for v in value):
        raise ValueError(f'{name} requires concrete integers')
    return list(value)


def _axis(value, rank):
    if type(value) is not int or not -rank <= value < rank:
        raise ValueError(f'dimension {value!r} outside rank {rank}')
    return value % rank


def _roll_parameters(shape, shifts, dims):
    shifts, dims = _integers(shifts, 'shifts'), _integers(dims, 'dims')
    if not shifts:
        raise ValueError('roll shifts cannot be empty')
    if not dims:
        if len(shifts) != 1:
            raise ValueError('flattened roll requires exactly one shift')
        return shifts, [], True
    if len(shifts) != len(dims):
        raise ValueError('roll dims and shifts must have equal lengths')
    return shifts, [_axis(d, len(shape)) for d in dims], False


def _broadcast(shapes):
    output = [1] * max(map(len, shapes), default=0)
    for shape in shapes:
        for j, n in enumerate(reversed(shape)):
            k = len(output)-1-j
            if output[k] == 1:
                output[k] = n
            elif n not in (1, output[k]):
                raise ValueError('incompatible broadcast shapes')
    return output


def lower_indexing_extension(node, values):
    base = _selection(node)
    if base is None:
        return None
    try:
        return _lower(node, values, base)
    except UnsupportedOperator:
        raise
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
        raise UnsupportedOperator(node.get('id', '?'), node.get('target', base), str(exc)) from exc


def _lower(node, values, base):
    outputs = deepcopy(node.get('outputs', []))
    if len(outputs) != 1:
        raise ValueError('exactly one tensor output required')
    out = outputs[0]
    shape, dtype = _shape(out), _dtype(out)
    args, kwargs = deepcopy(node.get('args', [])), deepcopy(node.get('kwargs', {}))
    params = _bind(base, args, kwargs)
    referenced = list(refs([args, kwargs]))
    for ref in referenced:
        if ref not in values:
            raise ValueError(f'undefined SSA reference {ref}')
    sem = {'domain': [{'var':f'i{j}', 'extent':n} for j,n in enumerate(shape)],
           'reads': [], 'reduction': None,
           'arithmetic': {'input_dtypes': [values[r].get('dtype') for r in referenced],
                          'output_dtype': dtype, 'rounding': 'none',
                          'reassociation': 'not-authorized-by-default'},
           'order': 'data-only', 'notes': []}
    attrs = {}

    def tensor(value):
        if not isinstance(value, dict) or set(value) != {'ref'} or not isinstance(value['ref'], str):
            raise ValueError('tensor operand requires an SSA reference')
        ref = value['ref']; meta = values[ref]
        source = _shape(meta)
        if out.get('device') is not None and meta.get('device') is not None and meta['device'] != out['device']:
            raise ValueError('indexing operation cannot change device')
        return ref, source, meta

    def literal(value):
        if isinstance(value, dict) and set(value) == {'ref'}:
            meta = values[value['ref']]
            if meta.get('kind') != 'scalar' or 'specialized_value' not in meta:
                raise ValueError('structural parameter requires specialized scalar')
            value = meta['specialized_value']
        if isinstance(value, (list, tuple)):
            return [literal(v) for v in value]
        return value

    def add_read(ref, indices, guard=None, role='pointwise'):
        read = {'value':ref, 'indices':indices, 'role':role}
        if guard is not None:
            read['guard'] = guard
        sem['reads'].append(read)

    def expect(expected):
        if shape != expected:
            raise ValueError(f'captured output shape {shape} disagrees with {expected}')

    if base in ('repeat', 'roll'):
        ref, source, meta = tensor(params['self'])
        if _dtype(meta) != dtype:
            raise ValueError('layout operation must preserve dtype')
        if base == 'repeat':
            repeats = _integers(literal(params['repeats']), 'repeats')
            if len(repeats) < len(source) or any(r < 0 for r in repeats):
                raise ValueError('repeat factors must be nonnegative and cover input rank')
            padded = [1] * (len(repeats)-len(source)) + source
            expect([n*r for n,r in zip(padded,repeats)])
            offset = len(shape)-len(source)
            indices = ['0' if n <= 1 else f'(i{j+offset} % {n})' for j,n in enumerate(source)]
            attrs.update(repeats=repeats)
        else:
            expect(source)
            shifts, dims, flatten = _roll_parameters(source, literal(params['shifts']), literal(params['dims']))
            if flatten:
                count = math.prod(source)
                flat = '(' + ' + '.join(f'(i{j} * {math.prod(source[j+1:])})' for j in range(len(source))) + ')' if source else '0'
                moved = f'(({flat} - ({shifts[0]})) % {max(1,count)})'
                indices = ['0' if n <= 1 else f'(({moved} // {max(1, math.prod(source[j+1:]))}) % {n})' for j,n in enumerate(source)]
            else:
                combined = [sum(s for s,d in zip(shifts,dims) if d == j) for j in range(len(source))]
                indices = ['0' if n <= 1 else f'((i{j} - ({s})) % {n})' for j,(n,s) in enumerate(zip(source,combined))]
            attrs.update(shifts=shifts, dims=dims, flattened=flatten)
        add_read(ref, indices, role='layout-select')
    elif base == 'stack':
        operands = params['tensors']
        if not isinstance(operands, (list, tuple)) or not operands:
            raise ValueError('stack requires a nonempty tensor list')
        sources = [tensor(v) for v in operands]
        source = sources[0][1]
        if any(s != source for _,s,_ in sources):
            raise ValueError('stack tensors must have identical shapes')
        if any(_dtype(m) != dtype for _,_,m in sources):
            raise ValueError('stack extension requires same dtype; mixed promotion needs explicit casts')
        dim = _axis(literal(params['dim']), len(source)+1)
        expected = source[:dim] + [len(sources)] + source[dim:]
        expect(expected)
        indices = [f'i{j if j < dim else j+1}' for j in range(len(source))]
        for k,(ref,_,_) in enumerate(sources):
            add_read(ref, list(indices), guard=f'(i{dim} == {k})', role='layout-select')
        attrs.update(dim=dim)
        sem['notes'].append('Exactly one guarded source read per output coordinate; inserted stack dimension selects a tensor, not an extra arithmetic dependency.')
    else:
        import torch
        operands = [params['self']] + ([] if base == 'bitwise_not' else [params['other']])
        metas = []
        promotion_operands = []
        operand_shapes = []
        for value in operands:
            if isinstance(value, dict) and set(value) == {'ref'} and values[value['ref']].get('kind', 'tensor') == 'tensor':
                ref, source, meta = tensor(value)
                dt = _dtype(meta)
                promotion_operands.append(torch.empty(() if not source else (1,), dtype=getattr(torch,dt), device='meta'))
                operand_shapes.append(source)
                metas.append((ref, source, dt))
            else:
                scalar = literal(value)
                if type(scalar) not in (bool, int, float):
                    raise ValueError('pointwise scalar must be bool/int/float')
                promotion_operands.append(scalar)
                operand_shapes.append([])
                metas.append((value.get('ref') if isinstance(value,dict) else None, [], type(scalar).__name__))
        if not any(isinstance(v,torch.Tensor) for v in promotion_operands):
            raise ValueError('at least one tensor operand required')
        expect(_broadcast(operand_shapes))
        promoted = promotion_operands[0].dtype if len(promotion_operands) == 1 else torch.result_type(*promotion_operands)
        promoted_name = str(promoted).removeprefix('torch.')
        if dtype != promoted_name:
            raise ValueError(f'output dtype {dtype} disagrees with ATen promotion {promoted_name}')
        if base.startswith('bitwise_'):
            if dtype not in _INT or any(dt not in _INT and dt not in ('int','bool') for _,_,dt in metas):
                raise ValueError('bitwise operations require boolean/integer operands')
            sem['arithmetic'].update(kind='logical-boolean' if dtype == 'bool' else 'fixed-width-bitwise',
                                     computation_dtype=dtype, promotion='ATen-result-type', rounding='none')
        else:
            if dtype == 'bool' or dtype.startswith('complex'):
                raise ValueError('floor_divide requires real non-bool arithmetic')
            sem['arithmetic'].update(kind='floor-division', promotion='ATen-result-type',
                                     computation_dtype='float32' if dtype in ('float16','bfloat16') else dtype,
                                     rounding='floor-quotient-then-output-dtype',
                                     exceptional_values='ATen CPU primitive: integer division by zero raises; floating nonfinite semantics retained')
        for ref,source,_ in metas:
            if ref is not None:
                offset = len(shape)-len(source)
                indices = ['0' if n == 1 else f'i{j+offset}' for j,n in enumerate(source)]
                add_read(ref,indices)
    if base in ('repeat','roll','stack'):
        sem['layout'] = {'kind':base, 'alias':'none', 'requires_materialization':False,
                         'output_strides':deepcopy(out.get('strides')),
                         'storage_semantics':'functional coordinate selection; allocation and schedule not prescribed'}
    result = deepcopy(node)
    result.update(op=base,args=args,kwargs=kwargs,outputs=outputs,attributes=attrs,semantics=sem)
    return result


def execute_indexing_extension(node, args, kwargs, attrs):
    """Run a closed explicit CPU primitive table; None means unrelated operator."""
    base = _selection(node)
    if base is None:
        return None
    import torch
    params = _bind(base,args,kwargs,attrs)
    def tensor(value):
        if not isinstance(value,torch.Tensor) or value.device.type != 'cpu' or value.layout != torch.strided:
            raise ValueError('indexing executor requires strided CPU tensor')
        return value
    if base == 'stack':
        xs = [tensor(v) for v in params['tensors']]
        if not xs or any(v.dtype != xs[0].dtype or v.shape != xs[0].shape for v in xs):
            raise ValueError('stack requires nonempty equal-shape same-dtype tensors')
        return torch.stack(xs,dim=_axis(params['dim'],xs[0].ndim+1))
    if base == 'repeat':
        x = tensor(params['self']); repeats = _integers(params['repeats'],'repeats')
        if len(repeats)<x.ndim or any(r<0 for r in repeats):
            raise ValueError('invalid repeats')
        return x.repeat(*repeats)
    if base == 'roll':
        x = tensor(params['self'])
        shifts,dims,flatten = _roll_parameters(list(x.shape),params['shifts'],params['dims'])
        return torch.roll(x,shifts=tuple(shifts),dims=None if flatten else tuple(dims))
    operands = [params['self']] + ([] if base == 'bitwise_not' else [params['other']])
    if not any(isinstance(v,torch.Tensor) for v in operands):
        raise ValueError('at least one CPU tensor required')
    for v in operands:
        if isinstance(v,torch.Tensor):
            tensor(v)
            if base.startswith('bitwise_') and str(v.dtype).removeprefix('torch.') not in _INT:
                raise ValueError('bitwise operation requires integer/bool dtype')
        elif type(v) not in ((bool,int) if base.startswith('bitwise_') else (bool,int,float)):
            raise ValueError('invalid scalar operand')
    if base == 'floor_divide':
        return torch.floor_divide(*operands)
    if base == 'bitwise_not':
        return torch.bitwise_not(*operands)
    if base == 'bitwise_and':
        return torch.bitwise_and(*operands)
    if base == 'bitwise_or':
        return torch.bitwise_or(*operands)
    return torch.bitwise_xor(*operands)


__all__ = ['lower_indexing_extension','execute_indexing_extension']
