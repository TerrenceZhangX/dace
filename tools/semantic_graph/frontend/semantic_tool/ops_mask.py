"""Closed functional mask, remainder and prefix-sum semantic extensions.

Index guards never inspect tensor values. masked_fill instead records explicit
machine-readable data_guards and a conservative may-dependence relation. An
index-only consumer MUST NOT call that relation an exact selected value cone.
The CPU helper chooses reviewed primitives, not arbitrary targets or task code.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .core import UnsupportedOperator, refs
from .ops_layout import _shape, _dtype, _address

_TARGETS = {
    'triu': {'default'}, 'tril': {'default'}, 'masked_fill': {'Scalar'},
    'fill': {'Scalar'}, 'remainder': {'Scalar', 'Tensor'}, 'cumsum': {'default'},
}
_FLOATS = {'float16', 'bfloat16', 'float32', 'float64'}
_INTS = {'uint8', 'int8', 'int16', 'int32', 'int64'}


def _selection(node, normalized=False):
    target = node.get('target')
    if isinstance(target, str):
        match = re.fullmatch(r'(?:torch\.ops\.)?aten\.([A-Za-z_][A-Za-z_0-9]*)(?:\.([A-Za-z_][A-Za-z_0-9]*))?', target)
        if match is None or match[1] not in _TARGETS:
            return None
        op, overload = match[1], match[2] or 'default'
    elif normalized and node.get('op') in _TARGETS:
        op = node['op']
        overload = node.get('attributes', {}).get('overload', 'Scalar' if op in {'fill', 'masked_fill'} else 'default')
    else:
        return None
    if overload not in _TARGETS[op]:
        raise ValueError(f'{op}.{overload}: only reviewed functional overloads are supported')
    return op, overload


def _bind(op, args, kwargs, attrs=None):
    signatures = {'triu': ('self', 'diagonal'), 'tril': ('self', 'diagonal'),
                  'fill': ('self', 'value'), 'masked_fill': ('self', 'mask', 'value'),
                  'remainder': ('self', 'other'), 'cumsum': ('self', 'dim', 'dtype')}
    names = signatures[op]
    defaults = {'diagonal': 0, 'dtype': None}
    positional = 2 if op == 'cumsum' else len(names)
    if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict):
        raise ValueError('args/kwargs must be a sequence/object')
    if len(args) > positional or set(kwargs) - set(names):
        raise ValueError('unknown, excess or keyword-only positional argument')
    result = dict(zip(names, args))
    for key, value in kwargs.items():
        if key in result:
            raise ValueError(f'duplicate argument {key}')
        result[key] = value
    for key in names:
        if key not in result:
            if key in (attrs or {}) and key != 'self':
                result[key] = attrs[key]
            elif key in defaults:
                result[key] = defaults[key]
            else:
                raise ValueError(f'missing argument {key}')
    return result


def _number(value):
    # This is a closed data tag, not arbitrary object/string deserialization.
    if isinstance(value, dict):
        if set(value) != {'float'} or type(value['float']) is not str or value['float'] not in ('inf', '-inf', 'nan'):
            raise ValueError('invalid scalar or nonfinite float tag')
        value = {'inf': math.inf, '-inf': -math.inf, 'nan': math.nan}[value['float']]
    if type(value) not in (bool, int, float):
        raise ValueError('a concrete real/bool scalar is required')
    return value


def _axis(value, rank):
    if type(value) is not int or not -max(rank, 1) <= value < max(rank, 1):
        raise ValueError('cumsum dimension outside tensor rank')
    return value % max(rank, 1)


def _promote(a, b):
    if a == b:
        return a
    if a == 'bool':
        return b
    if b == 'bool':
        return a
    floats = {a, b} & _FLOATS
    if floats:
        if 'float64' in floats:
            return 'float64'
        if 'float32' in floats or floats == {'float16', 'bfloat16'}:
            return 'float32'
        return next(iter(floats))
    if {a, b} == {'uint8', 'int8'}:
        return 'int16'
    return max((a, b), key=lambda d: int(re.search(r'\d+', d)[0]))


def _tensor_promotion(a, ashape, b, bshape):
    # ATen result_type distinguishes dimensioned and zero-dimensional tensors.
    if bool(ashape) == bool(bshape):
        return _promote(a, b)
    dim, zero = (a, b) if ashape else (b, a)
    category = lambda d: 2 if d in _FLOATS else 1 if d in _INTS else 0
    return zero if category(zero) > category(dim) else dim


def lower_mask_extension(node, values) -> dict | None:
    """Return fresh semantic annotations, None if unrelated; fail closed otherwise."""
    try:
        selected = _selection(node)
        return None if selected is None else _lower(node, values, *selected)
    except UnsupportedOperator:
        raise
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise UnsupportedOperator(node.get('id', '<missing>'), node.get('target', '<missing>'), str(exc)) from exc


def _lower(node, values, op, overload):
    from .semantics import _broadcast_index, _broadcast_shape

    outputs = deepcopy(node.get('outputs', []))
    if len(outputs) != 1:
        raise ValueError('exactly one tensor output required')
    out = outputs[0]
    shape, dtype = _shape(out), _dtype(out)
    args, kwargs = deepcopy(node.get('args', [])), deepcopy(node.get('kwargs', {}))
    params = _bind(op, args, kwargs)
    referenced = list(refs([args, kwargs]))
    if any(ref not in values for ref in referenced):
        raise ValueError('undefined or forward SSA reference')
    attrs = {'overload': overload, 'shape': shape}
    sem = {
        'domain': [{'var': f'i{j}', 'extent': n} for j, n in enumerate(shape)],
        'reads': [], 'reduction': None, 'dependence_kind': 'exact',
        'arithmetic': {'input_dtypes': [values[r].get('dtype') if values[r].get('kind') == 'scalar' else _dtype(values[r]) for r in referenced],
                       'output_dtype': dtype, 'rounding': 'at-output',
                       'reassociation': 'not-authorized-by-default'},
        'order': 'data-only', 'notes': [],
    }

    def tensor(name):
        value = params[name]
        if not isinstance(value, dict) or set(value) != {'ref'} or not isinstance(value['ref'], str):
            raise ValueError(f'{name} requires an SSA tensor reference')
        ref = value['ref']; meta = values[ref]
        source = _shape(meta)
        if meta.get('device') is not None and out.get('device') is not None and meta['device'] != out['device']:
            raise ValueError('operation cannot change device')
        return ref, source, _dtype(meta)

    def literal(value):
        if isinstance(value, dict) and set(value) == {'ref'}:
            meta = values[value['ref']]
            if meta.get('kind') != 'scalar' or 'specialized_value' not in meta:
                raise ValueError('scalar parameters require concrete specialization')
            sem.setdefault('specializations', []).append({'value': value['ref'], 'equals': deepcopy(meta['specialized_value'])})
            value = meta['specialized_value']
        return value

    def expect(expected_shape, expected_dtype):
        if shape != expected_shape or dtype != expected_dtype:
            raise ValueError(f'captured shape/dtype {shape}/{dtype} differs from semantic {expected_shape}/{expected_dtype}')

    def read(ref, indices, role='pointwise', **extra):
        sem['reads'].append({'value': ref, 'indices': indices, 'role': role, **extra})

    ref, source, source_dtype = tensor('self')
    indices = [f'i{j}' for j in range(len(shape))]
    if op in {'triu', 'tril'}:
        expect(source, source_dtype)
        if len(source) < 2:
            raise ValueError('triangular operation requires rank >= 2')
        diagonal = literal(params['diagonal'])
        if type(diagonal) is not int:
            raise ValueError('diagonal must be a concrete integer')
        guard = f'((i{len(shape)-1} - i{len(shape)-2}) {">=" if op == "triu" else "<="} ({diagonal}))'
        read(ref, indices, 'index-select', guard=guard)
        sem['constant_regions'] = [{'guard': f'not ({guard})', 'value': 0, 'dtype': dtype, 'conversion': 'ATen-scalar-to-output-dtype'}]
        sem['arithmetic']['rounding'] = 'none'
        attrs['diagonal'] = diagonal
        sem['notes'].append('Last-two-axis column minus row selects a source value; the complement is literal zero, including for NaN/Inf inputs, with no input value read.')
    elif op in {'fill', 'masked_fill'}:
        value = _number(literal(params['value']))
        attrs['value'] = value
        sem['arithmetic']['literal_conversion'] = {'value': value, 'to': dtype, 'rule': 'ATen-scalar-to-output-dtype'}
        region = {'value': value, 'dtype': dtype, 'conversion': 'ATen-scalar-to-output-dtype'}
        if op == 'fill':
            expect(source, source_dtype)
            region['guard'] = 'True'
            sem['metadata_reads'] = [{'value': ref, 'fields': ['shape', 'dtype', 'device']}]
            sem['notes'].append('Functional fill uses only input metadata: every output is a typed literal and no input value is a dependency.')
        else:
            mask_ref, mask_shape, mask_dtype = tensor('mask')
            if mask_dtype != 'bool':
                raise ValueError('masked_fill requires a bool mask')
            expect(_broadcast_shape([source, mask_shape]), source_dtype)
            source_indices = _broadcast_index(source, shape)
            mask_indices = _broadcast_index(mask_shape, shape)
            predicate = {'value': mask_ref, 'indices': mask_indices}
            sem['dependence_kind'] = 'may-dependence'
            sem['data_guards'] = [{'id': 'mask-false', 'predicate': deepcopy(predicate), 'equals': False},
                                  {'id': 'mask-true', 'predicate': deepcopy(predicate), 'equals': True}]
            read(mask_ref, mask_indices, 'predicate')
            read(ref, source_indices, 'data-select', data_guard='mask-false', dependence_kind='may-dependence')
            region['data_guard'] = 'mask-true'
            sem['selection'] = {'kind': 'data-dependent', 'predicate': predicate,
                                'true': {'literal': value, 'dtype': dtype},
                                'false': {'value': ref, 'indices': source_indices}}
            sem['notes'].append('Mask values, not indices, select literal versus data. Index-only scalarization retains both mask and possible input dependencies; this is NOT an exact selected cone. Value-aware execution must not read the unselected data branch.')
        sem['constant_regions'] = [region]
    elif op == 'remainder':
        if source_dtype not in _FLOATS | _INTS | {'bool'}:
            raise ValueError('remainder does not support complex tensors')
        if overload == 'Tensor':
            other_ref, other_shape, other_dtype = tensor('other')
            if other_dtype not in _FLOATS | _INTS | {'bool'}:
                raise ValueError('remainder does not support complex tensors')
            expect(_broadcast_shape([source, other_shape]), _tensor_promotion(source_dtype, source, other_dtype, other_shape))
            read(other_ref, _broadcast_index(other_shape, shape), operand=1)
            sem['arithmetic']['promotion'] = 'ATen-result_type-with-zero-dimensional-tensor-precedence'
        else:
            other = _number(literal(params['other']))
            expected_dtype = source_dtype
            if type(other) is float and source_dtype not in _FLOATS:
                # Captures do not record process default dtype. The output is the
                # evidence for this concrete capture's wrapped-scalar contract.
                if dtype not in _FLOATS:
                    raise ValueError('floating wrapped scalar must promote integral input to default floating dtype')
                expected_dtype = dtype
                attrs['captured_default_float_dtype'] = dtype
                sem['notes'].append('Integral tensor plus Python float uses the default floating dtype evidenced by the captured output; not an assertion of the process-global default.')
            elif type(other) is int and source_dtype == 'bool':
                expected_dtype = 'int64'
            expect(source, expected_dtype)
            if dtype in _INTS and other == 0:
                raise ValueError('integer remainder by literal zero is undefined')
            attrs['other'] = other
            sem['arithmetic'].update(promotion='ATen-wrapped-Python-scalar', literal_operands={'other': other})
        if dtype == 'bool':
            raise ValueError('bool remainder computation is unsupported by ATen')
        read(ref, _broadcast_index(source, shape), operand=0)
        sem['reads'].sort(key=lambda r: r['operand'])
        sem['pointwise'] = {'operator': 'remainder', 'definition': 'ATen floor-division remainder; nonzero finite result has divisor sign',
                            'nonfinite': 'ATen IEEE remainder behavior; not fmod and not an algebraic rewrite for nonfinite inputs',
                            'integer_zero_divisor': 'runtime error; never silently assigned a value'}
        sem['arithmetic']['computation_dtype'] = 'ATen-opmath-of-promoted-inputs'
    else:
        dim = _axis(literal(params['dim']), len(source))
        requested_dtype = literal(params['dtype'])
        if requested_dtype is None:
            expected_dtype = 'int64' if source_dtype in _INTS | {'bool'} else source_dtype
        else:
            if isinstance(requested_dtype, dict):
                if set(requested_dtype) != {'dtype'}:
                    raise ValueError('invalid dtype tag')
                requested_dtype = requested_dtype['dtype']
            if not isinstance(requested_dtype, str):
                raise ValueError('dtype must be a supported dtype string/tag')
            expected_dtype = _dtype({'dtype': requested_dtype})
        if expected_dtype == 'bool':
            raise ValueError('CPU cumsum does not support bool computation dtype')
        expect(source, expected_dtype)
        attrs.update(dim=dim, dtype=None if requested_dtype is None else expected_dtype)
        reduction_indices = ['r0' if j == dim else f'i{j}' for j in range(len(source))]
        guard = f'(r0 <= i{dim})' if source else 'True'
        read(ref, reduction_indices, 'reduction', guard=guard)
        sem['reduction'] = {'axes': [dim] if source else [],
                            'variables': [{'var': 'r0', 'extent': source[dim]}] if source else [],
                            'operator': 'sum', 'identity': 0, 'tree': 'unspecified', 'functional': True,
                            'kind': 'inclusive-prefix', 'guard': guard,
                            'group_size': f'(i{dim} + 1)' if source else 1}
        sem['arithmetic'].update(promotion='explicit-dtype-before-scan-or-int/bool-to-int64',
                                 computation_dtype=dtype, accumulation_dtype='ATen-backend-dependent-opmath',
                                 rounding='functional-prefix-reduction-implementation-dependent-and-at-output')
        sem['notes'].append('Output coordinate i reads every input coordinate 0..i along dim, inclusive; other coordinates are unchanged. Fixed axis extent plus r0<=i guard specifies the group, not a sequential scan or reduction tree.')
    sem['layout'] = {'kind': 'functional-fresh-output', 'alias': 'none',
                     'output_strides': deepcopy(out.get('strides')), 'output_storage_offset': out.get('storage_offset', 0),
                     'inputs': [{'value': r['value'], 'logical_index_map': r['indices'],
                                 'storage_index_map': _address(values[r['value']], r['indices']),
                                 **{key: r[key] for key in ('guard', 'data_guard', 'dependence_kind') if key in r}}
                                for r in sem['reads']],
                     'requires_materialization': False}
    result = deepcopy(node)
    result.update(op=op, args=args, kwargs=kwargs, outputs=outputs, attributes=attrs, semantics=sem)
    return result


def execute_mask_extension(node, args, kwargs, attrs):
    """Execute a closed primitive on strided CPU tensors; return None if unrelated.

    Resolved args take precedence; attributes fill omitted parameters. Nonfinite
    literal tags are strictly decoded here as well as in the capture lowerer.
    """
    selected = _selection(node, normalized=True)
    if selected is None:
        return None
    import torch

    op, overload = selected
    params = _bind(op, args, kwargs, attrs)

    def tensor(name):
        value = params[name]
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.layout != torch.strided:
            raise ValueError(f'{name} requires a strided CPU tensor')
        return value

    x = tensor('self')
    if op in {'triu', 'tril'}:
        diagonal = params['diagonal']
        if type(diagonal) is not int:
            raise ValueError('diagonal must be an integer')
        return (torch.triu if op == 'triu' else torch.tril)(x, diagonal=diagonal)
    if op == 'fill':
        return torch.full_like(x, _number(params['value']))
    if op == 'masked_fill':
        mask = tensor('mask')
        if mask.dtype != torch.bool:
            raise ValueError('masked_fill requires a bool mask')
        return torch.masked_fill(x, mask, _number(params['value']))
    if op == 'remainder':
        other = tensor('other') if overload == 'Tensor' else _number(params['other'])
        if not x.is_floating_point() and not x.is_complex():
            integral_other = isinstance(other, torch.Tensor) and not other.is_floating_point() and not other.is_complex()
            if (integral_other and bool((other == 0).any())) or (type(other) in (int, bool) and other == 0):
                raise ValueError('integer remainder by zero is undefined')
        return torch.remainder(x, other)
    dim = _axis(params['dim'], x.ndim)
    dtype = params['dtype']
    if dtype is not None:
        allowed = {name: getattr(torch, name) for name in _FLOATS | _INTS | {'complex64', 'complex128'}}
        if isinstance(dtype, dict):
            if set(dtype) != {'dtype'}:
                raise ValueError('invalid dtype tag')
            dtype = dtype['dtype']
        if isinstance(dtype, str):
            name = _dtype({'dtype': dtype})
            if name not in allowed:
                raise ValueError('unsupported cumsum dtype')
            dtype = allowed[name]
        elif not isinstance(dtype, torch.dtype) or dtype not in allowed.values():
            raise ValueError('unsupported cumsum dtype')
    return torch.cumsum(x, dim, dtype=dtype)
