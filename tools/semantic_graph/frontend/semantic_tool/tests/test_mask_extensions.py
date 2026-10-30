"""Generic real-function captures and independent tiny CPU/index oracles.

No benchmark task Python imports, reference execution, GPU or environment edits.
Existing updated captures are read only as operator/metadata regression fixtures.
"""
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import unittest

import torch

from semantic_tool.capture import capture
from semantic_tool.core import UnsupportedOperator
from semantic_tool.literals import encode_floats
from semantic_tool.ops_mask import lower_mask_extension, execute_mask_extension
from semantic_tool.scalarize import ancestors, index_eval, instantiate, value_key


def ref(name):
    return {'ref': name}


def meta(name, x):
    return {'id': name, 'kind': 'tensor', 'shape': list(x.shape),
            'dtype': str(x.dtype).removeprefix('torch.'), 'strides': list(x.stride()),
            'storage_offset': x.storage_offset(), 'device': 'cpu'}


def fixture(target, inputs, expected, tail=(), kwargs=None):
    values = {f'x{j}': meta(f'x{j}', x) for j, x in enumerate(inputs)}
    node = {'id': 'out', 'target': target, 'args': [ref(r) for r in values] + list(tail),
            'kwargs': kwargs or {}, 'outputs': [meta('out', expected)]}
    return node, values


def lower(target, inputs, expected, tail=(), kwargs=None):
    node, values = fixture(target, inputs, expected, tail, kwargs)
    return lower_mask_extension(node, values)


def points(shape):
    return itertools.product(*(range(n) for n in shape))


def mapped(read, coord, reduction=None):
    env = {f'i{j}': v for j, v in enumerate(coord)}
    if reduction is not None:
        env['r0'] = reduction
    return tuple(index_eval(expr, env) for expr in read['indices'])


def expanded(node, inputs):
    node = deepcopy(node)
    node.pop('target', None)  # test the extension independently of integration timing
    graph = {'format': 'semantic-graph-v1', 'inputs': [meta(f'x{j}', x) for j, x in enumerate(inputs)],
             'constants': {}, 'nodes': [node], 'outputs': ref('out')}
    return instantiate(graph)


class MaskExtensionTests(unittest.TestCase):
    def test_public_masked_fill_build_evaluate_verify_and_may_scalarize(self):
        from semantic_tool import build
        from semantic_tool.core import GraphValidationError
        from semantic_tool.evaluate import evaluate
        from semantic_tool.verify import verify_semantics

        x = torch.tensor([[math.nan, 2., math.inf], [4., -math.inf, 6.]])
        mask = torch.tensor([[True, False, True]])
        graph = build(lambda a, m: a.masked_fill(m, -math.inf), (x, mask),
                      input_names=['x', 'mask'], trusted=True)
        self.assertTrue(verify_semantics(graph)['pass'])
        self.assertEqual(graph['nodes'][0]['semantics']['dependence_kind'], 'may-dependence')
        self.assertEqual(graph['nodes'][0]['attributes']['value'], {'float': '-inf'})
        actual = evaluate(graph, {'x': x, 'mask': mask})
        expected = torch.tensor([[-math.inf, 2., -math.inf], [-math.inf, -math.inf, -math.inf]])
        torch.testing.assert_close(actual, expected)
        expanded_graph = instantiate(graph)
        self.assertEqual(expanded_graph['metadata']['dependence_kind'], 'may-dependence')
        self.assertIn('possible', expanded_graph['metadata']['scope'])
        altered = deepcopy(graph)
        altered['nodes'][0]['semantics']['data_guards'][0]['equals'] = True
        with self.assertRaises(GraphValidationError):
            verify_semantics(altered)

        # Also exercise a composed public path with three newly supported ops.
        finite = torch.tensor([[math.nan, 2., 3.], [4., 5., math.inf]])
        combined_mask = torch.tensor([[True, False, False], [False, False, True]])
        combined = build(lambda a, m: a.triu().masked_fill(m, 0).cumsum(-1),
                         (finite, combined_mask), input_names=['x', 'mask'], trusted=True)
        self.assertEqual([n['op'] for n in combined['nodes']], ['triu', 'masked_fill', 'cumsum'])
        self.assertTrue(verify_semantics(combined)['pass'])
        torch.testing.assert_close(evaluate(combined, {'x': finite, 'mask': combined_mask}),
                                   torch.tensor([[0., 2., 5.], [0., 5., 5.]]))
        self.assertEqual(instantiate(combined)['metadata']['dependence_kind'], 'may-dependence')

    def test_triangles_all_elements_batch_rectangular_noncontiguous(self):
        x = torch.arange(48, dtype=torch.float64).reshape(2, 4, 6).transpose(-1, -2)
        for op in ('triu', 'tril'):
            for diagonal in (-8, -1, 0, 2, 8):
                with self.subTest(op=op, diagonal=diagonal):
                    expected = torch.empty_like(x)
                    for coord in points(x.shape):
                        delta = coord[-1] - coord[-2]
                        expected[coord] = x[coord] if (delta >= diagonal if op == 'triu' else delta <= diagonal) else 0
                    node = lower(f'aten.{op}.default', [x], expected, [diagonal])
                    actual = execute_mask_extension(node, [x, diagonal], {}, node['attributes'])
                    torch.testing.assert_close(actual, expected)
                    graph = expanded(node, [x])
                    read = node['semantics']['reads'][0]
                    for coord in points(x.shape):
                        env = {f'i{j}': v for j, v in enumerate(coord)}
                        selected = index_eval(read['guard'], env)
                        self.assertEqual(ancestors(graph, value_key('out', coord)), {value_key('x0', coord)} if selected else set())
                        self.assertEqual(index_eval(node['semantics']['constant_regions'][0]['guard'], env), not selected)

    def test_triangles_boolean_complex_empty_and_nonfinite_zero_regions(self):
        inputs = [torch.tensor([[True, False], [True, True]]),
                  torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]]),
                  torch.empty(2, 0, 3), torch.full((3, 3), math.nan)]
        for x in inputs:
            for op in ('triu', 'tril'):
                expected = (torch.triu if op == 'triu' else torch.tril)(x)
                node = lower(f'aten.{op}.default', [x], expected)
                actual = execute_mask_extension(node, [x], {}, node['attributes'])
                torch.testing.assert_close(actual, expected, equal_nan=True)
                self.assertEqual(node['attributes']['diagonal'], 0)
                self.assertEqual(node['semantics']['dependence_kind'], 'exact')
                if torch.is_floating_point(x) and x.numel():
                    for coord in points(x.shape):
                        selected = coord[-1] >= coord[-2] if op == 'triu' else coord[-1] <= coord[-2]
                        if not selected:
                            self.assertEqual(actual[coord].item(), 0)

    def test_fill_metadata_only_all_indices_no_value_reads(self):
        for x in (torch.full((2, 3), math.nan).t(), torch.empty(0, 3), torch.tensor(123), torch.tensor(True)):
            expected = torch.full_like(x, 0)
            node = lower('aten.fill.Scalar', [x], expected, [0])
            self.assertEqual(node['semantics']['reads'], [])
            self.assertEqual(node['semantics']['metadata_reads'][0]['value'], 'x0')
            self.assertEqual(node['semantics']['constant_regions'][0]['guard'], 'True')
            actual = execute_mask_extension(node, [x, 0], {}, node['attributes'])
            torch.testing.assert_close(actual, expected)
            graph = expanded(node, [x])
            for coord in points(x.shape):
                self.assertEqual(ancestors(graph, value_key('out', coord)), set())
            self.assertIsNot(actual, x)

    def test_masked_fill_broadcast_selection_values_and_possible_cones(self):
        for x, mask in [(torch.arange(12.).reshape(2, 2, 3), torch.tensor([[True, False, True]])),
                        (torch.tensor([7.]), torch.tensor([[True], [False], [True]])),
                        (torch.tensor(math.nan), torch.tensor(False)),
                        (torch.empty(2, 0), torch.empty(1, 0, dtype=torch.bool))]:
            expected_shape = torch.broadcast_shapes(x.shape, mask.shape)
            bx, bm = x.expand(expected_shape), mask.expand(expected_shape)
            expected = torch.empty(expected_shape, dtype=x.dtype)
            for coord in points(expected.shape):
                expected[coord] = -math.inf if bm[coord].item() else bx[coord].item()
            node = lower('aten.masked_fill.Scalar', [x, mask], expected, [{'float': '-inf'}])
            sem = node['semantics']
            self.assertEqual(sem['dependence_kind'], 'may-dependence')
            self.assertEqual([g['equals'] for g in sem['data_guards']], [False, True])
            self.assertEqual(sem['constant_regions'][0]['data_guard'], 'mask-true')
            self.assertNotIn('guard', sem['constant_regions'][0])
            mask_read, data_read = sem['reads']
            self.assertEqual(mask_read['role'], 'predicate')
            self.assertEqual(data_read['data_guard'], 'mask-false')
            self.assertTrue(all('guard' not in r for r in sem['reads']))
            actual = execute_mask_extension(node, [x, mask, {'float': '-inf'}], {}, node['attributes'])
            torch.testing.assert_close(actual, expected, equal_nan=True)
            graph = expanded(node, [x, mask])
            self.assertEqual(graph['metadata']['dependence_kind'], 'may-dependence')
            self.assertIn('possible', graph['metadata']['scope'])
            for coord in points(expected.shape):
                xcoord, mcoord = mapped(data_read, coord), mapped(mask_read, coord)
                possible = {value_key('x0', xcoord), value_key('x1', mcoord)}
                self.assertEqual(ancestors(graph, value_key('out', coord)), possible)
                # Resolve the declared data guard independently using predicate values.
                selected_data = not mask[mcoord].item()
                self.assertEqual(selected_data, mask[mcoord].item() == sem['data_guards'][0]['equals'])
                if not selected_data:
                    self.assertTrue(torch.isneginf(actual[coord]).item())

    def test_masked_fill_dtype_retained_and_literal_nonfinite_not_multiply(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            x = torch.tensor([math.nan, math.inf, -math.inf, 9.], dtype=dtype)
            mask = torch.tensor([True, True, True, False])
            expected = torch.tensor([0., 0., 0., 9.], dtype=dtype)
            node = lower('aten.masked_fill.Scalar', [x, mask], expected, [0])
            actual = execute_mask_extension(node, [x, mask, 0], {}, node['attributes'])
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.dtype, x.dtype)

    def test_cumsum_every_prefix_axis_scalar_empty_and_negative_dimensions(self):
        for x in (torch.arange(24.).reshape(2, 3, 4).transpose(0, 2), torch.tensor(7, dtype=torch.int32),
                  torch.empty(2, 0, 3), torch.tensor([[True, False, True], [False, True, True]])):
            for dim in range(-max(x.ndim, 1), max(x.ndim, 1)):
                with self.subTest(shape=x.shape, dtype=x.dtype, dim=dim):
                    axis = dim % max(x.ndim, 1)
                    dtype = x.dtype if x.is_floating_point() else torch.int64
                    expected = torch.empty(x.shape, dtype=dtype)
                    for coord in points(x.shape):
                        if x.ndim:
                            expected[coord] = sum(x[coord[:axis] + (r,) + coord[axis+1:]].item() for r in range(coord[axis]+1))
                        else:
                            expected[coord] = x.item()
                    node = lower('aten.cumsum.default', [x], expected, [dim])
                    actual = execute_mask_extension(node, [x, dim], {}, node['attributes'])
                    torch.testing.assert_close(actual, expected)
                    sem = node['semantics']; reduction = sem['reduction']
                    self.assertEqual(reduction['tree'], 'unspecified')
                    self.assertTrue(reduction['functional'])
                    self.assertEqual(reduction['variables'], [{'var': 'r0', 'extent': x.shape[axis]}] if x.ndim else [])
                    graph = expanded(node, [x])
                    for coord in points(x.shape):
                        wanted = {value_key('x0', coord[:axis] + (r,) + coord[axis+1:]) for r in range(coord[axis]+1)} if x.ndim else {'x0[]'}
                        self.assertEqual(ancestors(graph, value_key('out', coord)), wanted)

    def test_cumsum_explicit_dtype_cast_before_reduction_and_overflow(self):
        x = torch.tensor([[120, 120, 120], [-120, -120, -120]], dtype=torch.int16)
        for dtype in (torch.int8, torch.int32, torch.int64, torch.float64, torch.complex64):
            expected = x.to(dtype).cumsum(-1, dtype=dtype)
            dtype_tag = {'dtype': str(dtype).removeprefix('torch.')}
            node = lower('aten.cumsum.default', [x], expected, [-1], {'dtype': dtype_tag})
            actual = execute_mask_extension(node, [x, -1], {'dtype': dtype_tag}, node['attributes'])
            torch.testing.assert_close(actual, expected)
            self.assertEqual(node['semantics']['arithmetic']['output_dtype'], dtype_tag['dtype'])
            self.assertEqual(node['semantics']['arithmetic']['computation_dtype'], dtype_tag['dtype'])
            torch.testing.assert_close(execute_mask_extension(node, [x], {}, node['attributes']), expected)

    def test_cumsum_low_precision_nonfinite_and_complex(self):
        for x in (torch.tensor([math.inf, -math.inf, 3.]), torch.linspace(-1, 1, 9).half(),
                  torch.linspace(-1, 1, 9).bfloat16(), torch.tensor([1+2j, -3+4j])):
            expected = torch.cumsum(x, 0)
            node = lower('aten.cumsum.default', [x], expected, [0])
            torch.testing.assert_close(execute_mask_extension(node, [x, 0], {}, node['attributes']), expected, equal_nan=True)

    def test_remainder_scalar_floor_sign_and_indices(self):
        for dtype in (torch.int64, torch.float64):
            x = torch.tensor([[-7, -4, -1], [0, 2, 8]], dtype=dtype).t()
            for divisor in (-3, 3):
                expected = torch.tensor([[value.item() % divisor for value in row] for row in x], dtype=dtype)
                node = lower('aten.remainder.Scalar', [x], expected, [divisor])
                actual = execute_mask_extension(node, [x, divisor], {}, node['attributes'])
                torch.testing.assert_close(actual, expected)
                graph = expanded(node, [x])
                for coord in points(x.shape):
                    self.assertEqual(ancestors(graph, value_key('out', coord)), {value_key('x0', coord)})

    def test_remainder_tensor_broadcast_every_element_and_mixed_promotion(self):
        cases = [(torch.tensor([[-7], [8]], dtype=torch.int8), torch.tensor([2, 3, 4], dtype=torch.uint8)),
                 (torch.tensor([[-7.5], [8.5]], dtype=torch.float16), torch.tensor([2., 3.], dtype=torch.bfloat16)),
                 (torch.tensor([[-7.5], [8.5]], dtype=torch.float32), torch.tensor(3., dtype=torch.float64)),
                 (torch.tensor([1, 0], dtype=torch.int8), torch.tensor(2., dtype=torch.float64)),
                 (torch.tensor(True), torch.tensor([2, 3], dtype=torch.int16))]
        for x, y in cases:
            dtype = torch.result_type(x, y)
            bx, by = torch.broadcast_tensors(x.to(dtype), y.to(dtype))
            expected = torch.empty_like(bx)
            for coord in points(expected.shape):
                expected[coord] = bx[coord].item() % by[coord].item()
            node = lower('aten.remainder.Tensor', [x, y], expected)
            actual = execute_mask_extension(node, [x, y], {}, node['attributes'])
            torch.testing.assert_close(actual, expected)
            graph = expanded(node, [x, y])
            for coord in points(expected.shape):
                wanted = {value_key(read['value'], mapped(read, coord)) for read in node['semantics']['reads']}
                self.assertEqual(ancestors(graph, value_key('out', coord)), wanted)

    def test_remainder_scalar_wrapped_promotion_and_nonfinite(self):
        for x, other in [(torch.tensor([1, 5], dtype=torch.int32), 2.5),
                         (torch.tensor([True, False]), 2),
                         (torch.tensor([1., -1., math.inf, math.nan]), {'float': 'inf'}),
                         (torch.tensor([1., -1.]), 0.0)]:
            value = math.inf if isinstance(other, dict) else other
            expected = torch.remainder(x, value)
            node = lower('aten.remainder.Scalar', [x], expected, [other])
            torch.testing.assert_close(execute_mask_extension(node, [x, other], {}, node['attributes']), expected, equal_nan=True)
            self.assertEqual(node['semantics']['arithmetic']['output_dtype'], str(expected.dtype).removeprefix('torch.'))

    def test_all_remainder_tensor_dtype_rank_promotion_pairs(self):
        from semantic_tool.ops_mask import _tensor_promotion
        names = ['bool', 'uint8', 'int8', 'int16', 'int32', 'int64',
                 'float16', 'bfloat16', 'float32', 'float64']
        for a, b in itertools.product(names, repeat=2):
            for ashape, bshape in (([2], [2]), ([], [2]), ([2], []), ([], [])):
                expected = torch.result_type(torch.empty(ashape, dtype=getattr(torch, a)),
                                             torch.empty(bshape, dtype=getattr(torch, b)))
                self.assertEqual(_tensor_promotion(a, ashape, b, bshape), str(expected).removeprefix('torch.'))

    def test_strict_nonfinite_tags_and_json_roundtrip(self):
        x = torch.ones(3)
        for tag in ('-inf', 'inf', 'nan'):
            node = lower('aten.fill.Scalar', [x], x, [{'float': tag}])
            encoded = json.loads(json.dumps(encode_floats(node), allow_nan=False))
            self.assertEqual(encoded['attributes']['value'], {'float': tag})
            actual = execute_mask_extension(encoded, [x], {}, encoded['attributes'])
            wanted = torch.full_like(x, float(tag))
            torch.testing.assert_close(actual, wanted, equal_nan=True)
        for bad in ({'float': 'Infinity'}, {'float': 1}, {'float': ['inf']}, {'float': 'inf', 'extra': 1}, 'nan'):
            node, values = fixture('aten.fill.Scalar', [x], x, [bad])
            with self.assertRaises(UnsupportedOperator):
                lower_mask_extension(node, values)
            with self.assertRaises(ValueError):
                execute_mask_extension(node, [x, bad], {}, {})

    def test_specialized_scalar_fill_no_dynamic_value_dependency(self):
        x = torch.arange(4.)
        node, values = fixture('aten.fill.Scalar', [x], x, [ref('constant')])
        values['constant'] = {'id': 'constant', 'kind': 'scalar', 'dtype': 'float', 'specialized_value': {'float': '-inf'}}
        result = lower_mask_extension(node, values)
        self.assertEqual(result['semantics']['reads'], [])
        self.assertEqual(result['semantics']['specializations'][0]['value'], 'constant')
        self.assertEqual(result['attributes']['value'], -math.inf)
        del values['constant']['specialized_value']
        with self.assertRaises(UnsupportedOperator):
            lower_mask_extension(node, values)

    def test_malformed_calls_fail_closed_and_unrelated_return_none(self):
        self.assertIsNone(lower_mask_extension({'target': 'aten.sin.default'}, {}))
        self.assertIsNone(execute_mask_extension({'target': 'aten.sin.default'}, [], {}, {}))
        x = torch.ones(2, 3)
        cases = [('aten.triu.out', [0], {}), ('aten.fill.Tensor', [2], {}),
                 ('aten.tril.default', [True], {}), ('aten.cumsum.default', [2], {}),
                 ('aten.cumsum.default', [0, {'dtype': 'float32'}], {}),
                 ('aten.triu.default', [0], {'diagonal': 1}), ('aten.fill.Scalar', [], {}),
                 ('aten.fill.Scalar', [1], {'unexpected': 1})]
        for target, tail, kwargs in cases:
            node, values = fixture(target, [x], x, tail, kwargs)
            with self.subTest(target=target, tail=tail, kwargs=kwargs), self.assertRaises(UnsupportedOperator):
                lower_mask_extension(node, values)
        node, values = fixture('aten.triu.default', [x], x)
        node['outputs'][0]['shape'] = [3, 2]
        with self.assertRaises(UnsupportedOperator):
            lower_mask_extension(node, values)
        node['outputs'][0]['shape'] = [2, 3]
        del values['x0']
        with self.assertRaises(UnsupportedOperator):
            lower_mask_extension(node, values)

    def test_masked_fill_wrong_mask_dtype_and_remainder_complex_rejected(self):
        x = torch.ones(2)
        for target, inputs, tail in [('aten.masked_fill.Scalar', [x, torch.ones(2)], [0]),
                                     ('aten.remainder.Tensor', [x.to(torch.complex64), x], [])]:
            node, values = fixture(target, inputs, x, tail)
            with self.assertRaises(UnsupportedOperator):
                lower_mask_extension(node, values)

    def test_integer_remainder_zero_fails_without_crash(self):
        x = torch.tensor([1, 2])
        node, values = fixture('aten.remainder.Scalar', [x], x, [0])
        with self.assertRaises(UnsupportedOperator):
            lower_mask_extension(node, values)
        with self.assertRaisesRegex(ValueError, 'zero'):
            execute_mask_extension(node, [x, 0], {}, {})
        with self.assertRaisesRegex(ValueError, 'zero'):
            execute_mask_extension({'target': 'aten.remainder.Tensor'}, [x, torch.tensor([1, 0])], {}, {})

    def test_cpu_only_helper_and_no_input_mutation(self):
        x = torch.tensor([[1., 2.], [3., 4.]])
        old = x.clone()
        for op, tail in [('triu.default', [0]), ('fill.Scalar', [2]), ('remainder.Scalar', [3]), ('cumsum.default', [0])]:
            node = {'target': 'aten.' + op}
            execute_mask_extension(node, [x] + tail, {}, {})
            torch.testing.assert_close(x, old)
            with self.assertRaisesRegex(ValueError, 'CPU'):
                execute_mask_extension(node, [x.to('meta')] + tail, {}, {})

    def test_real_function_captures_all_targets_and_helper_execution(self):
        x = torch.tensor([[-3., 1., 4.], [7., -8., 2.]])
        m = torch.tensor([[True, False, True]])
        y = torch.tensor([2., 3., 4.])
        cases = [(lambda a: a.triu(-1), (x,), 'aten.triu.default'),
                 (lambda a: a.tril(1), (x,), 'aten.tril.default'),
                 (lambda a, b: a.masked_fill(b, -math.inf), (x, m), 'aten.masked_fill.Scalar'),
                 (lambda a: torch.ops.aten.fill.Scalar(a, 5), (x,), 'aten.fill.Scalar'),
                 (lambda a: a.remainder(3), (x,), 'aten.remainder.Scalar'),
                 (lambda a, b: a.remainder(b), (x, y), 'aten.remainder.Tensor'),
                 (lambda a: a.cumsum(-1, dtype=torch.float64), (x,), 'aten.cumsum.default')]
        for fn, inputs, target in cases:
            with self.subTest(target=target):
                captured = capture(fn, inputs, trusted=True)
                json.dumps(captured, allow_nan=False)
                self.assertEqual(len(captured['nodes']), 1)
                raw = captured['nodes'][0]
                self.assertEqual(raw['target'], target)
                values = {v['id']: v for v in captured['inputs']}
                before = deepcopy(raw)
                node = lower_mask_extension(raw, values)
                self.assertEqual(raw, before)
                env = {v['id']: t for v, t in zip(captured['inputs'], inputs)}
                def resolve(value):
                    if isinstance(value, dict) and set(value) == {'ref'}:
                        return env[value['ref']]
                    if isinstance(value, list):
                        return [resolve(v) for v in value]
                    if isinstance(value, dict):
                        return {k: resolve(v) for k, v in value.items()}
                    return value
                actual = execute_mask_extension(node, resolve(raw['args']), resolve(raw['kwargs']), node['attributes'])
                torch.testing.assert_close(actual, fn(*inputs), equal_nan=True)
                json.dumps(encode_floats(node), allow_nan=False)

    def test_existing_updated_captures_operator_metadata_only(self):
        root = Path(__file__).resolve().parents[1] / 'semantic_tool' / 'runs'
        captures = list(root.glob('l1_updated_*/*/capture.json'))
        if not captures:
            self.skipTest('existing updated capture corpus not available')
        found = set(); count = 0
        for path in captures:
            graph = json.loads(path.read_text(encoding='utf-8'))
            values = {v['id']: v for v in graph['inputs']}
            values.update({key: dict(value, id=key) for key, value in graph.get('constants', {}).items()})
            for node in graph['nodes']:
                result = lower_mask_extension(node, values)
                if result is not None:
                    found.add(node['target']); count += 1
                    self.assertEqual(result['outputs'], node['outputs'])
                    json.dumps(encode_floats(result), allow_nan=False)
                values.update({v['id']: v for v in node['outputs']})
        self.assertGreaterEqual(count, 6)
        self.assertTrue({'aten.triu.default', 'aten.tril.default', 'aten.fill.Scalar',
                         'aten.masked_fill.Scalar', 'aten.remainder.Scalar', 'aten.cumsum.default'} <= found)


if __name__ == '__main__':
    unittest.main()
