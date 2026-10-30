"""Capture-owned tests; run with python -m unittest discover -s semantic_tool/tests -p test_capture.py."""
import json
from pathlib import Path
import unittest

import torch

from semantic_tool.capture import CaptureError, UntrustedCallableError, capture, trust_callable
from semantic_tool.task_io import (DEFAULT_TASK_ROOT, discover_tasks, input_metadata,
                                   load_task, make_example_args, resolve_expression)


class CaptureTests(unittest.TestCase):
    def checked(self, fn, args, names=None):
        graph = capture(fn, args, names, trusted=True)
        json.loads(json.dumps(graph, allow_nan=False))
        self.assertEqual(graph['format'], 'aten-capture-v1')
        defined = {item['id'] for item in graph['inputs']} | set(graph['constants'])
        def check(value):
            if isinstance(value, dict):
                if set(value) == {'ref'}:
                    self.assertIn(value['ref'], defined)
                else:
                    for part in value.values():
                        check(part)
            elif isinstance(value, list):
                for part in value:
                    check(part)
        for node in graph['nodes']:
            check(node['args'])
            check(node['kwargs'])
            for output in node['outputs']:
                self.assertNotIn(output['id'], defined)
                defined.add(output['id'])
        check(graph['outputs'])
        return graph

    def test_user_ref_string_mapping_fails_closed(self):
        for fn in (lambda x: {'ref': 'x'}, lambda x: [x, {'nested': {'ref': 'x'}}]):
            with self.assertRaisesRegex(CaptureError, 'reserved v1 reference'):
                capture(fn, (torch.empty(2, device='meta'),), trusted=True)
        graph = capture(lambda x: {'ref': x, 'dtype': {'dtype': 'float32'}},
                        (torch.empty(2, device='meta'),), trusted=True)
        self.assertIsInstance(graph['outputs']['ref'], dict)
        self.assertEqual(graph['outputs']['dtype'], {'dtype': 'float32'})

    def test_unknown_callable_never_executes(self):
        effects = []
        def unknown(x):
            effects.append('executed')
            return x + 1
        with self.assertRaises(UntrustedCallableError):
            capture(unknown, (torch.ones(2),))
        self.assertEqual(effects, [])
        trust_callable(unknown)
        graph = capture(unknown, (torch.ones(2),))
        self.assertEqual(graph['metadata']['tracing_mode'], 'fake')

    def test_broadcast_scalar0d_and_original_dtype(self):
        graph = self.checked(lambda x, y, z: (x * y + z).float(),
                             (torch.empty(2, 3, dtype=torch.bfloat16, device='meta'),
                              torch.empty(3, dtype=torch.bfloat16, device='meta'),
                              torch.empty((), dtype=torch.float32, device='meta')),
                             ['x', 'y', 'z'])
        self.assertEqual(graph['inputs'][2]['shape'], [])
        self.assertEqual(graph['inputs'][2]['kind'], 'tensor')
        self.assertEqual(graph['inputs'][0]['dtype'], 'bfloat16')
        self.assertTrue(any(n['target'] == 'aten._to_copy.default' for n in graph['nodes']))

    def test_scalar_specialization(self):
        graph = self.checked(lambda x, eps: x + eps, (torch.empty(2, device='meta'), 1e-5))
        scalar = graph['inputs'][1]
        self.assertEqual(scalar['kind'], 'scalar')
        self.assertEqual(scalar['specialized_value'], 1e-5)
        self.assertEqual(graph['nodes'][0]['args'][1], 1e-5)

    def test_shape_dtype_device_and_constants_strict_json(self):
        def fn(x):
            y = torch.tensor([1., 2.], dtype=torch.float32)
            return {'out': (x + y).to(dtype=torch.float64, device='cpu').clone(memory_format=torch.contiguous_format)}
        graph = self.checked(fn, (torch.ones(2),))
        self.assertTrue(graph['constants'])
        self.assertEqual(next(iter(graph['constants'].values()))['values'], [1., 2.])
        payload = json.dumps(graph)
        self.assertIn('"dtype": "float64"', payload)
        self.assertIn('"device": "cpu"', payload)
        self.assertIn('"memory_format": "contiguous_format"', payload)

    def test_multi_output_getitem_normalized(self):
        graph = self.checked(lambda x: torch.max(x, dim=1), (torch.empty(2, 3, device='meta'),))
        max_node = next(n for n in graph['nodes'] if n['target'] == 'aten.max.dim')
        self.assertEqual([d['id'] for d in max_node['outputs']], [max_node['id'] + ':0', max_node['id'] + ':1'])
        self.assertIsInstance(graph['outputs'], list)
        self.assertFalse(any('getitem' in n['target'] for n in graph['nodes']))

    def test_loop_mutation_is_functional_ssa(self):
        def fn(x):
            output = torch.zeros_like(x)
            for t in range(x.shape[1]):
                output[:1, t] = x[:1, t] + t
            return output
        original = torch.arange(6.).reshape(2, 3)
        graph = self.checked(fn, (original,))
        targets = graph['metadata']['targets']
        self.assertIn('aten.select_scatter.default', targets)
        self.assertIn('aten.slice_scatter.default', targets)
        self.assertNotIn('aten.copy_.default', targets)
        torch.testing.assert_close(original, torch.arange(6.).reshape(2, 3))

    def test_input_mutation_fail_closed(self):
        def fn(x):
            x.add_(1)
            return x
        with self.assertRaisesRegex(CaptureError, 'mutation|copy-back'):
            capture(fn, (torch.zeros(3),), trusted=True)

    def test_views_transpose_matmul_softmax_reductions(self):
        def fn(x, w):
            y = (x.view(2, 3) @ w.t()).transpose(0, 1)
            return [y.softmax(-1), y.sum(), y.mean(-1, keepdim=True)]
        graph = self.checked(fn, (torch.empty(6, device='meta'), torch.empty(4, 3, device='meta')))
        targets = graph['metadata']['targets']
        for target in ('aten.view.default', 'aten.mm.default', 'aten.transpose.int',
                       'aten._softmax.default', 'aten.sum.default', 'aten.mean.dim'):
            self.assertIn(target, targets)

    def test_huge_meta_shape_has_no_real_allocation(self):
        graph = self.checked(lambda x: x + 1, (torch.empty(10**10, device='meta'),))
        self.assertEqual(graph['inputs'][0]['shape'], [10**10])
        self.assertEqual(len(graph['nodes']), 1)

    def test_data_dependent_capture_fails(self):
        def fn(x):
            return x + 1 if x.sum().item() else x - 1
        with self.assertRaises(CaptureError):
            capture(fn, (torch.ones(2),), trusted=True)

    def test_non_json_and_callback_inputs_fail(self):
        with self.assertRaisesRegex(CaptureError, 'unsupported argument'):
            capture(lambda x, callback: callback(x), (torch.ones(2), lambda x: x), trusted=True)

    def test_nonfinite_literals_strict_json_roundtrip(self):
        import math
        graph = self.checked(lambda x: {'positive': x + float('inf'),
                                        'negative': x.masked_fill(x > 0, float('-inf')),
                                        'missing': float('nan')},
                             (torch.empty(3, device='meta'),))
        payload = json.dumps(graph, allow_nan=False)
        self.assertNotIn('Infinity', payload)
        self.assertNotIn('NaN', payload)
        restored = json.loads(payload)
        self.assertEqual(restored['metadata']['nonfinite_literals'], ['-inf', 'inf', 'nan'])
        self.assertEqual(restored['outputs']['missing'], {'float': 'nan'})
        self.assertTrue(math.isnan(float(restored['outputs']['missing']['float'])))
        self.assertTrue(any({'float': '-inf'} in n['args'] for n in restored['nodes']))
        self.assertTrue(any({'float': 'inf'} in n['args'] for n in restored['nodes']))

    def test_nonfinite_scalar_input_and_tensor_constant(self):
        graph = self.checked(lambda x, scale: x + scale, (torch.empty(2, device='meta'), float('-inf')))
        self.assertEqual(graph['inputs'][1]['specialized_value'], {'float': '-inf'})
        constant = torch.tensor([float('-inf'), float('inf'), float('nan')])
        graph = self.checked(lambda x: x + constant, (torch.ones(3),))
        values = next(iter(graph['constants'].values()))['values']
        self.assertEqual(values, [{'float': '-inf'}, {'float': 'inf'}, {'float': 'nan'}])

    def test_user_nonfinite_float_tag_mapping_fails_closed(self):
        for token in ('inf', '-inf', 'nan'):
            with self.assertRaisesRegex(CaptureError, 'reserved v1 float'):
                capture(lambda x: {'nested': {'float': token}}, (torch.empty(2, device='meta'),), trusted=True)
        graph = self.checked(lambda x: {'float': 'ordinary'}, (torch.empty(2, device='meta'),))
        self.assertEqual(graph['outputs'], {'float': 'ordinary'})


class TaskIOTests(unittest.TestCase):
    def test_custom_python_scalar_initialization_and_specialization(self):
        from types import SimpleNamespace
        definition = {'axes': {'n': {'type': 'var'}}, 'custom_inputs_entrypoint': 'get_inputs',
                      'inputs': {'x': {'shape': ['n'], 'dtype': 'float32'},
                                 'eps': {'shape': None, 'dtype': 'float32'},
                                 'enabled': {'shape': None, 'dtype': 'bool'}}}
        workload = {'inputs': {name: {'type': 'custom'} for name in definition['inputs']}}
        seen = []
        def generator(axes, device):
            seen.append(dict(axes))
            return {'x': torch.empty(axes['n'], device=device), 'eps': 0.125, 'enabled': True}
        meta = input_metadata(definition, workload, {'n': 3})
        self.assertTrue(meta[1]['pending_custom_initialization'])
        self.assertNotIn('specialized_value', meta[1])
        args = make_example_args(definition, workload, {'n': 3}, module=SimpleNamespace(get_inputs=generator), trusted=True)
        self.assertEqual(seen, [{'n': 3}])
        self.assertIs(type(args[1]), float)
        self.assertIs(type(args[2]), bool)
        resolved = input_metadata(definition, workload, {'n': 3}, scalar_values=dict(zip(definition['inputs'], args)))
        self.assertEqual(resolved[1]['specialized_value'], 0.125)
        self.assertNotIn('pending_custom_initialization', resolved[1])
        graph = capture(lambda x, eps, enabled: x + eps if enabled else x, args, trusted=True)
        self.assertEqual(graph['inputs'][1]['specialized_value'], 0.125)
        self.assertEqual(graph['inputs'][2]['kind'], 'scalar')

    def test_custom_scalar_missing_tensor_and_wrong_dtype_fail(self):
        from types import SimpleNamespace
        definition = {'axes': {}, 'custom_inputs_entrypoint': 'get_inputs',
                      'inputs': {'eps': {'shape': None, 'dtype': 'float32'}}}
        workload = {'inputs': {'eps': {'type': 'custom'}}}
        for value in ({}, {'eps': torch.tensor(0.1)}, {'eps': '0.1'}, {'eps': float('inf')}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_example_args(definition, workload, {},
                                  module=SimpleNamespace(get_inputs=lambda axes, device: value), trusted=True)
        effects = []
        with self.assertRaises(UntrustedCallableError):
            make_example_args(definition, workload, {}, module=SimpleNamespace(get_inputs=lambda *a: effects.append(1)))
        self.assertEqual(effects, [])

    def test_shape_expressions_are_not_eval(self):
        self.assertEqual(resolve_expression('n * (h + 1)', {'n': 2, 'h': 3}), 8)
        for bad in ['__import__("os").system("echo side-effect")', 'a.__class__', '[1][0]']:
            with self.assertRaises(ValueError):
                resolve_expression(bad, {})

    @unittest.skipUnless(DEFAULT_TASK_ROOT.is_dir(), 'Optional external L1 dataset unavailable; set SOL_EXECBENCH_L1_DIR')
    def test_trust_required_before_import(self):
        with self.assertRaises(UntrustedCallableError):
            load_task(discover_tasks()[0])

    @unittest.skipUnless(DEFAULT_TASK_ROOT.is_dir(), 'Optional external L1 dataset unavailable; set SOL_EXECBENCH_L1_DIR')
    def test_every_task_original_reference_and_dtype(self):
        paths = discover_tasks()
        self.assertGreaterEqual(len(paths), 8)
        for path in paths:
            with self.subTest(task=path.name):
                task = load_task(path, trusted=True)
                self.assertEqual(task.metadata['shape_scope'], 'reduced-probe-not-official')
                self.assertNotEqual(task.official_axes, task.probe_axes)
                graph = capture(task.function, task.example_args, task.input_names)
                json.dumps(graph, allow_nan=False)
                for actual, declared in zip(graph['inputs'], task.input_meta):
                    if declared['kind'] == 'tensor':
                        self.assertEqual(actual['dtype'], declared['dtype'])
                        self.assertEqual(actual['shape'], declared['shape'])
                self.assertGreater(len(graph['nodes']), 0)

    @unittest.skipUnless(DEFAULT_TASK_ROOT.is_dir(), 'Optional external L1 dataset unavailable; set SOL_EXECBENCH_L1_DIR')
    def test_non_dimension_axis_constants_do_not_shrink(self):
        path = next(p for p in discover_tasks() if p.name.startswith('046_'))
        task = load_task(path, trusted=True)
        self.assertEqual(task.probe_axes['attn_logit_softcapping'], 30)
        self.assertNotIn('attn_logit_softcapping', task.metadata['axis_changes'])

    @unittest.skipUnless(DEFAULT_TASK_ROOT.is_dir(), 'Optional external L1 dataset unavailable; set SOL_EXECBENCH_L1_DIR')
    def test_official_meta_axes(self):
        # Non-recurrent official workload is captured with no allocated payload.
        path = next(p for p in discover_tasks() if p.name.startswith('077_'))
        task = load_task(path, probe=False, trusted=True)
        self.assertEqual(task.metadata['shape_scope'], 'official-workload')
        self.assertEqual(task.official_axes, task.probe_axes)
        self.assertEqual(task.example_args[1].shape, (51866, 1280))
        graph = capture(task.function, task.example_args, task.input_names)
        self.assertEqual(graph['inputs'][1]['dtype'], task.definition['inputs']['weight']['dtype'])

    @unittest.skipUnless(DEFAULT_TASK_ROOT.is_dir(), 'Optional external L1 dataset unavailable; set SOL_EXECBENCH_L1_DIR')
    def test_custom_recurrent_initialization_unchanged_on_cpu(self):
        path = next(p for p in discover_tasks() if p.name.startswith('094_'))
        task = load_task(path, trusted=True, device='cpu')
        inputs = dict(zip(task.input_names, task.example_args))
        self.assertTrue(torch.all(inputs['max_state'] == -1e38))
        self.assertTrue(torch.all(inputs['num_state'] == 0))
        self.assertTrue(torch.all(inputs['den_state'] == 0))
        result = task.function(*task.example_args)
        self.assertTrue(all(torch.isfinite(t).all() for t in result))
        self.assertEqual(task.metadata['custom_initialization'], 'get_inputs')

    @unittest.skipUnless(DEFAULT_TASK_ROOT.is_dir(), 'Optional external L1 dataset unavailable; set SOL_EXECBENCH_L1_DIR')
    def test_gpu_is_rejected_and_cpu_size_is_bounded(self):
        path = next(p for p in discover_tasks() if p.name.startswith('077_'))
        with self.assertRaisesRegex(ValueError, 'GPU'):
            load_task(path, trusted=True, device='cuda')
        with self.assertRaisesRegex(ValueError, 'exceeds cap'):
            load_task(path, trusted=True, probe=False, device='cpu')


if __name__ == '__main__':
    unittest.main(verbosity=2)
