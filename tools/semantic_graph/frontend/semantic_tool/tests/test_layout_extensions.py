"""Synthetic CPU index enumeration only; never load/execute benchmark reference code."""
from copy import deepcopy
import itertools
import unittest

import torch
import torch.nn.functional as F

from semantic_tool.core import UnsupportedOperator, validate_graph
from semantic_tool.ops_layout import lower_layout_extension
from semantic_tool.scalarize import index_eval, instantiate


def metadata(name,tensor):
    return {'id':name,'name':name,'kind':'tensor','shape':list(tensor.shape),
            'dtype':str(tensor.dtype).removeprefix('torch.'),'strides':list(tensor.stride()),
            'storage_offset':tensor.storage_offset(),'device':'cpu'}


def ref(name):
    return {'ref':name}


def node(target,args,result,kwargs=None):
    return {'id':'output','target':target,'args':args,'kwargs':kwargs or {},
            'outputs':[metadata('output',result)]}


class LayoutExtensionTests(unittest.TestCase):
    def check_mapping(self,n,inputs,expected,extra_values=None):
        values={k:metadata(k,v) for k,v in inputs.items()}
        values.update(extra_values or {})
        snapshot=deepcopy((n,values))
        lower=lower_layout_extension(n,values)
        self.assertEqual((n,values),snapshot,'lowering must not mutate capture or metadata')
        self.assertEqual(lower['outputs'],n['outputs'])
        self.assertEqual(lower['semantics']['arithmetic']['output_dtype'],str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(lower['semantics']['layout']['alias'],'none')
        actual=torch.empty_like(expected)
        reads=lower['semantics']['reads']
        constants=lower['semantics'].get('constant_regions',[])
        total_copies=total_literals=0
        for coord in itertools.product(*(range(s) for s in expected.shape)):
            env={f'i{i}':c for i,c in enumerate(coord)}
            active=[r for r in reads if index_eval(r['guard'],env)]
            fills=[r for r in constants if index_eval(r['guard'],env)]
            self.assertEqual(len(active)+len(fills),1,(coord,active,fills))
            if active:
                total_copies+=1
                read=active[0]; source=inputs[read['value']]
                indices=tuple(index_eval(i,env) for i in read['indices'])
                self.assertEqual(len(indices),source.dim())
                self.assertTrue(all(0<=i<s for i,s in zip(indices,source.shape)))
                actual[coord]=source[indices]
                # Layout map is logical; physical offset must include source strides.
                maps=[m for m in lower['semantics']['layout']['inputs']
                      if m['value']==read['value'] and m['guard']==read['guard'] and m['logical_index_map']==read['indices']]
                self.assertTrue(maps)
                physical=index_eval(maps[0]['storage_index_map'],env)
                self.assertEqual(physical,source.storage_offset()+sum(i*s for i,s in zip(indices,source.stride())))
            else:
                total_literals+=1
                actual[coord]=fills[0]['value']
        torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
        self.assertEqual(actual.dtype,expected.dtype)
        self.assertEqual(actual.shape,expected.shape)
        return lower,total_copies,total_literals

    def test_cat_each_axis_and_negative_axes(self):
        for shape in [(2,3),(2,1,3),(1,2,1,3)]:
            for dim in range(len(shape)):
                for sign in (dim,dim-len(shape)):
                    with self.subTest(shape=shape,dim=sign):
                        a=torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float64).reshape(shape)
                        other=list(shape);other[dim]=1
                        b=torch.full(other,13.25,dtype=a.dtype)
                        expected=torch.cat((a,b,a),dim=sign)
                        n=node('aten.cat.default',[[ref('a'),ref('b'),ref('a')],sign],expected)
                        lowered,copies,literals=self.check_mapping(n,{'a':a,'b':b},expected)
                        self.assertEqual(copies,expected.numel());self.assertEqual(literals,0)
                        self.assertEqual(lowered['attributes']['dim'],dim)

    def test_cat_dtype_preservation(self):
        for dtype in (torch.bool,torch.int8,torch.int32,torch.int64,torch.float16,
                      torch.bfloat16,torch.float32,torch.float64,torch.complex64,torch.complex128):
            with self.subTest(dtype=dtype):
                a=torch.tensor([[0,1],[2,3]],dtype=dtype)
                b=torch.tensor([[4,5]],dtype=dtype)
                self.check_mapping(node('aten.cat.default',[[ref('a'),ref('b')],0],torch.cat([a,b],0)),{'a':a,'b':b},torch.cat([a,b],0))

    def test_cat_empty_inputs_and_legacy_empty_1d(self):
        cases=[(torch.empty(0),torch.arange(6.).reshape(2,3),1),
               (torch.empty(2,0),torch.arange(6.).reshape(2,3),1),
               (torch.empty(0),torch.empty(0),0),
               (torch.empty(0,2),torch.empty(0,3),1)]
        for a,b,dim in cases:
            with self.subTest(a=a.shape,b=b.shape,dim=dim):
                expected=torch.cat([a,b],dim)
                self.check_mapping(node('aten.cat.default',[[ref('a'),ref('b')],dim],expected),{'a':a,'b':b},expected)

    def test_cat_noncontiguous_offset_and_broadcast_stride(self):
        a=torch.arange(60.).reshape(5,12)[1:4,::3].t()
        b=torch.tensor([[-2.]]).expand(4,2)
        expected=torch.cat([a,b],-1)
        self.check_mapping(node('aten.cat.default',[[ref('a'),ref('b')],-1],expected),{'a':a,'b':b},expected)

    def test_cat_kwargs_and_specialized_axis(self):
        a=torch.arange(6.).reshape(2,3);expected=torch.cat([a,a],1)
        n=node('torch.ops.aten.cat.default',[],expected,{'tensors':[ref('a'),ref('a')],'dim':ref('axis')})
        self.check_mapping(n,{'a':a},expected,{'axis':{'id':'axis','kind':'scalar','dtype':'int','specialized_value':-1}})

    def test_pad_positive_negative_mixed_and_empty_specs(self):
        a=torch.arange(24.,dtype=torch.float64).reshape(2,3,4)
        cases=[[],[1,2],[0,0],[-1,0],[0,-1],[-1,-1],[1,2,2,1],[-1,2,1,-1],
               [1,0,0,1,1,0],[-1,1,-1,2,0,1],[-2,-2]]
        for pad in cases:
            with self.subTest(pad=pad):
                expected=F.pad(a,pad,value=-3.25)
                n=node('aten.constant_pad_nd.default',[ref('a'),pad,-3.25],expected)
                self.check_mapping(n,{'a':a},expected)

    def test_pad_all_literal_when_source_empty(self):
        a=torch.empty(2,0);expected=F.pad(a,[2,1],value=9)
        lowered,copies,literals=self.check_mapping(node('aten.constant_pad_nd.default',[ref('a'),[2,1],9],expected),{'a':a},expected)
        self.assertEqual(copies,0);self.assertEqual(literals,6)
        self.assertEqual(lowered['semantics']['constant_regions'][0]['value'],9)

    def test_pad_dtype_and_fill_conversion(self):
        for dtype in (torch.bool,torch.int8,torch.int32,torch.float16,torch.bfloat16,torch.float32,torch.float64,torch.complex64):
            for fill in (0,2,-1):
                with self.subTest(dtype=dtype,fill=fill):
                    a=torch.tensor([[0,1],[2,3]],dtype=dtype);expected=F.pad(a,[1,2,1,0],value=fill)
                    self.check_mapping(node('aten.constant_pad_nd.default',[ref('a'),[1,2,1,0],fill],expected),{'a':a},expected)

    def test_pad_nonfinite_float_literals(self):
        for fill in (float('inf'),float('-inf'),float('nan')):
            with self.subTest(fill=fill):
                a=torch.tensor([1.,2.]);expected=F.pad(a,[1,1],value=fill)
                self.check_mapping(node('aten.constant_pad_nd.default',[ref('a'),[1,1],fill],expected),{'a':a},expected)

    def test_pad_noncontiguous_source_and_channels_last(self):
        cases=[torch.arange(30.).reshape(5,6)[1:4,::2].t(),
               torch.arange(48.).reshape(2,2,3,4).contiguous(memory_format=torch.channels_last)]
        for a in cases:
            expected=F.pad(a,[1,2],value=7)
            self.check_mapping(node('aten.constant_pad_nd.default',[ref('a'),[1,2],7],expected),{'a':a},expected)

    def test_pad_default_fill_and_kwargs(self):
        a=torch.tensor([1,2],dtype=torch.int64);expected=F.pad(a,[1,2])
        self.check_mapping(node('aten.constant_pad_nd.default',[ref('a'),[1,2]],expected),{'a':a},expected)
        n=node('aten.constant_pad_nd.default',[],expected,{'self':ref('a'),'pad':[ref('left'),2],'value':ref('fill')})
        extras={'left':{'id':'left','kind':'scalar','dtype':'int','specialized_value':1},
                'fill':{'id':'fill','kind':'scalar','dtype':'int','specialized_value':0}}
        self.check_mapping(n,{'a':a},expected,extras)

    def test_pad_literal_regions_have_zero_scalar_edges(self):
        a=torch.tensor([2.,4.]);expected=F.pad(a,[1,2],value=-5)
        lower,_,_=self.check_mapping(node('aten.constant_pad_nd.default',[ref('a'),[1,2],-5],expected),{'a':a},expected)
        # No target: verify_semantics requires parent hook, not part of this module.
        # Scalarizer itself sees only normalized exact guarded reads.
        lower.pop('target')
        graph={'format':'semantic-graph-v1','inputs':[metadata('a',a)],'constants':{},'nodes':[lower],'outputs':ref('output')}
        validate_graph(graph)
        scalar=instantiate(graph)
        self.assertEqual(set(map(tuple,scalar['edges'])),{('a[0]','output[1]'),('a[1]','output[2]')})

    def test_unrelated_targets_return_none(self):
        self.assertIsNone(lower_layout_extension({'target':'aten.split.Tensor'},{}))
        self.assertIsNone(lower_layout_extension({'target':'aten.cat_extra.default'},{}))

    def test_fail_closed_cat_invalid_cases(self):
        a=torch.ones(2,3);b=torch.ones(2,4);expected=torch.ones(2,7)
        values={'a':metadata('a',a),'b':metadata('b',b)}
        valid=node('aten.cat.default',[[ref('a'),ref('b')],1],expected)
        cases=[]
        n=deepcopy(valid);n['target']='aten.cat.out';cases.append(n)
        n=deepcopy(valid);n['args'][1]=2;cases.append(n)
        n=deepcopy(valid);n['args'][1]=True;cases.append(n)
        n=deepcopy(valid);n['args'][0]=[];cases.append(n)
        n=deepcopy(valid);n['args'][0]=[ref('undefined')];cases.append(n)
        n=deepcopy(valid);n['outputs'][0]['shape']=[2,6];cases.append(n)
        n=deepcopy(valid);n['outputs'][0]['dtype']='int32';cases.append(n)
        n=deepcopy(valid);n['outputs'][0]['strides']=[1];cases.append(n)
        n=deepcopy(valid);n['outputs'][0]['storage_offset']=-1;cases.append(n)
        n=deepcopy(valid);n['outputs'][0]['device']='cuda';cases.append(n)
        n=deepcopy(valid);n['kwargs']['dim']=1;cases.append(n)
        n=deepcopy(valid);n['kwargs']['out']=ref('a');cases.append(n)
        n=deepcopy(valid);n['args'][1]=0;cases.append(n)
        n=deepcopy(valid);n['outputs']*=2;cases.append(n)
        for n in cases:
            with self.subTest(node=n):
                with self.assertRaises(UnsupportedOperator):lower_layout_extension(n,values)
        mixed=deepcopy(values);mixed['b']['dtype']='float64'
        with self.assertRaisesRegex(UnsupportedOperator,'same dtype'):lower_layout_extension(valid,mixed)

    def test_fail_closed_pad_invalid_cases(self):
        a=torch.ones(2,3);expected=F.pad(a,[1,1])
        values={'a':metadata('a',a)};valid=node('aten.constant_pad_nd.default',[ref('a'),[1,1]],expected)
        cases=[]
        for pad in ([1],[1,1,1,1,1,1],[1.0,1],[True,1],[-4,7]):
            n=deepcopy(valid);n['args'][1]=pad;cases.append(n)
        n=deepcopy(valid);n['outputs'][0]['shape']=[2,4];cases.append(n)
        n=deepcopy(valid);n['args'].append(ref('a'));cases.append(n)
        n=deepcopy(valid);n['args'].append('bad');cases.append(n)
        n=deepcopy(valid);n['args'].append(None);cases.append(n)
        n=deepcopy(valid);n['target']='aten.constant_pad_nd.out';cases.append(n)
        n=deepcopy(valid);n['outputs'][0]['dtype']='qint8';cases.append(n)
        for n in cases:
            with self.subTest(node=n):
                with self.assertRaises(UnsupportedOperator):lower_layout_extension(n,values)

    def test_pad_integer_overflow_rejected(self):
        a=torch.ones(2,dtype=torch.int8);expected=F.pad(a,[1,1])
        for fill in (128,-129,float('inf'),float('nan')):
            with self.subTest(fill=fill):
                n=node('aten.constant_pad_nd.default',[ref('a'),[1,1],fill],expected)
                with self.assertRaisesRegex(UnsupportedOperator,'integer dtype range'):
                    lower_layout_extension(n,{'a':metadata('a',a)})

    def test_pad_scalar_no_padding(self):
        a=torch.tensor(3.0);expected=F.pad(a,[])
        self.check_mapping(node('aten.constant_pad_nd.default',[ref('a'),[]],expected),{'a':a},expected)

    def test_pad_tagged_nonfinite_literals(self):
        a=torch.tensor([1.,2.])
        for tag in ('inf','-inf','nan'):
            fill=float(tag);expected=F.pad(a,[1,2],value=fill)
            with self.subTest(tag=tag):
                n=node('aten.constant_pad_nd.default',[ref('a'),[1,2],{'float':tag}],expected)
                self.check_mapping(n,{'a':a},expected)
                n=node('aten.constant_pad_nd.default',[ref('a'),[1,2],ref('fill')],expected)
                self.check_mapping(n,{'a':a},expected,{'fill':{'id':'fill','kind':'scalar','dtype':'float','specialized_value':{'float':tag}}})
        for invalid in ('Infinity','0',None,[],'__import__'):
            n=node('aten.constant_pad_nd.default',[ref('a'),[1,2],{'float':invalid}],F.pad(a,[1,2]))
            with self.subTest(invalid=invalid),self.assertRaises(UnsupportedOperator):
                lower_layout_extension(n,{'a':metadata('a',a)})

    def _capture_pipeline(self,function,inputs):
        import json
        from semantic_tool.capture import capture
        from semantic_tool.semantics import lower_capture
        from semantic_tool.evaluate import evaluate
        cg=capture(function,tuple(inputs.values()),list(inputs),trusted=True)
        cg=json.loads(json.dumps(cg,allow_nan=False))
        graph=lower_capture(cg)
        graph=json.loads(json.dumps(graph,allow_nan=False))
        validate_graph(graph)
        expanded=instantiate(graph,max_nodes=5000,max_edges=10000)
        actual=evaluate(graph,inputs)
        expected=function(*inputs.values())
        torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
        self.assertEqual(actual.dtype,expected.dtype)
        self.assertEqual(actual.shape,expected.shape)
        return cg,graph,expanded

    def test_real_function_cat_capture_lower_scalarize_evaluate(self):
        def synthetic_cat(a,b):
            return torch.cat((a,b,a),dim=-1)
        a=torch.arange(12.,dtype=torch.float64).reshape(3,4)[:,::2]
        b=torch.tensor([[3.],[4.],[5.]],dtype=a.dtype)
        cg,graph,expanded=self._capture_pipeline(synthetic_cat,{'a':a,'b':b})
        self.assertEqual([n['target'] for n in cg['nodes']],['aten.cat.default'])
        self.assertEqual(graph['nodes'][0]['op'],'cat')
        self.assertEqual(len(expanded['edges']),synthetic_cat(a,b).numel())

    def test_real_function_nonfinite_pad_capture_lower_scalarize_evaluate(self):
        for tag in ('-inf','inf','nan'):
            with self.subTest(tag=tag):
                fill=float(tag)
                def synthetic_pad(a):
                    return F.pad(a,(1,2,1,0),value=fill)
                a=torch.arange(6.,dtype=torch.float32).reshape(2,3)
                cg,graph,expanded=self._capture_pipeline(synthetic_pad,{'a':a})
                self.assertEqual([n['target'] for n in cg['nodes']],['aten.constant_pad_nd.default'])
                self.assertEqual(cg['nodes'][0]['args'][2],{'float':tag})
                self.assertEqual(graph['nodes'][0]['semantics']['constant_regions'][0]['value'],{'float':tag})
                self.assertEqual(len(expanded['edges']),a.numel(),'padding must have no input-value edges')


if __name__=='__main__':unittest.main()
