"""Direct indexing-extension tests independent of main mounting order."""
from copy import deepcopy
import itertools
import json
import unittest

import torch

from semantic_tool.core import UnsupportedOperator
from semantic_tool.ops_indexing import lower_indexing_extension, execute_indexing_extension


def meta(identifier, value):
    return {'id':identifier,'kind':'tensor','shape':list(value.shape),
            'dtype':str(value.dtype).removeprefix('torch.'),'strides':list(value.stride()),'device':'cpu'}


class IndexingExtensionTests(unittest.TestCase):
    def case(self, base, tensors, params, expected, overload='default'):
        vals = {f'x{i}':meta(f'x{i}',x) for i,x in enumerate(tensors)}
        encoded = [{'ref':f'x{i}'} for i in range(len(tensors))]
        rawargs = [encoded] + params if base == 'stack' else encoded + params
        node = {'id':'out','target':f'aten.{base}.{overload}','args':rawargs,'kwargs':{},'outputs':[meta('out',expected)]}
        graphnode = lower_indexing_extension(node,vals)
        self.assertIsNotNone(graphnode)
        json.dumps(graphnode,allow_nan=False)
        args = [tensors] + params if base == 'stack' else list(tensors) + params
        actual = execute_indexing_extension(graphnode,args,{},graphnode['attributes'])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
        self.assertEqual(actual.dtype,expected.dtype)
        self.assertEqual(graphnode['semantics']['order'],'data-only')
        return node, vals, graphnode

    def coordinate_check(self, node, tensors, expected):
        env = {f'x{i}':x for i,x in enumerate(tensors)}
        for coord in itertools.product(*(range(n) for n in expected.shape)):
            bind = {f'i{j}':v for j,v in enumerate(coord)}
            reads = []
            for read in node['semantics']['reads']:
                if eval(read.get('guard','True'),{'__builtins__':{}},bind):
                    index = tuple(eval(v,{'__builtins__':{}},bind) for v in read['indices'])
                    reads.append(env[read['value']][index])
            self.assertEqual(len(reads),1)
            torch.testing.assert_close(reads[0],expected[coord],rtol=0,atol=0)

    def test_repeat_prepend_noncontiguous_and_zero(self):
        x = torch.arange(6).reshape(2,3).t()
        for repeats in ([2,3],[2,1,2],[0,3]):
            expected = x.repeat(*repeats)
            _,_,node = self.case('repeat',[x],[repeats],expected)
            self.coordinate_check(node,[x],expected)
        scalar = torch.tensor(7)
        _,_,node = self.case('repeat',[scalar],[[2,3]],scalar.repeat(2,3))
        self.coordinate_check(node,[scalar],scalar.repeat(2,3))

    def test_roll_axes_negative_duplicate_flatten(self):
        x = torch.arange(24).reshape(4,6).t()
        for shifts,dims in (([2],[0]),([-9],[-1]),([2,-3],[0,0]),([3],[])):
            expected = torch.roll(x,shifts=tuple(shifts),dims=None if not dims else tuple(dims))
            _,_,node = self.case('roll',[x],[shifts,dims],expected)
            self.coordinate_check(node,[x],expected)
        empty = torch.empty(0,3)
        self.case('roll',[empty],[[4],[0]],torch.roll(empty,4,0))

    def test_stack_negative_dim_scalars_duplicate_inputs(self):
        x = torch.arange(6).reshape(2,3).t()
        y = x+20
        for dim in (0,1,-1):
            expected = torch.stack([x,y,x],dim=dim)
            _,_,node = self.case('stack',[x,y,x],[dim],expected)
            self.coordinate_check(node,[x,y,x],expected)
        a,b = torch.tensor(3),torch.tensor(7)
        _,_,node = self.case('stack',[a,b],[0],torch.stack([a,b]))
        self.coordinate_check(node,[a,b],torch.stack([a,b]))

    def test_floor_divide_negative_integer_and_broadcast(self):
        x = torch.tensor([[-5,-4,-1,0,4,5]],dtype=torch.int32)
        y = torch.tensor([[2],[-2]],dtype=torch.int64)
        _,_,node = self.case('floor_divide',[x,y],[],torch.floor_divide(x,y))
        self.assertEqual(node['semantics']['arithmetic']['rounding'],'floor-quotient-then-output-dtype')
        self.case('floor_divide',[x],[2],torch.floor_divide(x,2),overload='Scalar')

    def test_floor_divide_float_dtype_and_zero_dim_promotion(self):
        for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
            x = torch.tensor([-3.2,-1.1,0.,2.9],dtype=dtype)
            y = torch.tensor(2.,dtype=torch.float64)
            _,_,node = self.case('floor_divide',[x,y],[],torch.floor_divide(x,y))
            self.assertEqual(node['outputs'][0]['dtype'],str(torch.floor_divide(x,y).dtype).removeprefix('torch.'))

    def test_bitwise_boolean_and_integer_broadcast(self):
        for dtype in (torch.bool,torch.uint8,torch.int16,torch.int64):
            x = torch.tensor([[0,1,3]],dtype=dtype)
            y = torch.tensor([[1],[0]],dtype=dtype)
            for base,fn in [('bitwise_and',torch.bitwise_and),('bitwise_or',torch.bitwise_or),('bitwise_xor',torch.bitwise_xor)]:
                self.case(base,[x,y],[],fn(x,y),overload='Tensor')
            self.case('bitwise_not',[x],[],torch.bitwise_not(x))
        a = torch.tensor([1,2,3],dtype=torch.int16)
        b = torch.tensor([2,4,6],dtype=torch.int64)
        self.case('bitwise_and',[a,b],[],torch.bitwise_and(a,b),overload='Tensor')
        self.case('bitwise_and',[a],[3],torch.bitwise_and(a,3),overload='Scalar')

    def test_reject_unknown_invalid_overload_dtype_and_shape(self):
        self.assertIsNone(lower_indexing_extension({'target':'aten.sin.default'},{}))
        self.assertIsNone(execute_indexing_extension({'op':'unknown'},[],{},{}))
        x = torch.ones(2,3)
        node,vals,_ = self.case('repeat',[x],[[2,1]],x.repeat(2,1))
        wrong = deepcopy(node);wrong['target']='aten.repeat.out'
        with self.assertRaises(UnsupportedOperator):lower_indexing_extension(wrong,vals)
        wrong = deepcopy(node);wrong['outputs'][0]['shape']=[3,4]
        with self.assertRaises(UnsupportedOperator):lower_indexing_extension(wrong,vals)
        wrong = deepcopy(node);wrong['target']='aten.bitwise_not.default';wrong['args']=[{'ref':'x0'}];wrong['outputs']=[meta('out',x)]
        with self.assertRaisesRegex(UnsupportedOperator,'bitwise'):lower_indexing_extension(wrong,vals)
        with self.assertRaisesRegex(ValueError,'CPU'):
            execute_indexing_extension({'op':'repeat'},[torch.empty(2,device='meta'),[2]],{}, {})

    def test_reject_stack_mixed_dtype_bad_roll_and_repeat(self):
        x,y = torch.ones(2),torch.ones(2,dtype=torch.int64)
        node={'id':'o','target':'aten.stack.default','args':[[{'ref':'x0'},{'ref':'x1'}],0],'kwargs':{},'outputs':[meta('o',torch.stack([x,y]))]}
        with self.assertRaisesRegex(UnsupportedOperator,'same dtype'):
            lower_indexing_extension(node,{'x0':meta('x0',x),'x1':meta('x1',y)})
        with self.assertRaises(ValueError):
            execute_indexing_extension({'op':'roll'},[x,[1,2],[]],{}, {})
        with self.assertRaises(ValueError):
            execute_indexing_extension({'op':'repeat'},[x,[-1]],{}, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
