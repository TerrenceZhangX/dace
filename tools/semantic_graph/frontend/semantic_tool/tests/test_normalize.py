import unittest,torch
from semantic_tool.core import build,validate_graph
from semantic_tool.scalarize import instantiate,ancestors
from semantic_tool.evaluate import evaluate
from semantic_tool.capture import capture
from semantic_tool.semantics import lower_capture

class NormalizeTests(unittest.TestCase):
 def test_split_tail_exact_ranges(self):
  def fn(x):return torch.split(x,2,dim=1)
  x=torch.randn(2,5);g=build(fn,(x,),trusted=True)
  y=evaluate(g,{'x':x});self.assertEqual(len(y),3)
  for a,b in zip(y,fn(x)):self.assertTrue(torch.equal(a,b))
  ex=instantiate(g);self.assertEqual(len(ex['outputs']),10)
 def test_unbind_negative_axis(self):
  def fn(x):return torch.unbind(x,dim=-1)
  x=torch.randn(2,3);g=build(fn,(x,),trusted=True)
  for a,b in zip(evaluate(g,{'x':x}),fn(x)):self.assertTrue(torch.equal(a,b))
 def test_split_sizes_zero_piece(self):
  def fn(x):return torch.split(x,[1,0,3],dim=0)
  x=torch.randn(4);g=build(fn,(x,),trusted=True)
  self.assertEqual([list(y.shape) for y in evaluate(g,{'x':x})],[[1],[0],[3]])
 def test_kwargs_layout(self):
  # Exercise actual capture graph with parameters rewritten to equivalent kwargs.
  def fn(x):return torch.cat([x,x],dim=1)
  x=torch.randn(2,3);cg=capture(fn,(x,),trusted=True);n=cg['nodes'][0];n['kwargs']={'tensors':n['args'][0],'dim':1};n['args']=[]
  g=lower_capture(cg);self.assertTrue(torch.equal(evaluate(g,{'x':x}),fn(x)))
 def test_nonfinite_return_and_scalar(self):
  def fn(x,alpha):return x+alpha,float('nan')
  x=torch.ones(2);g=build(fn,(x,float('inf')),trusted=True)
  y=evaluate(g,{'x':x,'alpha':float('inf')});self.assertTrue(torch.isinf(y[0]).all());self.assertTrue(y[1]!=y[1])
if __name__=='__main__':unittest.main()
