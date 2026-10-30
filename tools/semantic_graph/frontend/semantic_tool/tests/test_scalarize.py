import unittest
from semantic_tool.scalarize import instantiate,ancestors,ExpansionLimit,index_eval
from semantic_tool.core import validate_graph,GraphValidationError

def inp(i,shape):return {'id':i,'name':i,'shape':shape,'dtype':'float32','kind':'tensor'}
def graph(inputs,node):return {'format':'semantic-graph-v1','inputs':inputs,'nodes':[node],'outputs':{'ref':'y'},'constants':{}}
def node(shape,reads,reduction=None):return {'id':'op0','op':'add','args':[{'ref':r['value']} for r in reads],'kwargs':{},'outputs':[inp('y',shape)],'semantics':{'domain':[{'var':f'i{i}','extent':v} for i,v in enumerate(shape)],'reads':reads,'reduction':reduction,'order':'data-only'}}

class ScalarTests(unittest.TestCase):
 def test_pointwise_independence(self):
  g=graph([inp('a',[4]),inp('b',[4])],node([4],[{'value':'a','indices':['i0']},{'value':'b','indices':['i0']}]))
  out=instantiate(g)
  for i in range(4):self.assertEqual(ancestors(out,f'y[{i}]'),{f'a[{i}]',f'b[{i}]'})
 def test_broadcast(self):
  g=graph([inp('a',[2,3]),inp('b',[3])],node([2,3],[{'value':'a','indices':['i0','i1']},{'value':'b','indices':['i1']}]))
  self.assertEqual(ancestors(instantiate(g),'y[1,2]'),{'a[1,2]','b[2]'})
 def test_rows_not_global_reduction(self):
  g=graph([inp('a',[2,3])],node([2],[{'value':'a','indices':['i0','r0']}],{'axes':[1],'variables':[{'var':'r0','extent':3}],'operator':'sum','tree':'unspecified'}))
  ex=instantiate(g)
  self.assertEqual(ancestors(ex,'y[1]'),{f'a[1,{r}]' for r in range(3)})
  self.assertEqual(ex['metadata']['reduction_groups'],2)
 def test_scatter_guard(self):
  g=graph([inp('base',[4]),inp('src',[2])],node([4],[{'value':'base','indices':['i0'],'guard':'i0 < 1 or i0 >= 3'},{'value':'src','indices':['i0-1'],'guard':'i0 >= 1 and i0 < 3'}]))
  ex=instantiate(g)
  self.assertEqual(ancestors(ex,'y[0]'),{'base[0]'})
  self.assertEqual(ancestors(ex,'y[2]'),{'src[1]'})
 def test_budget(self):
  g=graph([inp('a',[100])],node([100],[{'value':'a','indices':['i0']}]))
  with self.assertRaises(ExpansionLimit):instantiate(g,max_nodes=10)
 def test_bad_index(self):
  g=graph([inp('a',[3])],node([3],[{'value':'a','indices':['i0+1']}]))
  with self.assertRaises(GraphValidationError):instantiate(g)
 def test_index_expression_rejection(self):
  with self.assertRaises(GraphValidationError):index_eval("__import__('os').system('echo bad')",{})
if __name__=='__main__':unittest.main()
