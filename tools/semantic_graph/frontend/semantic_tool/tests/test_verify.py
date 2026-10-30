import copy,unittest
from semantic_tool.verify import verify_semantics
from semantic_tool.semantics import lower_capture
from semantic_tool.core import GraphValidationError

def make():
    a={'id':'a','name':'a','kind':'tensor','shape':[2,3],'dtype':'float32','strides':[3,1]}
    b={'id':'b','name':'b','kind':'tensor','shape':[3],'dtype':'float32','strides':[1]}
    c={'id':'c','kind':'tensor','shape':[2,3],'dtype':'float32','strides':[3,1]}
    return lower_capture({'format':'aten-capture-v1','inputs':[a,b],'nodes':[{'id':'add','target':'aten.add.Tensor','args':[{'ref':'a'},{'ref':'b'}],'kwargs':{},'outputs':[c]}],'outputs':{'ref':'c'},'constants':{},'metadata':{}})
class AnnotationTests(unittest.TestCase):
 def test_valid(self):self.assertTrue(verify_semantics(make())['pass'])
 def test_changed_read(self):
    g=make();g['nodes'][0]['semantics']['reads'][1]['indices']=['0']
    with self.assertRaises(GraphValidationError):verify_semantics(g)
 def test_changed_operator(self):
    g=make();g['nodes'][0]['op']='mul'
    with self.assertRaises(GraphValidationError):verify_semantics(g)
 def test_changed_shape(self):
    g=make();g['nodes'][0]['outputs'][0]['shape']=[3,2]
    with self.assertRaises((GraphValidationError,NotImplementedError)):verify_semantics(g)
if __name__=='__main__':unittest.main()
