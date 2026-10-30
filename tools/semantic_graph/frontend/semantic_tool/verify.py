"""Recompute semantic annotations from captured targets to detect stale/tampered maps.
This checks consistency, not authenticating user-supplied code/graph provenance.
"""
from copy import deepcopy
from .core import validate_graph,GraphValidationError,graph_hash

def verify_semantics(graph):
    from .semantics import lower_capture
    validate_graph(graph)
    if graph['format']!='semantic-graph-v1':raise GraphValidationError('Requires semantic graph')
    capture=deepcopy(graph);capture['format']='aten-capture-v1'
    for node in capture['nodes']:
        for key in ('op','attributes','semantics'):node.pop(key,None)
    regenerated=lower_capture(capture)
    if len(regenerated['nodes'])!=len(graph['nodes']):raise GraphValidationError('Semantic node count changed')
    for expected,actual in zip(regenerated['nodes'],graph['nodes']):
        for key in ('id','op','attributes','semantics','outputs'):
            if expected.get(key)!=actual.get(key):
                raise GraphValidationError(f'{actual["id"]}: semantic annotation differs from target/args rule: {key}')
    return {'pass':True,'nodes':len(graph['nodes']),'graph_hash':graph_hash(graph),'scope':'annotation consistency only, not source authenticity or universal equivalence'}
