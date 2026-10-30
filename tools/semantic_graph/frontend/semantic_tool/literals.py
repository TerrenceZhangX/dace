"""Strict JSON tags for IEEE nonfinite literals, not general object decoding."""
import math

def decode_float(value):
    if isinstance(value,dict) and set(value)=={'float'} and value['float'] in ('inf','-inf','nan'):
        return {'inf':float('inf'),'-inf':-float('inf'),'nan':float('nan')}[value['float']]
    return value

def encode_floats(value):
    if isinstance(value,float) and not math.isfinite(value):return {'float':'nan' if math.isnan(value) else 'inf' if value>0 else '-inf'}
    if isinstance(value,dict):return {k:encode_floats(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode_floats(v) for v in value]
    return value

def equal_tagged(a,b):
    return encode_floats(a)==encode_floats(b)
