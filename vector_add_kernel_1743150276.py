from neuronxcc import nki
import neuronxcc.nki.language as nl
from torch_neuronx import nki_jit
import numpy as np

@nki_jit
def vector_add_kernel(v1, v2, result):
    # Get dimensions 
    sz_p, sz_f = v1.shape
    
    # Create index arrays
    i_p = nl.arange(sz_p)[:, None]
    i_f = nl.arange(sz_f)[None, :]

    # Load input vectors into on-chip memory
    v1_tile = nl.load(v1[i_p, i_f])
    v2_tile = nl.load(v2[i_p, i_f])
    
    # Perform vector addition
    out_tile = nl.add(v1_tile, v2_tile)
    
    # Store result back to output tensor
    nl.store(result[i_p, i_f], out_tile)