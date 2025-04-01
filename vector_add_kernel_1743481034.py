from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
    
    # Create indices for proper tensor handling
    i_p = nl.arange(a_tensor.shape[0])[:, None]
    i_f = nl.arange(1)[None, :]
    
    # Initialize result tensor
    result = nl.zeros((a_tensor.shape[0], 1), dtype=a_tensor.dtype, buffer=nl.sbuf)
    
    # Load vectors and perform addition
    a_tile = nl.load(a_tensor[i_p, i_f])
    b_tile = nl.load(b_tensor[i_p, i_f])
    temp = nl.add(a_tile, b_tile)
    
    # Store result
    nl.store(result[i_p, i_f], temp)
    
    return result