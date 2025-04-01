from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize result tensor with proper shape
    result = nl.zeros(a_tensor.shape[0], dtype=a_tensor.dtype, buffer=nl.sbuf)
    
    # Load vectors into on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform addition
    temp = nl.add(a_tile, b_tile)
    
    # Store result
    nl.store(result, temp)
    
    return result