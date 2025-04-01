from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Ensure vectors are the same length
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize result tensor on SBUF
    result = nl.zeros(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.sbuf)
    
    # Load input vectors from HBM to on-chip memory (SBUF)
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform vector addition using NKI add operation
    temp = nl.add(a_tile, b_tile)
    
    # Store result in our result tensor
    nl.store(result, temp)
    
    return result