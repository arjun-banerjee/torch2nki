from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize output tensor in HBM
    result = nl.zeros((a_tensor.shape[0], 1), dtype=a_tensor.dtype, buffer=nl.hbm)
    
    # Load vectors into on-chip memory and add
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    temp = nl.add(a_tile, b_tile)
    
    # Store result to output tensor
    nl.store(result, temp)
    
    return result