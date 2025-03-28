from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Ensure both tensors are 1D and same length
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize result tensor
    result = nl.zeros(a_tensor.shape, dtype=a_tensor.dtype)
    
    # Load vectors from HBM to on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform element-wise addition
    temp = nl.add(a_tile, b_tile)
    
    # Store result in result tensor
    nl.store(result, temp)
    
    return result