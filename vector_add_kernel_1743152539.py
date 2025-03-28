from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(v1_tensor, v2_tensor):
    # Verify input tensors have same shape
    if v1_tensor.shape != v2_tensor.shape:
        raise ValueError("Input tensors must have the same shape")
    
    # Initialize result tensor with zeros
    result = nl.zeros(v1_tensor.shape, dtype=v1_tensor.dtype)
    
    # Load input vectors from HBM to SBUF
    v1_tile = nl.load(v1_tensor)
    v2_tile = nl.load(v2_tensor)
    
    # Perform vector addition
    temp = nl.add(v1_tile, v2_tile)
    
    # Store result
    nl.store(result, temp)
    
    return result