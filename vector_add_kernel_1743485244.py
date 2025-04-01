from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Ensure vectors have same length
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same shape")
    
    # Initialize result tensor on SBUF
    result = nl.zeros(a_tensor.shape, dtype=nl.float32, buffer=nl.sbuf)
    
    # Load input vectors from HBM to SBUF
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform element-wise addition
    temp = nl.add(a_tile, b_tile)
    
    # Store result back to result tensor
    result = temp
    
    return result