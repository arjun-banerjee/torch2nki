from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Ensure both tensors are 1D
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
        
    # Load vectors from HBM to SBUF
    a = nl.load(a_tensor)
    b = nl.load(b_tensor)
    
    # Initialize result tensor
    result = nl.zeros((a_tensor.shape[0],), dtype=nl.float32, buffer=nl.sbuf)
    
    # Perform vector addition
    temp = nl.add(a, b)
    
    # Store result in result tensor
    result = temp
    
    return result