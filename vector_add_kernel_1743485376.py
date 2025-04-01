from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Ensure both tensors are 1D
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
        
    # Create index arrays for proper dimensionality
    i_p = nl.arange(a_tensor.shape[0])[:, None]
    
    # Load vectors from HBM to SBUF
    a = nl.load(a_tensor[i_p])
    b = nl.load(b_tensor[i_p])
    
    # Initialize result tensor on SBUF
    result = nl.zeros((a_tensor.shape[0],), dtype=nl.float32, buffer=nl.sbuf)
    
    # Perform vector addition
    temp = nl.add(a, b)
    
    # Store result back to result tensor
    nl.store(result, temp)
    
    return result