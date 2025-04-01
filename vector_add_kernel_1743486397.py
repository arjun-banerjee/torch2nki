from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same length")
        
    # Initialize result array
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input tensors into SBUF
    a_buf = nl.load(a_tensor)
    b_buf = nl.load(b_tensor)
    
    # Perform element-wise addition
    temp = nl.add(a_buf, b_buf)
    
    # Store result back to HBM
    nl.store(result, temp)
    
    return result