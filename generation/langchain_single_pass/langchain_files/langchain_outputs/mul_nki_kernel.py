from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_mul(a_tensor, b_tensor):
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Tensors must have the same shape")
        
    # Initialize result array
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input tensors
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform element-wise multiplication
    temp = nl.multiply(a_tile, b_tile)
    
    # Store result
    nl.store(result, temp)
    
    return result