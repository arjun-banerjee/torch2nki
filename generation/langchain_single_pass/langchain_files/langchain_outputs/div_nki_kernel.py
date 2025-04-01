from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_div(a_tensor, b_tensor):
    # Ensure tensors have the same shape
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Tensors must have the same shape")
    
    # Initialize result tensor in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load tensors into on-chip memory (SBUF)
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform element-wise division
    temp_result = nl.divide(a_tile, b_tile)
    
    # Store result back to HBM
    nl.store(result, temp_result)
    
    return result