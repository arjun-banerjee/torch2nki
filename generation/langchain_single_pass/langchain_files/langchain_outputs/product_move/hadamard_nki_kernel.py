from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_hadamard(a_tensor, b_tensor):
    # Check if tensors have the same shape
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Both tensors must have the same shape")
    
    # Initialize result tensor in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load tensors from HBM to on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform element-wise multiplication
    output_tile = nl.multiply(a_tile, b_tile)
    
    # Store result back to HBM
    nl.store(result, output_tile)
    
    return result