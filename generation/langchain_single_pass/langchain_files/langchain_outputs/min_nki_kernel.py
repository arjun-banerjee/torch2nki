from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_min(a_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input tensor to SBUF
    a_tile = nl.load(a_tensor)
    
    # Transpose to make reduction axis free
    a_tile_t = nl.transpose(a_tile)
    
    # Compute min on free axis
    temp = nl.min(a_tile_t, axis=1)
    
    # Store result back to HBM
    nl.store(result, temp)
    
    return result