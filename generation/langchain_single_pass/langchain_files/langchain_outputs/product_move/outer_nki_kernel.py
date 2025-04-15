from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_outer(a_tensor, b_tensor):
    # Get shapes of input vectors
    m = a_tensor.shape[0]
    n = b_tensor.shape[0]
    
    # Initialize result array with proper shape and buffer
    result = nl.ndarray((m, n), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Create index ranges for broadcasting
    i = nl.arange(m)[:, None]  # Column vector
    j = nl.arange(n)[None, :]  # Row vector
    
    # Load input vectors into on-chip memory
    a_tile = nl.load(a_tensor[i])
    b_tile = nl.load(b_tensor[j])
    
    # Compute outer product through element-wise multiplication of broadcasted vectors
    out_tile = nl.multiply(a_tile, b_tile)
    
    # Store result back to HBM
    nl.store(result, out_tile)
    
    return result