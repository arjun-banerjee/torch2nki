from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_relu(a_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    data_tile = nl.load(a_tensor)
    
    # Compute ReLU: max(0, x)
    # Create a zeros tensor of same shape and dtype as input
    zeros = nl.zeros(data_tile.shape, dtype=data_tile.dtype)
    
    # Compute element-wise maximum between input and zeros
    output_tile = nl.maximum(data_tile, zeros)
    
    # Store result back to HBM
    nl.store(result, output_tile)
    
    return result