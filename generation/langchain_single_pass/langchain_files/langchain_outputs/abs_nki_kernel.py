from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_abs(a_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    input_tile = nl.load(a_tensor)
    
    # Compute absolute values
    output_tile = nl.abs(input_tile)
    
    # Store results back to HBM
    nl.store(result, value=output_tile)
    
    return result