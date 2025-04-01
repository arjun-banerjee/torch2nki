from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tanh(a_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    input_tile = nl.load(a_tensor)
    
    # Compute tanh using built-in NKI function
    output_tile = nl.tanh(input_tile)
    
    # Store result back to HBM
    nl.store(result, output_tile)
    
    return result