from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_atan(a_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input to SBUF
    x = nl.load(a_tensor)
    
    # Calculate arctangent using hardware instruction
    dummy = nl.arctan(x)
    
    # Store result back to HBM
    nl.store(result, value=dummy)
    
    return result