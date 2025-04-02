from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_linalg_vecdot(vec1, vec2):
    # Initialize scalar result
    result = nl.zeros((), dtype=vec1.dtype, buffer=nl.shared_hbm)
    
    # Compute element-wise multiplication
    product = nl.multiply(vec1, vec2)
    
    # Sum up all elements along proper axis
    temp = nl.sum(nl.transpose(product), axis=1)
    
    # Store scalar result
    nl.store(result, temp)
    
    return result