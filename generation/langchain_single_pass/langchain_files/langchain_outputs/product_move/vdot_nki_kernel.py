from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vdot(a_tensor, b_tensor):
    # Initialize result array in HBM
    result = nl.ndarray((), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input vectors
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Compute element-wise multiplication
    prod = nl.multiply(a_tile, b_tile)
    
    # Sum the products after transposing
    prod_transpose = nl.transpose(prod)
    sum_result = nl.sum(prod_transpose, axis=1)
    result_final = nl.transpose(sum_result)
    
    # Store result
    nl.store(result, result_final)
    
    return result