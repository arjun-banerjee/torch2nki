from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_inner(a_tensor, b_tensor):
    # Initialize result array
    result = nl.ndarray((), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load inputs into SBUF memory
    a_sbuf = nl.load(a_tensor)
    b_sbuf = nl.load(b_tensor)
    
    # Multiply the vectors element-wise
    product = nl.multiply(nl.transpose(a_sbuf), nl.transpose(b_sbuf))
    
    # Sum up the products along axis=1 to get inner product
    summed = nl.sum(product, axis=1)
    
    # Store result
    nl.store(result, summed)
    
    return result