from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_acos(a_tensor):
    # Initialize result array
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data
    x = nl.load(a_tensor)
    
    # Constants
    pi_half = 1.5707963267948966
    
    # Calculate powers for polynomial approximation
    x2 = nl.multiply(x, x)
    x3 = nl.multiply(x2, x)
    x5 = nl.multiply(x3, x2)
    x7 = nl.multiply(x5, x2)
    
    # Calculate terms
    term1 = x
    term2 = nl.multiply(x3, 1.0/6.0)
    term3 = nl.multiply(x5, 3.0/40.0)
    term4 = nl.multiply(x7, 15.0/336.0)
    
    # Sum terms for arcsin approximation
    arcsin = nl.add(term1, term2)
    arcsin = nl.add(arcsin, term3)
    arcsin = nl.add(arcsin, term4)
    
    # Calculate acos using pi/2 - arcsin(x)
    temp = nl.subtract(pi_half, arcsin)
    
    # Store result
    nl.store(result, temp)
    
    return result