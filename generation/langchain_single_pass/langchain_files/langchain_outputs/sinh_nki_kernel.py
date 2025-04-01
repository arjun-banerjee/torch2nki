from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sinh(a_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    a_sbuf = nl.load(a_tensor)
    
    # Calculate e^x and e^-x using nl.exp
    pos_exp = nl.exp(a_sbuf)
    neg_exp = nl.exp(nl.subtract(0, a_sbuf))  # -x
    
    # Calculate sinh = (e^x - e^-x) / 2
    temp = nl.subtract(pos_exp, neg_exp)
    temp = nl.divide(temp, 2.0)
    
    # Store result back to HBM
    nl.store(result, temp)
    
    return result