from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_cross(a_tensor, b_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray((3,), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Create partition dimension index
    i_p = nl.arange(1)[:, None]
    
    # Load vector components individually
    a0 = nl.load(a_tensor[0])
    a1 = nl.load(a_tensor[1])
    a2 = nl.load(a_tensor[2])
    
    b0 = nl.load(b_tensor[0])
    b1 = nl.load(b_tensor[1])
    b2 = nl.load(b_tensor[2])
    
    # Compute cross product components
    x = nl.multiply(a1, b2) - nl.multiply(a2, b1)
    y = nl.multiply(a2, b0) - nl.multiply(a0, b2)
    z = nl.multiply(a0, b1) - nl.multiply(a1, b0)
    
    # Store results
    nl.store(result[0], x)
    nl.store(result[1], y)
    nl.store(result[2], z)
    
    return result