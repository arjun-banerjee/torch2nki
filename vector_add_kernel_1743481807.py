from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit 
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
    
    # Load vectors into on-chip memory and add
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    temp = nl.add(a_tile, b_tile)
    
    # Create result tensor and store computed value
    result = nl.zeros(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.hbm)
    nl.store(result, temp)
    
    return result