from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same length")
        
    i_p = nl.arange(a_tensor.shape[0])[:, None]
    result_sbuf = nl.zeros((a_tensor.shape[0], 1), dtype=a_tensor.dtype, buffer=nl.sbuf)
    
    a_tile = nl.load(a_tensor[i_p])
    b_tile = nl.load(b_tensor[i_p])
    
    temp = nl.add(a_tile, b_tile)
    nl.store(result_sbuf, temp)
    
    return result_sbuf