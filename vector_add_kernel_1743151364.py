from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit  
def nki_vector_add(a_tensor, b_tensor, out_tensor):
    # Load input tensors into on-chip memory
    i_p = nl.arange(a_tensor.shape[0])[:, None]
    
    # Load input vectors
    a_tile = nl.load(a_tensor[i_p])
    b_tile = nl.load(b_tensor[i_p])
    
    # Perform addition
    result = nl.add(a_tile, b_tile)
    
    # Store result back to output tensor
    nl.store(out_tensor[i_p], result)