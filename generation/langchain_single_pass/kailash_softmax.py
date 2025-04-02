from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_softmax(a_tensor):
    # Initialize result array
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load tensor into SBUF
    input_tile = nl.load(a_tensor)
    
    
    # Compute log_softmax by applying softmax first, then taking log
    softmax_output = nl.softmax(nl.transpose(input_tile), axis=1)
    #output = nl.log(softmax_output)
    
    # Store result back to HBM
    nl.store(result, nl.transpose(softmax_output))
    
    return result