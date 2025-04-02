from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sigmoid(a_tensor):
    # Initialize the result tensor
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load the input tensor into SBUF
    input_tile = nl.load(a_tensor)
    
    # Apply sigmoid function
    dummy_variable = nl.sigmoid(input_tile)
    
    # Store result back to HBM
    nl.store(result, dummy_variable)

    return result