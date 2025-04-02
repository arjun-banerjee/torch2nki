from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_dot(a_tensor, b_tensor):
    # Ensure input tensors are 1D and same length
    if len(a_tensor.shape) != 1 or len(b_tensor.shape) != 1:
        raise ValueError("Input tensors must be 1D")
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Input tensors must have same length")

    # Initialize result as scalar 
    result = nl.ndarray((), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Create index arrays for reshaping
    i_p = nl.arange(a_tensor.shape[0])[:, None]
    i_f = nl.arange(1)[None, :]
    
    # Load tensors and reshape through indexing
    a_sbuf = nl.load(a_tensor[i_p])  # Shape: [N, 1]
    b_sbuf = nl.load(b_tensor[i_p])  # Shape: [N, 1]
    
    # Perform matrix multiplication: [1,N] x [N,1] -> [1,1]
    temp = nl.matmul(nl.transpose(a_sbuf), b_sbuf)
    
    # Store scalar result
    nl.store(result, temp[0,0])
    
    return result