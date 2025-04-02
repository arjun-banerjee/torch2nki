from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_einsum(equation, x, y):
    # Get shapes from input tensors
    M, K = x.shape
    K2, N = y.shape
    
    # Initialize result array for matrix multiplication output in HBM
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Load inputs into SBUF memory with explicit float32 dtype
    x_data = nl.load(x, dtype=nl.float32)
    y_data = nl.load(y, dtype=nl.float32)
    
    # Compute matrix multiplication
    output = nl.matmul(x_data, y_data)
    
    # Store result back to HBM
    nl.store(result, output)
    
    return result