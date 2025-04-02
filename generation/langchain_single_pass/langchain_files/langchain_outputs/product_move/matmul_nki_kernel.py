from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_matmul(a_tensor, b_tensor):
    # Initialize result array with the correct shape (M x N)
    M, K = a_tensor.shape
    _, N = b_tensor.shape
    result = nl.ndarray((M, N), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Since we're using transpose_x=True, we need to load the data appropriately
    # Load input matrices into on-chip memory (SBUF)
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform matrix multiplication
    # Using transpose_x=True for optimal performance as recommended in docs
    matmul_result = nl.matmul(nl.transpose(a_tile), b_tile, transpose_x=True)
    
    # Store the result back to HBM
    nl.store(result, matmul_result)
    
    return result