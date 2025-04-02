from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_mm(a_tensor, b_tensor):
    # Initialize result array 
    result = nl.ndarray((a_tensor.shape[0], b_tensor.shape[1]), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load inputs into SBUF and transpose first matrix for optimal performance
    a_tile = nl.load_transpose2d(a_tensor) 
    b_tile = nl.load(b_tensor)
    
    # Perform matrix multiplication with transposed input
    out_tile = nl.matmul(a_tile, b_tile, transpose_x=True)
    
    # Store result back to HBM
    nl.store(result, out_tile)

    return result