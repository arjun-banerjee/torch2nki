from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_mv(matrix, vector):
    # Initialize result array
    result = nl.ndarray((matrix.shape[0],), dtype=matrix.dtype, buffer=nl.shared_hbm)
    
    # Load matrix and vector into SBUF
    matrix_sb = nl.load(matrix)
    vector_sb = nl.load(vector)
    
    # Add dimension to vector to match matrix shape for matmul
    vector_expanded = nl.expand_dims(vector_sb, axis=1)
    
    # Compute matrix-vector product using matmul
    output = nl.matmul(matrix_sb, vector_expanded)
    
    # Store result back to HBM
    nl.store(result, output[:, 0])
    
    return result