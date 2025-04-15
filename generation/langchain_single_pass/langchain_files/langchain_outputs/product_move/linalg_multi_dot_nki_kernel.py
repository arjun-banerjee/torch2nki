from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_linalg_multi_dot(matrix_list):
    # Initialize result with first matrix shape
    result = nl.ndarray(matrix_list[0].shape, dtype=matrix_list[0].dtype, buffer=nl.shared_hbm)
    
    # Load first matrix into temporary buffer
    temp = nl.load(matrix_list[0])
    
    # Multiply remaining matrices sequentially
    for i in range(1, len(matrix_list)):
        next_matrix = nl.load(matrix_list[i])
        temp = nl.matmul(nl.transpose(temp), next_matrix, transpose_x=True)
    
    # Store final result
    nl.store(result, temp)
    
    return result