from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sum(a_tensor):
    # Initialize scalar result array in HBM
    result = nl.ndarray((), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load data into SBUF and transpose for axis 1 reduction
    input_sbuf = nl.load(a_tensor)
    transposed = nl.transpose(input_sbuf)
    
    # First sum reduces along rows
    row_sums = nl.sum(transposed, axis=1)
    # Second sum reduces the row sums to final scalar
    final_sum = nl.sum(row_sums, axis=1)
    
    # Store scalar result
    nl.store(result, final_sum)
    
    return result