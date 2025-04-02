from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_softmax(a_tensor, b_tensor):
    # Get shapes of input tensors
    m, n = a_tensor.shape
    p, q = b_tensor.shape
    
    # Initialize result array with proper dimensions
    result = nl.ndarray((m*p, n*q), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Process each block of the Kronecker product
    for i in nl.affine_range(m):
        for j in nl.affine_range(n):
            # Load scalar value from first tensor
            a_val = nl.load(a_tensor[i, j], dtype=nl.float32)
            
            # Load block from second tensor
            b_block = nl.load(b_tensor, dtype=nl.float32)
            
            # Compute block of Kronecker product
            block = nl.multiply(a_val, b_block)
            
            # Store result in appropriate position
            for k in nl.affine_range(p):
                for l in nl.affine_range(q):
                    nl.store(result[i*p + k, j*q + l], block[k, l])
    
    return result
