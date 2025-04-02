from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sum(input_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray((), dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    
    # Initialize accumulator in psum buffer
    acc = nl.zeros((1,1), dtype=input_tensor.dtype, buffer=nl.psum)
    
    # Accumulate sum using loop
    for i in nl.affine_range(input_tensor.shape[0]):
        val = nl.load(input_tensor[i])
        acc += val
    
    # Store final result
    nl.store(result, acc[0,0])
    
    return result