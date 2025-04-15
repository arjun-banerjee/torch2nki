from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sin(a_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Load input tensor from HBM to SBUF
    input_tile = nl.load(a_tensor)

    # Compute sine using the NKI provided function
    sin_output = nl.sin(input_tile)

    # Store result back to HBM
    nl.store(result, sin_output)

    return result