from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_atan(a_tensor):
    # Initialize the result array with the same shape and dtype as the input tensor
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Load the input tensor into on-chip memory (SBUF)
    input_tile = nl.load(a_tensor)

    # Compute the arctangent of the input tile using NKI's arctan function
    atan_output = nl.arctan(input_tile)

    # Store the result back to HBM from on-chip memory
    nl.store(result, atan_output)

    # Return the result
    return result