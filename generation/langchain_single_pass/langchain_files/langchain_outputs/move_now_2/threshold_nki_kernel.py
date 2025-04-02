from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_threshold(input_tensor, threshold, value):
    # Initialize the result array
    result = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    # Create a dummy variable for the computation
    dummy = nl.load(input_tensor)

    # Apply the threshold: if element > threshold, keep it; otherwise, replace with value
    condition = nl.greater(dummy, threshold)
    thresholded_output = nl.where(condition, dummy, value)

    # Store the result back to HBM
    nl.store(result, thresholded_output)

    return result