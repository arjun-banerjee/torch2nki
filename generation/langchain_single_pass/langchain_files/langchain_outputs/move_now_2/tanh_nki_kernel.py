from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tanh(input_tensor):
    # Initialize a result array for holding the output
    result = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    # Load the input tensor into SBUF
    input_sb = nl.load(input_tensor)

    # Compute the exponential terms
    exp_input = nl.exp(nl.multiply(input_sb, 2))  # exp(2 * x)

    # Calculate numerator and denominator of tanh
    numerator = nl.subtract(exp_input, 1)  # exp(2 * x) - 1
    denominator = nl.add(exp_input, 1)      # exp(2 * x) + 1
    
    # Calculate tanh
    tanh_output = nl.divide(numerator, denominator)

    # Store the result back to HBM
    nl.store(result, tanh_output)

    return result