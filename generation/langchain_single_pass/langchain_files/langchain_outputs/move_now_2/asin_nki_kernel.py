from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_asin(input_tensor):
    # Initialize a result array with the same shape and dtype as the input tensor
    result = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    # Load input data from HBM
    input_tile = nl.load(input_tensor)

    # Initialize the output container
    output_tile = nl.zeros(input_tile.shape, dtype=input_tile.dtype)

    # Variables for Taylor series coefficients
    term = input_tile  # First term is just x
    n = nl.zeros(input_tile.shape, dtype=input_tile.dtype)  # This will act as 'n' in the series
    output_tile += term  # Adding the first term of the series

    for i in nl.affine_range(1, 100):  # Calculate more terms in the series
        n += 2  # Increment n by 2 for each term
        term = nl.multiply(term, nl.power(input_tile, 2))  # Update term = term * x^2
        term = nl.multiply(term, nl.power(n, -1))  # Divide by (2n + 1)
        output_tile += term  # Accumulate the terms into output

    # Store result back to HBM
    nl.store(result, output_tile)

    return result