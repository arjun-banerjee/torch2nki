from neuronxcc import nki
import neuronxcc.nki.language as nl


def nki_vector_softmax(input_tensor):
    # Ensure input_tensor is 1D
    if input_tensor.ndim != 1:
        raise ValueError("Input tensor must be 1D.")

    # Get the shape of the input tensor
    input_size = input_tensor.shape[0]

    # Initialize a dummy variable for storing  exponential values
    exp_values = nl.zeros((input_size,), dtype=input_tensor.dtype)
    
    # Compute e^x for each element
    for i in nl.affine_range(input_size):
        exp_values[i] = nl.exp(nl.load(input_tensor[i]))

    # Calculate the sum of the exponential values
    sum_exp = nl.sum(exp_values, axis=0)

    # Initialize a result variable to store softmax output
    result = nl.zeros((input_size,), dtype=input_tensor.dtype)

    # Calculate softmax by dividing exponentials by the sum
    for i in nl.affine_range(input_size):
        result[i] = exp_values[i] / sum_exp  # This will broadcast the sum for each element

    return result