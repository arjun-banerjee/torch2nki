from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_acos(a_tensor):
    # Load a_tensor from HBM to SBUF for processing
    a_tile = nl.load(a_tensor)

    # Validate input range using masking
    mask_invalid = nl.less_equal(a_tile, -1.0) | nl.greater_equal(a_tile, 1.0)

    # Initialize result array without any initial values
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Compute acos using Taylor series for valid inputs
    # We will use the Taylor series expansion for acos:
    # acos(x) = pi/2 - sqrt(1-x^2) + x*(1-x^2)*(3/8 + x^2*(-1/48 + x^2*...) )
    
    pi_over_two = 1.57079632679
    x_squared = nl.multiply(a_tile, a_tile)
    
    # Supervised calculation only for valid inputs
    acos_values = nl.zeros(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Compute valid values
    valid_acos = nl.logical_not(mask_invalid)
    valid_values = nl.load(a_tile[valid_acos])
    
    acos_values[valid_acos] = pi_over_two - nl.sqrt(nl.subtract(1.0, nl.multiply(valid_values, valid_values)))

    # Store the computed values back to the result tensor
    nl.store(result, acos_values)

    return result