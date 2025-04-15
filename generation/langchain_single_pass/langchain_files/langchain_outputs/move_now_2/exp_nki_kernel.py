from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_exp(input_tensor):
    # Determine the shape of the input tensor
    if len(input_tensor.shape) == 1:
        sz_p = input_tensor.shape[0]  # Single dimension
        sz_f = 1  # Setting features to 1 for consistency in processing
    elif len(input_tensor.shape) == 2:
        sz_p, sz_f = input_tensor.shape  # Two dimensions
    else:
        raise ValueError("Input tensor must be 1D or 2D")

    # Initialize result array with the same shape and dtype as input_tensor
    result = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    # Process the input tensor in tiles
    i_f = nl.arange(sz_f)[None, :]
    trip_count = (sz_p + nl.tile_size.pmax - 1) // nl.tile_size.pmax  # Ceil division

    for p in nl.affine_range(trip_count):
        # Generate indices for the input tensor
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]

        # Load input data from external memory to on-chip memory
        in_tile = nl.load(input_tensor[i_p, :] if len(input_tensor.shape) == 2 else input_tensor[i_p], mask=(i_p < sz_p))

        # Compute the exponential of the input tile
        exp_tile = nl.exp(in_tile)

        # Store the results back to the result tensor
        nl.store(result[i_p, :] if len(result.shape) == 2 else result[i_p], value=exp_tile, mask=(i_p < sz_p))

    return result