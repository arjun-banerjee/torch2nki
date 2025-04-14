from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_logsumexp(a_tensor, dim=-1):
    # Get dimensions
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle default case (dim=-1)
    if dim < 0:
        dim = ndim + dim
        
    # Calculate output shape by removing the dimension we're reducing over
    output_shape = shape[:dim] + shape[dim+1:]
    
    # Initialize result array
    result = nl.ndarray(output_shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # For now, we support reduction along the last dimension only
    if dim != ndim - 1:
        raise ValueError("Currently only supports reduction along the last dimension")
        
    # Handle 2D tensors with reduction along last dimension
    sz_p, sz_f = shape[0], shape[1]  # Assuming 2D tensor [batch, features]
    trip_count = math.ceil(sz_p / nl.tile_size.pmax)
    
    # Generate the free dimension index array
    i_f = nl.arange(sz_f)[None, :]
    
    # Process the tensor in tiles
    for p in nl.affine_range(trip_count):
        # Generate tensor indices for the current tile
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load input data from external memory to on-chip memory
        x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
        
        # Find the maximum value along the last dimension
        max_val = nl.max(x_tile, axis=1, keepdims=True)
        
        # Subtract the maximum value and compute exponentials for numerical stability
        exp_vals = nl.exp(nl.subtract(x_tile, max_val))
        
        # Sum the exponentials along the last dimension
        sum_exp = nl.sum(exp_vals, axis=1, keepdims=False)
        
        # Calculate the log of the sum and add the max value back
        out_tile = nl.add(nl.log(sum_exp), max_val[:, 0])
        
        # Store the results back to external memory
        nl.store(result[i_p[:, 0]], value=out_tile, mask=(i_p[:, 0] < sz_p))
    
    return result