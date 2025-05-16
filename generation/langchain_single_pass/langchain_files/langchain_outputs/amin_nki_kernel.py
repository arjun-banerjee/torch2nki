from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_amin(a_tensor):
    # Get the shape of the input tensor
    shape = a_tensor.shape
    
    # If the input is empty, return default value
    if any(dim == 0 for dim in shape):
        return nl.full((), float('inf'), dtype=a_tensor.dtype)
    
    # Handle scalar input
    if len(shape) == 0:
        return nl.load(a_tensor)
    
    # For multi-dimensional tensors, we'll flatten to 2D for processing
    # First dimension is partition dimension (limited by hardware)
    # All other dimensions are flattened into a single free dimension
    if len(shape) == 1:
        sz_p = shape[0]
        sz_f = 1
    else:
        sz_p = shape[0]
        sz_f = math.prod(shape[1:])
    
    # Calculate the number of tiles needed
    trip_count = math.ceil(sz_p / nl.tile_size.pmax)
    
    # Initialize result with maximum possible value
    result = nl.full((), float('inf'), dtype=a_tensor.dtype)
    
    # Generate the free dimension index array
    i_f = nl.arange(sz_f)[None, :]
    
    # Process the tensor in tiles
    for p in nl.affine_range(trip_count):
        # Generate tensor indices for the current tile
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load input data from external memory to on-chip memory
        # Only load up to the actual size of the tensor
        if len(shape) == 1:
            # For 1D tensors
            x_tile = nl.load(a_tensor[i_p[:, 0]], mask=(i_p[:, 0] < sz_p))
            # Reshape to 2D for consistent processing
            x_tile = x_tile[:, None]
        else:
            # For multi-dimensional tensors, we load with flattened indices
            x_tile = nl.load(a_tensor.reshape((sz_p, sz_f))[i_p, i_f], mask=(i_p < sz_p))
        
        # Find minimum value in this tile
        tile_min = nl.min(x_tile, axis=1)
        
        # Update the global minimum
        result = nl.minimum(result, tile_min)
    
    return result