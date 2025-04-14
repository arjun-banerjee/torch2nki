from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_prod(a_tensor):
    # Handle empty tensor case
    if len(a_tensor.shape) == 0:
        # For scalar input, just return it
        return nl.load(a_tensor)
    
    # Initialize result as 1 (multiplicative identity)
    result = nl.ones((), dtype=a_tensor.dtype, buffer=nl.psum)
    
    # Reshape tensor into 2D for easier processing
    # We'll use a row-major layout where each row is a tile
    total_elements = 1
    for dim in a_tensor.shape:
        total_elements *= dim
    
    # If tensor is empty (has a dimension of size 0), return 1
    if total_elements == 0:
        return result
    
    # Calculate number of tiles needed
    max_tile_size = nl.tile_size.pmax
    num_tiles = math.ceil(total_elements / max_tile_size)
    
    # Process tensor in tiles
    for i in nl.affine_range(num_tiles):
        # Calculate indices for this tile
        indices = i * max_tile_size + nl.arange(max_tile_size)
        
        # Create mask for valid elements (to handle the last tile)
        mask = indices < total_elements
        
        # Create flattened view of tensor for loading
        # We need to compute flat indices from tile indices
        if len(a_tensor.shape) == 1:
            # For 1D tensor, indices are directly usable
            tile_data = nl.load(a_tensor[indices], mask=mask)
        else:
            # For multi-dimensional tensors, we need to create a flattened view
            # This is a simplified approach - in real scenarios, we would need to
            # calculate the actual multi-dimensional indices
            flat_view = a_tensor.reshape((total_elements,))
            tile_data = nl.load(flat_view[indices], mask=mask)
        
        # Multiply all elements in the tile together
        # First multiply elements within the tile
        tile_prod = nl.prod(tile_data, axis=0, mask=mask)
        
        # Then multiply with the running product
        result = nl.multiply(result, tile_prod)
    
    return result