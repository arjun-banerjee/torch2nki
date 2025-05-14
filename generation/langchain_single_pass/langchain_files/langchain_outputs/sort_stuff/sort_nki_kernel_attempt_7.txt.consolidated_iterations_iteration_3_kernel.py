from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get the shape of the input tensor
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension index
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Copy input tensor to result tensor initially
    # We need to handle tensors of any size by processing them in tiles
    
    # Calculate the size of the dimension to sort
    sort_dim_size = shape[dim]
    
    # Determine the total number of slices to process
    # This is the product of all dimensions except the sort dimension
    num_slices = 1
    for i in range(ndim):
        if i != dim:
            num_slices *= shape[i]
    
    # Process the tensor in chunks that respect hardware limitations
    max_tile_size = nl.tile_size.pmax  # Maximum partition size
    
    # Create indices for all dimensions
    indices = []
    for i in range(ndim):
        if i == dim:
            # For the sort dimension, we'll use the full range
            indices.append(nl.arange(shape[i]))
        else:
            # For other dimensions, we'll create a placeholder index
            indices.append(0)
    
    # Process each slice
    for slice_idx in nl.affine_range(num_slices):
        # Calculate the multi-dimensional index for this slice
        remaining = slice_idx
        for i in range(ndim):
            if i != dim:
                div = 1
                for j in range(i+1, ndim):
                    if j != dim:
                        div *= shape[j]
                indices[i] = remaining // div
                remaining = remaining % div
        
        # Load the slice to sort
        slice_to_sort = nl.zeros((sort_dim_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
        
        # Create index for loading
        load_indices = []
        for i in range(ndim):
            if i == dim:
                load_indices.append(nl.arange(shape[i]))
            else:
                load_indices.append(indices[i])
        
        # Load the slice
        for pos in nl.affine_range(math.ceil(sort_dim_size / max_tile_size)):
            start_idx = pos * max_tile_size
            end_idx = min(start_idx + max_tile_size, sort_dim_size)
            size = end_idx - start_idx
            
            # Create index for this tile
            tile_indices = []
            for i in range(ndim):
                if i == dim:
                    tile_indices.append(nl.arange(start_idx, end_idx))
                else:
                    tile_indices.append(indices[i])
            
            # Load the tile
            tile_data = nl.load(a_tensor[tuple(tile_indices)])
            
            # Store in the slice buffer
            slice_to_sort[start_idx:end_idx] = tile_data
        
        # Sort the slice using bubble sort
        for i in nl.affine_range(sort_dim_size):
            for j in nl.affine_range(sort_dim_size - 1):
                # Compare adjacent elements
                a = slice_to_sort[j]
                b = slice_to_sort[j+1]
                
                # If a > b, swap them
                cond = nl.greater(a, b)
                slice_to_sort[j] = nl.where(cond, b, a)
                slice_to_sort[j+1] = nl.where(cond, a, b)
        
        # Store the sorted slice back
        for pos in nl.affine_range(math.ceil(sort_dim_size / max_tile_size)):
            start_idx = pos * max_tile_size
            end_idx = min(start_idx + max_tile_size, sort_dim_size)
            size = end_idx - start_idx
            
            # Create index for this tile
            store_indices = []
            for i in range(ndim):
                if i == dim:
                    store_indices.append(nl.arange(start_idx, end_idx))
                else:
                    store_indices.append(indices[i])
            
            # Store the sorted tile
            nl.store(result[tuple(store_indices)], slice_to_sort[start_idx:end_idx])
    
    return result