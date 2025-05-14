from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension indexing
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy input tensor to result
    # Process in tiles to respect hardware limitations
    if ndim == 1:
        # For 1D tensors, dim must be 0
        size = shape[0]
        # Process in tiles of maximum size nl.tile_size.pmax
        trip_count = math.ceil(size / nl.tile_size.pmax)
        for i in nl.affine_range(trip_count):
            # Calculate indices for current tile
            start_idx = i * nl.tile_size.pmax
            # Create index array for loading
            indices = nl.arange(nl.tile_size.pmax)
            # Load data with masking to handle boundary
            tile = nl.load(a_tensor[start_idx + indices], mask=(start_idx + indices < size))
            # Sort the tile using bubble sort
            for j in nl.static_range(nl.tile_size.pmax - 1):
                for k in nl.static_range(nl.tile_size.pmax - j - 1):
                    # Compare adjacent elements
                    is_greater = nl.greater(tile[k], tile[k+1])
                    # Swap if needed using conditional selection
                    temp_k = tile[k]
                    temp_k1 = tile[k+1]
                    # Only perform swap when k+1 is in bounds and is_greater is true
                    mask_valid = (k+1 < nl.tile_size.pmax) & (start_idx + k + 1 < size)
                    tile[k] = nl.where(is_greater & mask_valid, temp_k1, temp_k)
                    tile[k+1] = nl.where(is_greater & mask_valid, temp_k, temp_k1)
            
            # Store the sorted tile back to result
            nl.store(result[start_idx + indices], tile, mask=(start_idx + indices < size))
    else:
        # For multi-dimensional tensors, handle the specified dimension
        # Get the size of the dimension to sort
        dim_size = shape[dim]
        
        # Calculate number of slices to process
        # Each slice is a 1D array along the sorting dimension
        num_slices = 1
        for i in range(ndim):
            if i != dim:
                num_slices *= shape[i]
        
        # Process slices in tiles to respect hardware limitations
        trip_count = math.ceil(num_slices / nl.tile_size.pmax)
        
        for slice_idx in nl.affine_range(trip_count):
            # Calculate base indices for the current batch of slices
            base_slice_idx = slice_idx * nl.tile_size.pmax
            
            # Process each slice in the current batch
            for i in nl.static_range(nl.tile_size.pmax):
                # Skip if beyond the number of slices
                if base_slice_idx + i >= num_slices:
                    continue
                
                # Calculate multi-dimensional indices for this slice
                slice_indices = []
                remaining_idx = base_slice_idx + i
                for d in range(ndim):
                    if d != dim:
                        # Calculate index for this dimension
                        dim_size_d = shape[d]
                        idx_d = remaining_idx % dim_size_d
                        remaining_idx = remaining_idx // dim_size_d
                        slice_indices.append(idx_d)
                    else:
                        # For the sorting dimension, we'll use the full range
                        slice_indices.append(slice(0, dim_size))
                
                # Load the slice data
                # We need to convert slice_indices to proper indexing format
                if ndim == 2:
                    if dim == 0:
                        slice_data = nl.load(a_tensor[:, slice_indices[1]])
                    else:  # dim == 1
                        slice_data = nl.load(a_tensor[slice_indices[0], :])
                elif ndim == 3:
                    if dim == 0:
                        slice_data = nl.load(a_tensor[:, slice_indices[1], slice_indices[2]])
                    elif dim == 1:
                        slice_data = nl.load(a_tensor[slice_indices[0], :, slice_indices[2]])
                    else:  # dim == 2
                        slice_data = nl.load(a_tensor[slice_indices[0], slice_indices[1], :])
                
                # Sort the slice data using bubble sort
                for j in nl.static_range(dim_size - 1):
                    for k in nl.static_range(dim_size - j - 1):
                        # Compare adjacent elements
                        is_greater = nl.greater(slice_data[k], slice_data[k+1])
                        # Swap if needed
                        temp_k = slice_data[k]
                        temp_k1 = slice_data[k+1]
                        slice_data[k] = nl.where(is_greater, temp_k1, temp_k)
                        slice_data[k+1] = nl.where(is_greater, temp_k, temp_k1)
                
                # Store the sorted slice back to result
                if ndim == 2:
                    if dim == 0:
                        nl.store(result[:, slice_indices[1]], slice_data)
                    else:  # dim == 1
                        nl.store(result[slice_indices[0], :], slice_data)
                elif ndim == 3:
                    if dim == 0:
                        nl.store(result[:, slice_indices[1], slice_indices[2]], slice_data)
                    elif dim == 1:
                        nl.store(result[slice_indices[0], :, slice_indices[2]], slice_data)
                    else:  # dim == 2
                        nl.store(result[slice_indices[0], slice_indices[1], :], slice_data)
    
    return result