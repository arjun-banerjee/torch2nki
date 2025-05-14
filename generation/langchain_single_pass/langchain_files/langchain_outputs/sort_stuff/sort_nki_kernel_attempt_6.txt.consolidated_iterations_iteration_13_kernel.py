from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimensions
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Get shape of input tensor
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Initialize result arrays with same shape and dtype as input
    values = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    indices = nl.ndarray(a_tensor.shape, dtype=nl.int32, buffer=nl.shared_hbm)
    
    # Special handling for 1D tensors
    if ndim == 1:
        # Get tensor size
        size = shape[0]
        
        # Maximum tile size for processing
        tile_size = min(size, nl.tile_size.pmax)
        trip_count = math.ceil(size / tile_size)
        
        # Process the tensor in tiles
        for p in nl.affine_range(trip_count):
            # Calculate indices for current tile
            start_idx = p * tile_size
            end_idx = min(start_idx + tile_size, size)
            valid_size = end_idx - start_idx
            
            # Load current tile
            i_p = nl.arange(tile_size)
            tile = nl.load(a_tensor[start_idx:end_idx], mask=(i_p < valid_size))
            
            # Initialize indices for current tile
            idx_tile = nl.arange(tile_size) + start_idx
            
            # Bubble sort implementation
            for i in range(valid_size):
                for j in range(valid_size - i - 1):
                    # Compare adjacent elements
                    cond = nl.greater(tile[j], tile[j+1])
                    
                    # Swap values if needed
                    tile_j = tile[j]
                    tile_j1 = tile[j+1]
                    tile[j] = nl.where(cond, tile_j1, tile_j)
                    tile[j+1] = nl.where(cond, tile_j, tile_j1)
                    
                    # Swap indices if needed
                    idx_j = idx_tile[j]
                    idx_j1 = idx_tile[j+1]
                    idx_tile[j] = nl.where(cond, idx_j1, idx_j)
                    idx_tile[j+1] = nl.where(cond, idx_j, idx_j1)
            
            # Store results back to output arrays
            nl.store(values[start_idx:end_idx], tile[:valid_size], mask=(i_p < valid_size))
            nl.store(indices[start_idx:end_idx], idx_tile[:valid_size], mask=(i_p < valid_size))
        
        return values, indices
    
    # For multi-dimensional tensors
    else:
        # Calculate the number of slices to sort
        num_slices = 1
        for i in range(ndim):
            if i != dim:
                num_slices *= shape[i]
                
        # Reshape the tensor to (num_slices, shape[dim])
        # We'll process each slice independently
        
        # Size of dimension to sort along
        sort_dim_size = shape[dim]
        
        # Maximum tile size for processing
        tile_size = min(sort_dim_size, nl.tile_size.pmax)
        
        # Process each slice
        slice_shape_pre = shape[:dim]
        slice_shape_post = shape[dim+1:]
        
        # Initialize indices for all dimensions
        idx_ranges = []
        for i in range(ndim):
            if i == dim:
                continue
            idx_ranges.append(nl.arange(shape[i]))
        
        # For each possible index combination (excluding sort dimension)
        for pre_idx in nl.affine_range(math.prod(slice_shape_pre) if slice_shape_pre else 1):
            for post_idx in nl.affine_range(math.prod(slice_shape_post) if slice_shape_post else 1):
                # Extract pre and post indices
                pre_indices = []
                tmp_pre_idx = pre_idx
                for i in range(len(slice_shape_pre)):
                    pre_indices.append(tmp_pre_idx % slice_shape_pre[-(i+1)])
                    tmp_pre_idx //= slice_shape_pre[-(i+1)]
                pre_indices.reverse()
                
                post_indices = []
                tmp_post_idx = post_idx
                for i in range(len(slice_shape_post)):
                    post_indices.append(tmp_post_idx % slice_shape_post[-(i+1)])
                    tmp_post_idx //= slice_shape_post[-(i+1)]
                post_indices.reverse()
                
                # Load the entire slice to sort
                slice_data = nl.zeros((sort_dim_size,), dtype=a_tensor.dtype)
                slice_indices = nl.arange(sort_dim_size)
                
                # Load the slice in tiles if needed
                for t in nl.affine_range(math.ceil(sort_dim_size / tile_size)):
                    start_idx = t * tile_size
                    end_idx = min(start_idx + tile_size, sort_dim_size)
                    valid_size = end_idx - start_idx
                    
                    i_p = nl.arange(tile_size)
                    
                    # Create index tuples for loading
                    if ndim == 2:
                        if dim == 0:
                            tile_data = nl.load(a_tensor[start_idx:end_idx, post_indices[0]], mask=(i_p < valid_size))
                        else:  # dim == 1
                            tile_data = nl.load(a_tensor[pre_indices[0], start_idx:end_idx], mask=(i_p < valid_size))
                    elif ndim == 3:
                        if dim == 0:
                            tile_data = nl.load(a_tensor[start_idx:end_idx, post_indices[0], post_indices[1]], mask=(i_p < valid_size))
                        elif dim == 1:
                            tile_data = nl.load(a_tensor[pre_indices[0], start_idx:end_idx, post_indices[0]], mask=(i_p < valid_size))
                        else:  # dim == 2
                            tile_data = nl.load(a_tensor[pre_indices[0], pre_indices[1], start_idx:end_idx], mask=(i_p < valid_size))
                    
                    # Store into our slice buffer
                    for j in range(valid_size):
                        slice_data[start_idx + j] = tile_data[j]
                
                # Bubble sort the slice
                for i in range(sort_dim_size):
                    for j in range(sort_dim_size - i - 1):
                        # Compare adjacent elements
                        cond = nl.greater(slice_data[j], slice_data[j+1])
                        
                        # Swap values if needed
                        val_j = slice_data[j]
                        val_j1 = slice_data[j+1]
                        slice_data[j] = nl.where(cond, val_j1, val_j)
                        slice_data[j+1] = nl.where(cond, val_j, val_j1)
                        
                        # Swap indices if needed
                        idx_j = slice_indices[j]
                        idx_j1 = slice_indices[j+1]
                        slice_indices[j] = nl.where(cond, idx_j1, idx_j)
                        slice_indices[j+1] = nl.where(cond, idx_j, idx_j1)
                
                # Store the sorted slice back in tiles if needed
                for t in nl.affine_range(math.ceil(sort_dim_size / tile_size)):
                    start_idx = t * tile_size
                    end_idx = min(start_idx + tile_size, sort_dim_size)
                    valid_size = end_idx - start_idx
                    
                    i_p = nl.arange(tile_size)
                    
                    # Create tiles for storing
                    tile_values = nl.zeros((tile_size,), dtype=a_tensor.dtype)
                    tile_indices = nl.zeros((tile_size,), dtype=nl.int32)
                    
                    # Copy from slice to tiles
                    for j in range(valid_size):
                        tile_values[j] = slice_data[start_idx + j]
                        tile_indices[j] = slice_indices[start_idx + j]
                    
                    # Store back to output arrays
                    if ndim == 2:
                        if dim == 0:
                            nl.store(values[start_idx:end_idx, post_indices[0]], tile_values, mask=(i_p < valid_size))
                            nl.store(indices[start_idx:end_idx, post_indices[0]], tile_indices, mask=(i_p < valid_size))
                        else:  # dim == 1
                            nl.store(values[pre_indices[0], start_idx:end_idx], tile_values, mask=(i_p < valid_size))
                            nl.store(indices[pre_indices[0], start_idx:end_idx], tile_indices, mask=(i_p < valid_size))
                    elif ndim == 3:
                        if dim == 0:
                            nl.store(values[start_idx:end_idx, post_indices[0], post_indices[1]], tile_values, mask=(i_p < valid_size))
                            nl.store(indices[start_idx:end_idx, post_indices[0], post_indices[1]], tile_indices, mask=(i_p < valid_size))
                        elif dim == 1:
                            nl.store(values[pre_indices[0], start_idx:end_idx, post_indices[0]], tile_values, mask=(i_p < valid_size))
                            nl.store(indices[pre_indices[0], start_idx:end_idx, post_indices[0]], tile_indices, mask=(i_p < valid_size))
                        else:  # dim == 2
                            nl.store(values[pre_indices[0], pre_indices[1], start_idx:end_idx], tile_values, mask=(i_p < valid_size))
                            nl.store(indices[pre_indices[0], pre_indices[1], start_idx:end_idx], tile_indices, mask=(i_p < valid_size))
        
        return values, indices