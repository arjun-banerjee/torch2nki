from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy the input tensor to result
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Calculate sizes for processing
    size_to_sort = shape[dim]
    
    # Calculate total number of sort operations needed
    total_sort_operations = 1
    for i in range(ndim):
        if i != dim:
            total_sort_operations *= shape[i]
    
    # Determine pre_dim_size and post_dim_size
    pre_dim_size = 1
    for i in range(dim):
        pre_dim_size *= shape[i]
    
    post_dim_size = 1
    for i in range(dim + 1, ndim):
        post_dim_size *= shape[i]
    
    # Process in tiles to handle architecture limitations
    max_tile_size = nl.tile_size.pmax
    pre_tiles = math.ceil(pre_dim_size / max_tile_size)
    post_tiles = math.ceil(post_dim_size / max_tile_size)
    
    # Copy input to result first
    for pre_tile in nl.affine_range(pre_tiles):
        pre_offset = pre_tile * max_tile_size
        pre_size = min(max_tile_size, pre_dim_size - pre_offset)
        
        for post_tile in nl.affine_range(post_tiles):
            post_offset = post_tile * max_tile_size
            post_size = min(max_tile_size, post_dim_size - post_offset)
            
            # Create indices for this tile
            pre_indices = pre_offset + nl.arange(pre_size)
            post_indices = post_offset + nl.arange(post_size)
            
            # Convert flat indices to multidimensional indices
            multi_indices = []
            for pre_idx in nl.affine_range(pre_size):
                pre_idx_val = pre_offset + pre_idx
                pre_multi_idx = []
                temp_idx = pre_idx_val
                for i in range(dim):
                    idx_size = 1
                    for j in range(i + 1, dim):
                        idx_size *= shape[j]
                    pre_multi_idx.append(temp_idx // idx_size)
                    temp_idx = temp_idx % idx_size
                
                for sort_idx in nl.affine_range(size_to_sort):
                    for post_idx in nl.affine_range(post_size):
                        post_idx_val = post_offset + post_idx
                        post_multi_idx = []
                        temp_idx = post_idx_val
                        for i in range(dim + 1, ndim):
                            idx_size = 1
                            for j in range(i + 1, ndim):
                                idx_size *= shape[j]
                            post_multi_idx.append(temp_idx // idx_size)
                            temp_idx = temp_idx % idx_size
                        
                        # Construct full index
                        full_idx = pre_multi_idx + [sort_idx] + post_multi_idx
                        
                        # Load value from input tensor
                        if ndim == 1:
                            value = nl.load(a_tensor[sort_idx])
                            nl.store(result[sort_idx], value)
                        elif ndim == 2:
                            if dim == 0:
                                value = nl.load(a_tensor[sort_idx, post_idx_val])
                                nl.store(result[sort_idx, post_idx_val], value)
                            else:  # dim == 1
                                value = nl.load(a_tensor[pre_idx_val, sort_idx])
                                nl.store(result[pre_idx_val, sort_idx], value)
                        else:  # Higher dimensions handled through scalar operations
                            # This is a placeholder - we'll need more complex indexing for higher dims
                            # For now we just copy the input to output
                            if dim == 0:
                                value = nl.load(a_tensor[sort_idx])
                                nl.store(result[sort_idx], value)
                            else:
                                # This is a simplified approach - actual implementation would need proper indexing
                                pass
    
    # Now perform the bubble sort on each slice along the specified dimension
    for pre_tile in nl.affine_range(pre_tiles):
        pre_offset = pre_tile * max_tile_size
        pre_size = min(max_tile_size, pre_dim_size - pre_offset)
        
        for post_tile in nl.affine_range(post_tiles):
            post_offset = post_tile * max_tile_size
            post_size = min(max_tile_size, post_dim_size - post_offset)
            
            # For each slice to sort
            for pre_idx in nl.affine_range(pre_size):
                pre_idx_val = pre_offset + pre_idx
                
                for post_idx in nl.affine_range(post_size):
                    post_idx_val = post_offset + post_idx
                    
                    # Load the slice to sort into on-chip memory
                    slice_data = nl.zeros((size_to_sort,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                    
                    # Load the slice
                    for i in nl.affine_range(size_to_sort):
                        if ndim == 1:
                            slice_data[i] = nl.load(result[i])
                        elif ndim == 2:
                            if dim == 0:
                                slice_data[i] = nl.load(result[i, post_idx_val])
                            else:  # dim == 1
                                slice_data[i] = nl.load(result[pre_idx_val, i])
                        else:
                            # For higher dimensions, we would need proper indexing
                            # This is a simplified placeholder
                            pass
                    
                    # Perform bubble sort on the slice
                    for i in nl.affine_range(size_to_sort):
                        for j in nl.affine_range(size_to_sort - 1):
                            # Check if swap needed
                            if nl.greater(slice_data[j], slice_data[j + 1]):
                                # Swap values
                                temp = slice_data[j]
                                slice_data[j] = slice_data[j + 1]
                                slice_data[j + 1] = temp
                    
                    # Store the sorted slice back
                    for i in nl.affine_range(size_to_sort):
                        if ndim == 1:
                            nl.store(result[i], slice_data[i])
                        elif ndim == 2:
                            if dim == 0:
                                nl.store(result[i, post_idx_val], slice_data[i])
                            else:  # dim == 1
                                nl.store(result[pre_idx_val, i], slice_data[i])
                        else:
                            # For higher dimensions, we would need proper indexing
                            # This is a simplified placeholder
                            pass
    
    return result