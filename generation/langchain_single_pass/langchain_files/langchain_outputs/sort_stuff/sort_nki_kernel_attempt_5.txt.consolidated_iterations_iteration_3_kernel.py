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
    
    # Copy input tensor to result first
    tensor_shape = a_tensor.shape
    ndim = len(tensor_shape)
    
    # Determine the size of the dimension to sort
    sort_dim_size = tensor_shape[dim]
    
    # Calculate total elements to process
    total_elements = 1
    for i in range(ndim):
        if i != dim:
            total_elements *= tensor_shape[i]
    
    # Determine tiling strategy
    max_tile_size = nl.tile_size.pmax
    trip_count = math.ceil(total_elements / max_tile_size)
    
    # First copy the input to result
    for p in nl.affine_range(trip_count):
        # Generate indices for current tile
        indices = p * max_tile_size + nl.arange(max_tile_size)
        
        # Create mask for valid indices
        mask = indices < total_elements
        
        # Convert flat index to multi-dimensional index
        multi_indices = []
        temp_indices = indices
        for i in range(ndim-1, -1, -1):
            if i == dim:
                multi_indices.insert(0, nl.arange(tensor_shape[i]))
                continue
                
            div = 1
            for j in range(i+1, ndim):
                if j != dim:
                    div *= tensor_shape[j]
            
            idx = temp_indices // div
            temp_indices = temp_indices % div
            multi_indices.insert(0, idx)
        
        # Load input data, copy to result
        in_tile = nl.load(a_tensor[tuple(multi_indices)], mask=mask)
        nl.store(result[tuple(multi_indices)], in_tile, mask=mask)
    
    # Now perform the bubble sort along the specified dimension
    for i in nl.affine_range(sort_dim_size):
        for j in nl.affine_range(sort_dim_size - 1):
            # Process in tiles
            for p in nl.affine_range(trip_count):
                # Generate indices for current tile
                indices = p * max_tile_size + nl.arange(max_tile_size)
                
                # Create mask for valid indices
                mask = indices < total_elements
                
                # Convert flat index to multi-dimensional index
                multi_indices = []
                temp_indices = indices
                for k in range(ndim-1, -1, -1):
                    if k == dim:
                        multi_indices.insert(0, j)
                        continue
                        
                    div = 1
                    for l in range(k+1, ndim):
                        if l != dim:
                            div *= tensor_shape[l]
                    
                    idx = temp_indices // div
                    temp_indices = temp_indices % div
                    multi_indices.insert(0, idx)
                
                # Get indices for j and j+1 positions
                j_indices = list(multi_indices)
                jp1_indices = list(multi_indices)
                j_indices[dim] = j
                jp1_indices[dim] = j + 1
                
                # Load values at j and j+1
                val_j = nl.load(result[tuple(j_indices)], mask=mask)
                val_jp1 = nl.load(result[tuple(jp1_indices)], mask=mask)
                
                # Compare and swap if needed
                swap_needed = nl.greater(val_j, val_jp1)
                
                # Create temporary tiles for swapping
                temp_j = nl.zeros_like(val_j)
                temp_jp1 = nl.zeros_like(val_jp1)
                
                # Perform the swap
                temp_j = nl.where(swap_needed, val_jp1, val_j)
                temp_jp1 = nl.where(swap_needed, val_j, val_jp1)
                
                # Store back the results
                nl.store(result[tuple(j_indices)], temp_j, mask=mask)
                nl.store(result[tuple(jp1_indices)], temp_jp1, mask=mask)
    
    return result