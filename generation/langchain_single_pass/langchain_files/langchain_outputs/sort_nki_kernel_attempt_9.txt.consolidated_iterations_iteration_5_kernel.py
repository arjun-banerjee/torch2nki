from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape
    tensor_shape = a_tensor.shape
    ndim = len(tensor_shape)
    
    # Handle negative dimensions
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy the input tensor to result
    if ndim == 1:
        # For 1D tensor, we sort the entire array
        size = tensor_shape[0]
        
        # Define the maximum number of elements to process in one tile
        max_size = nl.tile_size.pmax
        trip_count = math.ceil(size / max_size)
        
        # Copy input to result first
        for p in nl.affine_range(trip_count):
            i_p = p * max_size + nl.arange(max_size)
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            nl.store(result[i_p], value=in_tile, mask=(i_p < size))
        
        # Bubble sort algorithm
        for i in range(size):
            for p in nl.affine_range(trip_count):
                i_p = p * max_size + nl.arange(max_size)
                
                # Load current tile
                current_tile = nl.load(result[i_p], mask=(i_p < size))
                
                # For each element in the tile, compare with the next element
                for j in range(max_size - 1):
                    # Only process valid indices
                    if p * max_size + j + 1 < size:
                        # Load the current and next element
                        curr = nl.load(result[p * max_size + j])
                        next_val = nl.load(result[p * max_size + j + 1])
                        
                        # Compare and swap if needed
                        is_greater = nl.greater(curr, next_val)
                        if is_greater:
                            nl.store(result[p * max_size + j], value=next_val)
                            nl.store(result[p * max_size + j + 1], value=curr)
                
                # Handle boundary between tiles
                if p < trip_count - 1 and (p + 1) * max_size < size:
                    # Compare last element of current tile with first element of next tile
                    last_curr = nl.load(result[(p + 1) * max_size - 1])
                    first_next = nl.load(result[(p + 1) * max_size])
                    
                    # Compare and swap if needed
                    is_greater = nl.greater(last_curr, first_next)
                    if is_greater:
                        nl.store(result[(p + 1) * max_size - 1], value=first_next)
                        nl.store(result[(p + 1) * max_size], value=last_curr)
    
    else:
        # For multi-dimensional tensors
        # Determine the length of the dimension to sort
        sort_dim_size = tensor_shape[dim]
        
        # Calculate the number of slices to sort
        total_slices = 1
        for d in range(ndim):
            if d != dim:
                total_slices *= tensor_shape[d]
        
        # Copy input to result first
        for i in range(total_slices):
            # Calculate multi-dimensional indices for the current slice
            indices = []
            temp_i = i
            for d in range(ndim):
                if d != dim:
                    dim_size = tensor_shape[d]
                    indices.append(temp_i % dim_size)
                    temp_i //= dim_size
                else:
                    indices.append(0)  # Placeholder, will be filled in the inner loop
            
            # Copy the entire slice
            for j in range(sort_dim_size):
                indices[dim] = j
                # Convert indices list to tuple for indexing
                idx_tuple = tuple(indices)
                val = nl.load(a_tensor[idx_tuple])
                nl.store(result[idx_tuple], value=val)
            
            # Sort this slice using bubble sort
            for k in range(sort_dim_size):
                for j in range(sort_dim_size - 1):
                    indices[dim] = j
                    idx_j = tuple(indices)
                    
                    indices[dim] = j + 1
                    idx_j_plus_1 = tuple(indices)
                    
                    val_j = nl.load(result[idx_j])
                    val_j_plus_1 = nl.load(result[idx_j_plus_1])
                    
                    # Compare and swap if needed
                    is_greater = nl.greater(val_j, val_j_plus_1)
                    if is_greater:
                        nl.store(result[idx_j], value=val_j_plus_1)
                        nl.store(result[idx_j_plus_1], value=val_j)
    
    return result