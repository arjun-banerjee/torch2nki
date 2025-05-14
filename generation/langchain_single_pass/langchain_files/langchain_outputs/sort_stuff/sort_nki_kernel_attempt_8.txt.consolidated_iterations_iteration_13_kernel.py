from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension indexing
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy input to result
    tensor_shape = a_tensor.shape
    ndims = len(tensor_shape)
    
    # Handle 1D tensor case
    if ndims == 1:
        # For 1D tensor, we'll sort the entire array
        # Calculate the number of elements to process
        n = tensor_shape[0]
        
        # Process in tiles to respect hardware limitations
        max_tile_size = nl.tile_size.pmax
        
        # First, load the entire tensor into result
        for p_offset in nl.affine_range(math.ceil(n / max_tile_size)):
            # Create indices for current tile
            i_p = nl.arange(max_tile_size)
            
            # Calculate actual indices and load with masking
            actual_indices = p_offset * max_tile_size + i_p
            
            # Load the current tile with masking to handle boundaries
            tile_data = nl.load(a_tensor[actual_indices], mask=(actual_indices < n))
            
            # Store to result
            nl.store(result[actual_indices], value=tile_data, mask=(actual_indices < n))
        
        # Bubble sort implementation
        for i in nl.affine_range(n):
            for j_offset in nl.affine_range(math.ceil((n-i-1) / max_tile_size)):
                # Process in tiles
                j_base = j_offset * max_tile_size
                j_indices = j_base + nl.arange(max_tile_size)
                
                # Load current elements with masking
                valid_j = (j_indices < (n-i-1))
                current = nl.load(result[j_indices], mask=valid_j)
                
                # Load next elements with masking
                next_indices = j_indices + 1
                next_val = nl.load(result[next_indices], mask=valid_j)
                
                # Compare and swap if needed
                swap_needed = nl.greater(current, next_val)
                
                # Where swap is needed, update values
                new_current = nl.where(swap_needed, next_val, current)
                new_next = nl.where(swap_needed, current, next_val)
                
                # Store the updated values
                nl.store(result[j_indices], value=new_current, mask=valid_j)
                nl.store(result[next_indices], value=new_next, mask=valid_j)
    
    # Handle multi-dimensional tensor case
    else:
        # Determine the size of the dimension to sort
        sort_dim_size = tensor_shape[dim]
        
        # Calculate the number of slices to sort
        num_slices = 1
        for i in range(ndims):
            if i != dim:
                num_slices *= tensor_shape[i]
        
        # Process each slice separately
        max_slice_size = nl.tile_size.pmax
        max_sort_size = nl.tile_size.pmax
        
        # First, copy the input tensor to result
        for slice_offset in nl.affine_range(math.ceil(num_slices / max_slice_size)):
            slice_base = slice_offset * max_slice_size
            
            for sort_offset in nl.affine_range(math.ceil(sort_dim_size / max_sort_size)):
                sort_base = sort_offset * max_sort_size
                
                # Create multi-dimensional indices based on flat slice index
                for slice_idx in nl.affine_range(max_slice_size):
                    actual_slice_idx = slice_base + slice_idx
                    
                    if actual_slice_idx < num_slices:
                        # Create indices for non-sort dimensions
                        indices = []
                        temp_idx = actual_slice_idx
                        for i in range(ndims):
                            if i != dim:
                                dim_size = tensor_shape[i]
                                idx = temp_idx % dim_size
                                temp_idx = temp_idx // dim_size
                                indices.append(idx)
                            else:
                                indices.append(0)  # Placeholder for sort dimension
                        
                        # Process the sort dimension
                        for k in nl.affine_range(max_sort_size):
                            sort_idx = sort_base + k
                            
                            if sort_idx < sort_dim_size:
                                # Update sort dimension index
                                indices[dim] = sort_idx
                                
                                # Create index tuple for loading/storing
                                # This is done by loading individual elements
                                elem = nl.load(a_tensor[tuple(indices)])
                                nl.store(result[tuple(indices)], value=elem)
        
        # Now sort each slice along the specified dimension
        for slice_offset in nl.affine_range(math.ceil(num_slices / max_slice_size)):
            slice_base = slice_offset * max_slice_size
            
            for slice_idx in nl.affine_range(max_slice_size):
                actual_slice_idx = slice_base + slice_idx
                
                if actual_slice_idx < num_slices:
                    # Create indices for non-sort dimensions
                    indices = []
                    temp_idx = actual_slice_idx
                    for i in range(ndims):
                        if i != dim:
                            dim_size = tensor_shape[i]
                            idx = temp_idx % dim_size
                            temp_idx = temp_idx // dim_size
                            indices.append(idx)
                        else:
                            indices.append(0)  # Placeholder for sort dimension
                    
                    # Bubble sort implementation for this slice
                    for i in nl.affine_range(sort_dim_size):
                        for j in nl.affine_range(sort_dim_size - i - 1):
                            # Create indices for current and next elements
                            curr_indices = list(indices)
                            next_indices = list(indices)
                            curr_indices[dim] = j
                            next_indices[dim] = j + 1
                            
                            # Load values
                            curr_val = nl.load(result[tuple(curr_indices)])
                            next_val = nl.load(result[tuple(next_indices)])
                            
                            # Compare and swap if needed
                            if nl.greater(curr_val, next_val).item():
                                nl.store(result[tuple(curr_indices)], value=next_val)
                                nl.store(result[tuple(next_indices)], value=curr_val)
    
    return result