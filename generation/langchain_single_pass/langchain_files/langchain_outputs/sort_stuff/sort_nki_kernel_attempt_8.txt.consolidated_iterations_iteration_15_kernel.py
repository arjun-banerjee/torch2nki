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
    
    # Get tensor shape
    tensor_shape = a_tensor.shape
    ndims = len(tensor_shape)
    
    # Special case for 1D tensor
    if ndims == 1:
        # Calculate the number of elements to sort
        n = tensor_shape[0]
        
        # Process in tiles to handle large tensors
        tile_size = min(n, nl.tile_size.pmax)
        
        # First copy input to result
        for p_offset in nl.affine_range(math.ceil(n / tile_size)):
            # Calculate indices for current tile
            start_idx = p_offset * tile_size
            
            # Load input data
            i_p = nl.arange(tile_size)
            in_tile = nl.load(a_tensor[start_idx + i_p], mask=(start_idx + i_p < n))
            
            # Store to result
            nl.store(result[start_idx + i_p], value=in_tile, mask=(start_idx + i_p < n))
        
        # Bubble sort implementation
        for i in range(n):
            for j_offset in nl.affine_range(math.ceil((n - i - 1) / tile_size)):
                j_start = j_offset * tile_size
                
                # Load current elements
                j_indices = nl.arange(tile_size)
                curr_indices = j_start + j_indices
                next_indices = j_start + j_indices + 1
                
                curr_vals = nl.load(result[curr_indices], mask=(curr_indices < n - i - 1))
                next_vals = nl.load(result[next_indices], mask=(next_indices < n - i - 1))
                
                # Compare and swap if needed
                swap_needed = nl.greater(curr_vals, next_vals)
                
                # Update values
                new_curr = nl.where(swap_needed, next_vals, curr_vals)
                new_next = nl.where(swap_needed, curr_vals, next_vals)
                
                # Store updated values
                nl.store(result[curr_indices], value=new_curr, mask=(curr_indices < n - i - 1))
                nl.store(result[next_indices], value=new_next, mask=(next_indices < n - i - 1))
    
    # For multi-dimensional tensors
    else:
        # First copy input to result
        total_elements = 1
        for i in range(ndims):
            total_elements *= tensor_shape[i]
        
        # Copy input to result using tiling
        tile_size = min(total_elements, nl.tile_size.pmax)
        for p_offset in nl.affine_range(math.ceil(total_elements / tile_size)):
            # Calculate linear indices for current tile
            linear_start = p_offset * tile_size
            linear_indices = linear_start + nl.arange(tile_size)
            
            # Convert linear indices to multi-dimensional indices
            multi_indices = []
            remaining_indices = linear_indices
            
            # Load and store each element
            # We need to handle each element individually for multi-dimensional tensors
            for i in range(min(tile_size, total_elements - linear_start)):
                if linear_start + i >= total_elements:
                    break
                
                # Calculate multi-dimensional index for this element
                idx = linear_start + i
                multi_idx = []
                temp_idx = idx
                
                for d in range(ndims-1, -1, -1):
                    # Calculate stride for this dimension
                    stride = 1
                    for d2 in range(d+1, ndims):
                        stride *= tensor_shape[d2]
                    
                    # Calculate index for this dimension
                    dim_idx = temp_idx // stride
                    multi_idx.insert(0, dim_idx)
                    temp_idx = temp_idx % stride
                
                # Load value from input tensor
                val = nl.load(a_tensor[tuple(multi_idx)])
                
                # Store to result tensor
                nl.store(result[tuple(multi_idx)], value=val)
        
        # Sort along the specified dimension
        dim_size = tensor_shape[dim]
        
        # Calculate number of slices to sort
        num_slices = 1
        for i in range(ndims):
            if i != dim:
                num_slices *= tensor_shape[i]
        
        # Process each slice
        for slice_idx in range(num_slices):
            # Convert slice_idx to multi-dimensional index
            slice_multi_idx = []
            temp_idx = slice_idx
            
            # Skip the sort dimension in the index calculation
            dim_sizes = []
            for d in range(ndims):
                if d != dim:
                    dim_sizes.append(tensor_shape[d])
            
            # Calculate indices for dimensions before and after the sort dimension
            for d in range(ndims):
                if d != dim:
                    # Calculate stride for this dimension
                    stride = 1
                    idx_dim = dim_sizes.index(tensor_shape[d])
                    for d2 in range(idx_dim+1, len(dim_sizes)):
                        stride *= dim_sizes[d2]
                    
                    # Calculate index for this dimension
                    dim_idx = temp_idx // stride
                    slice_multi_idx.append((d, dim_idx))
                    temp_idx = temp_idx % stride
            
            # Bubble sort this slice
            for i in range(dim_size):
                for j in range(dim_size - i - 1):
                    # Create multi-dimensional indices for current and next elements
                    curr_idx = []
                    next_idx = []
                    
                    for d in range(ndims):
                        if d == dim:
                            curr_idx.append(j)
                            next_idx.append(j+1)
                        else:
                            for dim_id, idx in slice_multi_idx:
                                if dim_id == d:
                                    curr_idx.append(idx)
                                    next_idx.append(idx)
                    
                    # Load values
                    curr_val = nl.load(result[tuple(curr_idx)])
                    next_val = nl.load(result[tuple(next_idx)])
                    
                    # Compare and swap if needed
                    if nl.greater(curr_val, next_val).item():
                        # Swap values
                        nl.store(result[tuple(curr_idx)], value=next_val)
                        nl.store(result[tuple(next_idx)], value=curr_val)
    
    return result