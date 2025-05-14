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
    # We need to handle arbitrarily sized tensors by tiling
    
    # Reshape the problem to handle sorting along any dimension
    # We'll reshape conceptually to [outer_dims, sort_dim_size, inner_dims]
    
    # Calculate sizes
    sort_dim_size = shape[dim]
    
    # First, copy input tensor to result tensor
    if ndim == 1:
        # Special case for 1D tensor
        batch_size = 128  # Process in batches to respect hardware limitations
        num_batches = math.ceil(shape[0] / batch_size)
        
        for i in nl.affine_range(num_batches):
            start_idx = i * batch_size
            # Load a batch from input
            indices = nl.arange(batch_size)
            input_data = nl.load(a_tensor[start_idx + indices], mask=(start_idx + indices < shape[0]))
            
            # Store to result
            nl.store(result[start_idx + indices], input_data, mask=(start_idx + indices < shape[0]))
    else:
        # For multi-dimensional tensors, we need to copy all elements
        if dim == 0:
            # When sorting along first dimension
            dim_size = shape[0]
            inner_size = 1
            for i in range(1, ndim):
                inner_size *= shape[i]
            
            batch_size = min(128, dim_size)  # Respect hardware limitations
            inner_batch = min(128, inner_size)
            
            for i in nl.affine_range(math.ceil(dim_size / batch_size)):
                for j in nl.affine_range(math.ceil(inner_size / inner_batch)):
                    start_i = i * batch_size
                    start_j = j * inner_batch
                    
                    # Create indices
                    indices_i = nl.arange(batch_size)[:, None]
                    indices_j = nl.arange(inner_batch)[None, :]
                    
                    # Load data
                    input_data = nl.load(
                        a_tensor[(start_i + indices_i).reshape(-1), (start_j + indices_j).reshape(-1)],
                        mask=((start_i + indices_i < dim_size) & (start_j + indices_j < inner_size))
                    )
                    
                    # Store data
                    nl.store(
                        result[(start_i + indices_i).reshape(-1), (start_j + indices_j).reshape(-1)],
                        input_data,
                        mask=((start_i + indices_i < dim_size) & (start_j + indices_j < inner_size))
                    )
        else:
            # For other dimensions, we'll reshape and handle as 2D case
            outer_size = 1
            for i in range(0, dim):
                outer_size *= shape[i]
            
            inner_size = 1
            for i in range(dim + 1, ndim):
                inner_size *= shape[i]
            
            batch_outer = min(128, outer_size)
            batch_dim = min(128, sort_dim_size)
            
            for i in nl.affine_range(math.ceil(outer_size / batch_outer)):
                for j in nl.affine_range(math.ceil(sort_dim_size / batch_dim)):
                    start_i = i * batch_outer
                    start_j = j * batch_dim
                    
                    # Create indices
                    indices_i = nl.arange(batch_outer)[:, None]
                    indices_j = nl.arange(batch_dim)[None, :]
                    
                    # Load data
                    input_data = nl.load(
                        a_tensor[(start_i + indices_i).reshape(-1), (start_j + indices_j).reshape(-1)],
                        mask=((start_i + indices_i < outer_size) & (start_j + indices_j < sort_dim_size))
                    )
                    
                    # Store data
                    nl.store(
                        result[(start_i + indices_i).reshape(-1), (start_j + indices_j).reshape(-1)],
                        input_data,
                        mask=((start_i + indices_i < outer_size) & (start_j + indices_j < sort_dim_size))
                    )
    
    # Now perform bubble sort on the result tensor along the specified dimension
    if ndim == 1:
        # For 1D tensor, sort the entire array
        n = shape[0]
        batch_size = min(128, n)
        
        # Bubble sort
        for i in nl.static_range(n):
            for j in nl.affine_range(math.ceil(n / batch_size)):
                start_idx = j * batch_size
                
                # Load current batch
                indices = nl.arange(batch_size)
                current = nl.load(result[start_idx + indices], mask=(start_idx + indices < n - i))
                
                # Load next elements
                next_indices = indices + 1
                next_elements = nl.load(result[start_idx + next_indices], mask=((start_idx + next_indices < n - i) & (next_indices < batch_size)))
                
                # Compare and swap if needed
                swap_mask = nl.greater(current, next_elements) & (next_indices < batch_size) & (start_idx + next_indices < n - i)
                new_current = nl.where(swap_mask, next_elements, current)
                new_next = nl.where(swap_mask, current, next_elements)
                
                # Store back
                nl.store(result[start_idx + indices], new_current, mask=(start_idx + indices < n - i))
                nl.store(result[start_idx + next_indices], new_next, mask=((start_idx + next_indices < n - i) & (next_indices < batch_size)))
    else:
        # For multi-dimensional tensors, sort along the specified dimension
        if dim == 0:
            # Sort along first dimension
            dim_size = shape[0]
            inner_size = 1
            for i in range(1, ndim):
                inner_size *= shape[i]
            
            # Bubble sort along dimension 0
            for i in nl.static_range(dim_size):
                for j in nl.affine_range(math.ceil(inner_size / 128)):
                    start_j = j * 128
                    
                    # Create indices for inner dimensions
                    inner_indices = nl.arange(min(128, inner_size))
                    
                    # Only process up to dim_size - i - 1 to avoid out-of-bounds
                    # Load current elements
                    current_indices_dim = nl.full((min(128, inner_size),), 0, dtype=nl.int32)
                    current = nl.load(
                        result[current_indices_dim, start_j + inner_indices], 
                        mask=(current_indices_dim < dim_size - i) & (start_j + inner_indices < inner_size)
                    )
                    
                    # Load next elements
                    next_indices_dim = nl.full((min(128, inner_size),), 1, dtype=nl.int32)
                    next_elements = nl.load(
                        result[next_indices_dim, start_j + inner_indices], 
                        mask=(next_indices_dim < dim_size - i) & (start_j + inner_indices < inner_size)
                    )
                    
                    # Compare and swap
                    swap_mask = nl.greater(current, next_elements) & (next_indices_dim < dim_size - i) & (start_j + inner_indices < inner_size)
                    new_current = nl.where(swap_mask, next_elements, current)
                    new_next = nl.where(swap_mask, current, next_elements)
                    
                    # Store back
                    nl.store(
                        result[current_indices_dim, start_j + inner_indices], 
                        new_current, 
                        mask=(current_indices_dim < dim_size - i) & (start_j + inner_indices < inner_size)
                    )
                    nl.store(
                        result[next_indices_dim, start_j + inner_indices], 
                        new_next, 
                        mask=(next_indices_dim < dim_size - i) & (start_j + inner_indices < inner_size)
                    )
        else:
            # Sort along other dimensions
            # For simplicity, we'll focus on 2D case
            if ndim == 2:
                rows = shape[0]
                cols = shape[1]
                
                if dim == 1:
                    # Sort each row
                    for i in nl.affine_range(rows):
                        # Bubble sort this row
                        for j in nl.static_range(cols):
                            # Process in batches to respect hardware limitations
                            batch_size = min(128, cols - j - 1)
                            
                            for k in nl.affine_range(math.ceil((cols - j - 1) / batch_size)):
                                start_idx = k * batch_size
                                
                                # Load current elements
                                current_indices = nl.arange(batch_size)
                                current = nl.load(result[i, start_idx + current_indices], mask=(start_idx + current_indices < cols - j))
                                
                                # Load next elements
                                next_indices = current_indices + 1
                                next_elements = nl.load(result[i, start_idx + next_indices], mask=((start_idx + next_indices < cols - j) & (next_indices < batch_size)))
                                
                                # Compare and swap if needed
                                swap_mask = nl.greater(current, next_elements) & (next_indices < batch_size) & (start_idx + next_indices < cols - j)
                                new_current = nl.where(swap_mask, next_elements, current)
                                new_next = nl.where(swap_mask, current, next_elements)
                                
                                # Store back
                                nl.store(result[i, start_idx + current_indices], new_current, mask=(start_idx + current_indices < cols - j))
                                nl.store(result[i, start_idx + next_indices], new_next, mask=((start_idx + next_indices < cols - j) & (next_indices < batch_size)))
    
    return result