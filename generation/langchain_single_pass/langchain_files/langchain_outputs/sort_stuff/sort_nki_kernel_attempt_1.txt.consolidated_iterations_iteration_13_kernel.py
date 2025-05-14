from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and normalize dimension
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Convert negative dim to positive
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle 1D case
    if ndim == 1:
        size = shape[0]
        # Create workspace buffer for sorting
        temp = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.sbuf)
        
        # Load input tensor into temp buffer
        i = nl.arange(size)
        temp_data = nl.load(a_tensor[i])
        
        # Bubble sort
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Get current and next element
                j_idx = nl.arange(size - 1)
                curr = nl.load(temp[j_idx])
                next_idx = j_idx + 1
                next_val = nl.load(temp[next_idx])
                
                # Compare and swap if needed
                mask = nl.greater(curr, next_val)
                tmp = nl.copy(curr)
                nl.store(temp[j_idx], nl.copy(next_val), mask=mask)
                nl.store(temp[next_idx], tmp, mask=mask)
        
        # Store sorted result
        nl.store(result[nl.arange(size)], nl.load(temp[nl.arange(size)]))
    
    # Handle 2D case
    elif ndim == 2:
        # Get dimensions
        dim0 = shape[0]
        dim1 = shape[1]
        
        # Sort along dimension 0
        if dim == 0:
            # Process in tiles to respect hardware limitations
            max_tile = nl.tile_size.pmax
            trips = math.ceil(dim0 / max_tile)
            
            for col in nl.affine_range(dim1):
                # For each column, sort all elements
                
                # First, copy the entire column to result
                i_all = nl.arange(dim0)
                col_data = nl.load(a_tensor[i_all, col])
                nl.store(result[i_all, col], col_data)
                
                # Then sort each column using bubble sort
                for _ in nl.affine_range(dim0):
                    for t in nl.affine_range(trips):
                        start = t * max_tile
                        i_p = nl.arange(max_tile)
                        i_idx = start + i_p
                        
                        # Load current batch of elements
                        mask = i_idx < (dim0 - 1)
                        curr = nl.load(result[i_idx, col], mask=mask)
                        next_idx = i_idx + 1
                        mask_next = next_idx < dim0
                        next_val = nl.load(result[next_idx, col], mask=mask_next)
                        
                        # Compare and swap if needed
                        swap_mask = (i_idx < (dim0 - 1)) & (nl.greater(curr, next_val))
                        tmp = nl.copy(curr)
                        nl.store(result[i_idx, col], nl.copy(next_val), mask=swap_mask)
                        nl.store(result[next_idx, col], tmp, mask=swap_mask)
                        
        # Sort along dimension 1
        else:  # dim == 1
            # Process in tiles to respect hardware limitations
            max_tile = nl.tile_size.pmax
            trips = math.ceil(dim1 / max_tile)
            
            for row in nl.affine_range(dim0):
                # For each row, sort all elements
                
                # First, copy the entire row to result
                i_all = nl.arange(dim1)
                row_data = nl.load(a_tensor[row, i_all])
                nl.store(result[row, i_all], row_data)
                
                # Then sort each row using bubble sort
                for _ in nl.affine_range(dim1):
                    for t in nl.affine_range(trips):
                        start = t * max_tile
                        i_p = nl.arange(max_tile)
                        i_idx = start + i_p
                        
                        # Load current batch of elements
                        mask = i_idx < (dim1 - 1)
                        curr = nl.load(result[row, i_idx], mask=mask)
                        next_idx = i_idx + 1
                        mask_next = next_idx < dim1
                        next_val = nl.load(result[row, next_idx], mask=mask_next)
                        
                        # Compare and swap if needed
                        swap_mask = (i_idx < (dim1 - 1)) & (nl.greater(curr, next_val))
                        tmp = nl.copy(curr)
                        nl.store(result[row, i_idx], nl.copy(next_val), mask=swap_mask)
                        nl.store(result[row, next_idx], tmp, mask=swap_mask)
                        
    # Higher dimensional case
    else:
        # For dimensions > 2, we need to handle the general case
        # First, copy input to result
        if dim == ndim - 1:
            # Last dimension is special case, can be handled more efficiently
            # Copy input to result first
            last_dim_size = shape[dim]
            
            # Process in tiles for partition dimension
            max_tile = nl.tile_size.pmax
            
            # Calculate the product of dimensions before the sort dimension
            before_dims_size = 1
            for i in range(dim):
                before_dims_size *= shape[i]
                
            trips = math.ceil(before_dims_size / max_tile)
            
            for t in nl.affine_range(trips):
                start = t * max_tile
                i_p = nl.arange(max_tile)
                i_idx = start + i_p
                i_f = nl.arange(last_dim_size)[None, :]
                
                # Load and store batch
                mask = i_idx[:, None] < before_dims_size
                batch_data = nl.load(a_tensor[i_idx[:, None], i_f], mask=mask)
                nl.store(result[i_idx[:, None], i_f], batch_data, mask=mask)
                
            # Now sort each slice along the last dimension
            for _ in nl.affine_range(last_dim_size):
                for t in nl.affine_range(trips):
                    start = t * max_tile
                    i_p = nl.arange(max_tile)
                    i_idx = start + i_p
                    
                    # For each i_idx < before_dims_size, sort the corresponding slice
                    for j in nl.affine_range(last_dim_size - 1):
                        j_idx = nl.arange(last_dim_size - 1)
                        
                        # Load current and next elements
                        mask = i_idx[:, None] < before_dims_size
                        curr = nl.load(result[i_idx[:, None], j_idx[None, :]], mask=mask)
                        next_idx = j_idx + 1
                        next_val = nl.load(result[i_idx[:, None], next_idx[None, :]], mask=mask)
                        
                        # Compare and swap if needed
                        swap_mask = mask & nl.greater(curr, next_val)
                        tmp = nl.copy(curr)
                        nl.store(result[i_idx[:, None], j_idx[None, :]], nl.copy(next_val), mask=swap_mask)
                        nl.store(result[i_idx[:, None], next_idx[None, :]], tmp, mask=swap_mask)
        else:
            # For sorting along non-last dimensions, we copy the input first
            # Create flattened indices for copying
            max_tile = nl.tile_size.pmax
            total_size = 1
            for i in range(ndim):
                total_size *= shape[i]
                
            trips = math.ceil(total_size / max_tile)
            
            # Copy input to result
            for t in nl.affine_range(trips):
                start = t * max_tile
                i_p = nl.arange(max_tile)
                flat_idx = start + i_p
                
                # Create multi-dimensional indices
                indices = []
                temp_flat = nl.copy(flat_idx)
                
                for d in range(ndim-1, -1, -1):
                    dim_size = shape[d]
                    indices.insert(0, temp_flat % dim_size)
                    temp_flat = temp_flat // dim_size
                
                # Load and store batch
                mask = flat_idx < total_size
                
                # We can't use dynamic indices here, so we'll implement a simplified version
                # that sorts along dimension 0 for higher-dimensional tensors
                if dim == 0:
                    # Sort along first dimension
                    dim0_size = shape[0]
                    rest_size = total_size // dim0_size
                    
                    for slice_idx in nl.affine_range(rest_size):
                        # Copy slice
                        i_all = nl.arange(dim0_size)
                        slice_data = nl.load(a_tensor[i_all, slice_idx // shape[1]])
                        nl.store(result[i_all, slice_idx // shape[1]], slice_data)
                        
                        # Sort slice
                        for _ in nl.affine_range(dim0_size):
                            for i in nl.affine_range(dim0_size - 1):
                                i_idx = nl.arange(dim0_size - 1)
                                curr = nl.load(result[i_idx, slice_idx // shape[1]])
                                next_val = nl.load(result[i_idx + 1, slice_idx // shape[1]])
                                
                                # Compare and swap
                                swap_mask = nl.greater(curr, next_val)
                                tmp = nl.copy(curr)
                                nl.store(result[i_idx, slice_idx // shape[1]], nl.copy(next_val), mask=swap_mask)
                                nl.store(result[i_idx + 1, slice_idx // shape[1]], tmp, mask=swap_mask)
    
    return result