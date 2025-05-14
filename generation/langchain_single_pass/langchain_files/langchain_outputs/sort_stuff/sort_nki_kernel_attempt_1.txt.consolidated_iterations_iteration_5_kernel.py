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
    
    # Handle 1D case separately
    if ndim == 1:
        size = shape[0]
        # Simple bubble sort for 1D tensor
        for i in nl.affine_range(size-1):
            for j in nl.affine_range(size-i-1):
                # Load current and next element
                curr_val = nl.load(a_tensor[j])
                next_val = nl.load(a_tensor[j+1])
                
                # Compare and swap if needed
                cond = nl.greater(curr_val, next_val)
                
                # Store the smaller value at j
                nl.store(result[j], nl.where(cond, next_val, curr_val))
                
                # Store the larger value at j+1
                nl.store(result[j+1], nl.where(cond, curr_val, next_val))
                
                # Update a_tensor for next iteration
                nl.store(a_tensor[j], nl.load(result[j]))
                nl.store(a_tensor[j+1], nl.load(result[j+1]))
    
    # Handle 2D case
    elif ndim == 2:
        rows, cols = shape
        
        # Sort along dimension 0 (rows)
        if dim == 0:
            # For each column
            for col in nl.affine_range(cols):
                # Bubble sort the column
                for i in nl.affine_range(rows-1):
                    for j in nl.affine_range(rows-i-1):
                        # Load current and next element
                        curr_val = nl.load(a_tensor[j, col])
                        next_val = nl.load(a_tensor[j+1, col])
                        
                        # Compare and swap if needed
                        cond = nl.greater(curr_val, next_val)
                        
                        # Store the smaller value at j
                        nl.store(result[j, col], nl.where(cond, next_val, curr_val))
                        
                        # Store the larger value at j+1
                        nl.store(result[j+1, col], nl.where(cond, curr_val, next_val))
                        
                        # Update a_tensor for next iteration
                        nl.store(a_tensor[j, col], nl.load(result[j, col]))
                        nl.store(a_tensor[j+1, col], nl.load(result[j+1, col]))
        
        # Sort along dimension 1 (columns)
        else:  # dim == 1
            # For each row
            for row in nl.affine_range(rows):
                # Bubble sort the row
                for i in nl.affine_range(cols-1):
                    for j in nl.affine_range(cols-i-1):
                        # Load current and next element
                        curr_val = nl.load(a_tensor[row, j])
                        next_val = nl.load(a_tensor[row, j+1])
                        
                        # Compare and swap if needed
                        cond = nl.greater(curr_val, next_val)
                        
                        # Store the smaller value at j
                        nl.store(result[row, j], nl.where(cond, next_val, curr_val))
                        
                        # Store the larger value at j+1
                        nl.store(result[row, j+1], nl.where(cond, curr_val, next_val))
                        
                        # Update a_tensor for next iteration
                        nl.store(a_tensor[row, j], nl.load(result[row, j]))
                        nl.store(a_tensor[row, j+1], nl.load(result[row, j+1]))
    
    # For higher dimensional tensors
    else:
        # First copy input to result
        # We need to do this in tiles to handle large tensors
        
        # Calculate the product of dimensions for tiling
        total_size = 1
        for i in range(ndim):
            total_size *= shape[i]
            
        # Process in tiles
        tile_size = 128  # Max tile size
        num_tiles = math.ceil(total_size / tile_size)
        
        # Copy input to result first
        for t in nl.affine_range(num_tiles):
            start_idx = t * tile_size
            # Create indices for this tile
            indices = start_idx + nl.arange(tile_size)
            # Apply mask to handle the last tile which might be smaller
            mask = indices < total_size
            
            # Convert flat indices to multi-dimensional indices
            # This is a simplified approach - we'll load and store one element at a time
            for idx in nl.affine_range(tile_size):
                flat_idx = start_idx + idx
                if flat_idx < total_size:
                    # Convert flat index to multi-dimensional indices
                    # This is a simple approach for demonstration
                    multi_idx = []
                    temp_idx = flat_idx
                    for i in range(ndim-1, -1, -1):
                        dim_size = shape[i]
                        multi_idx.insert(0, temp_idx % dim_size)
                        temp_idx = temp_idx // dim_size
                    
                    # Create a tuple of indices for loading and storing
                    # We need to access each element individually
                    if ndim == 3:
                        elem = nl.load(a_tensor[multi_idx[0], multi_idx[1], multi_idx[2]])
                        nl.store(result[multi_idx[0], multi_idx[1], multi_idx[2]], elem)
                    elif ndim == 4:
                        elem = nl.load(a_tensor[multi_idx[0], multi_idx[1], multi_idx[2], multi_idx[3]])
                        nl.store(result[multi_idx[0], multi_idx[1], multi_idx[2], multi_idx[3]], elem)
                    # Add more cases for higher dimensions if needed
        
        # Now perform the sort along the specified dimension
        # We need to handle each slice along the sort dimension separately
        
        # For simplicity, we'll implement this for 3D tensors
        # Extension to higher dimensions follows the same pattern
        if ndim == 3:
            d1, d2, d3 = shape
            if dim == 0:
                for i2 in nl.affine_range(d2):
                    for i3 in nl.affine_range(d3):
                        # Bubble sort along dimension 0
                        for i in nl.affine_range(d1-1):
                            for j in nl.affine_range(d1-i-1):
                                curr_val = nl.load(result[j, i2, i3])
                                next_val = nl.load(result[j+1, i2, i3])
                                cond = nl.greater(curr_val, next_val)
                                nl.store(result[j, i2, i3], nl.where(cond, next_val, curr_val))
                                nl.store(result[j+1, i2, i3], nl.where(cond, curr_val, next_val))
            elif dim == 1:
                for i1 in nl.affine_range(d1):
                    for i3 in nl.affine_range(d3):
                        # Bubble sort along dimension 1
                        for i in nl.affine_range(d2-1):
                            for j in nl.affine_range(d2-i-1):
                                curr_val = nl.load(result[i1, j, i3])
                                next_val = nl.load(result[i1, j+1, i3])
                                cond = nl.greater(curr_val, next_val)
                                nl.store(result[i1, j, i3], nl.where(cond, next_val, curr_val))
                                nl.store(result[i1, j+1, i3], nl.where(cond, curr_val, next_val))
            else:  # dim == 2
                for i1 in nl.affine_range(d1):
                    for i2 in nl.affine_range(d2):
                        # Bubble sort along dimension 2
                        for i in nl.affine_range(d3-1):
                            for j in nl.affine_range(d3-i-1):
                                curr_val = nl.load(result[i1, i2, j])
                                next_val = nl.load(result[i1, i2, j+1])
                                cond = nl.greater(curr_val, next_val)
                                nl.store(result[i1, i2, j], nl.where(cond, next_val, curr_val))
                                nl.store(result[i1, i2, j+1], nl.where(cond, curr_val, next_val))
    
    return result