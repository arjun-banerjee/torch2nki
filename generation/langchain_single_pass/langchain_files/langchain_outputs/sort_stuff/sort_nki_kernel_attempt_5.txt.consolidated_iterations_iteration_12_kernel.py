from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension
    ndim = len(a_tensor.shape)
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy the input tensor to result
    if ndim == 1:
        # Handle 1D tensor case
        size = a_tensor.shape[0]
        tile_size = min(size, nl.tile_size.pmax)
        
        # Copy input to result first
        for i in nl.affine_range(math.ceil(size / tile_size)):
            start_idx = i * tile_size
            # Create indices for current tile
            indices = nl.arange(tile_size) + start_idx
            # Load input data
            in_tile = nl.load(a_tensor[indices], mask=(indices < size))
            # Store to result
            nl.store(result[indices], in_tile, mask=(indices < size))
        
        # Bubble sort implementation for 1D tensor
        for i in range(size):
            for j in nl.affine_range(math.ceil((size - i - 1) / tile_size)):
                start_idx = j * tile_size
                # Create indices for current tile
                indices = nl.arange(tile_size) + start_idx
                # Load current values
                values = nl.load(result[indices], mask=(indices < size - i - 1))
                
                # Calculate next indices (shifted by 1)
                next_indices = indices + 1
                # Load next values
                next_values = nl.load(result[next_indices], mask=(indices < size - i - 1))
                
                # Compare and swap if needed
                swap_needed = nl.greater(values, next_values)
                new_values = nl.where(swap_needed, next_values, values)
                new_next_values = nl.where(swap_needed, values, next_values)
                
                # Store the updated values
                nl.store(result[indices], new_values, mask=(indices < size - i - 1))
                nl.store(result[next_indices], new_next_values, mask=(indices < size - i - 1))
    else:
        # Handle multi-dimensional tensor
        # For simplicity, we'll focus on 2D case but this can be extended
        if ndim == 2:
            if dim == 0:
                # Sort along first dimension
                size0 = a_tensor.shape[0]
                size1 = a_tensor.shape[1]
                
                # Copy input to result
                for i in nl.affine_range(math.ceil(size0 / nl.tile_size.pmax)):
                    start_idx = i * nl.tile_size.pmax
                    # Create indices for current tile
                    i_p = nl.arange(nl.tile_size.pmax)[:, None] + start_idx
                    i_f = nl.arange(size1)[None, :]
                    # Load input data
                    in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < size0))
                    # Store to result
                    nl.store(result[i_p, i_f], in_tile, mask=(i_p < size0))
                
                # Sort each column independently
                for col in range(size1):
                    for i in range(size0):
                        for j in nl.affine_range(math.ceil((size0 - i - 1) / nl.tile_size.pmax)):
                            start_idx = j * nl.tile_size.pmax
                            # Create indices for current tile
                            indices = nl.arange(nl.tile_size.pmax) + start_idx
                            
                            # Load current values (specific column)
                            mask = (indices < size0 - i - 1)
                            values = nl.load(result[indices, col], mask=mask)
                            
                            # Calculate next indices (shifted by 1)
                            next_indices = indices + 1
                            # Load next values
                            next_values = nl.load(result[next_indices, col], mask=mask)
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(values, next_values)
                            new_values = nl.where(swap_needed, next_values, values)
                            new_next_values = nl.where(swap_needed, values, next_values)
                            
                            # Store the updated values
                            nl.store(result[indices, col], new_values, mask=mask)
                            nl.store(result[next_indices, col], new_next_values, mask=mask)
            else:
                # Sort along second dimension (dim == 1)
                size0 = a_tensor.shape[0]
                size1 = a_tensor.shape[1]
                
                # Copy input to result
                for i in nl.affine_range(math.ceil(size0 / nl.tile_size.pmax)):
                    start_idx = i * nl.tile_size.pmax
                    # Create indices for current tile
                    i_p = nl.arange(nl.tile_size.pmax)[:, None] + start_idx
                    i_f = nl.arange(size1)[None, :]
                    # Load input data
                    in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < size0))
                    # Store to result
                    nl.store(result[i_p, i_f], in_tile, mask=(i_p < size0))
                
                # Sort each row independently
                for row in range(size0):
                    for i in range(size1):
                        for j in nl.affine_range(math.ceil((size1 - i - 1) / nl.tile_size.fmax)):
                            start_idx = j * nl.tile_size.fmax
                            # Create indices for current tile
                            indices = nl.arange(nl.tile_size.fmax) + start_idx
                            
                            # Load current values (specific row)
                            mask = (indices < size1 - i - 1)
                            values = nl.load(result[row, indices], mask=mask)
                            
                            # Calculate next indices (shifted by 1)
                            next_indices = indices + 1
                            # Load next values
                            next_values = nl.load(result[row, next_indices], mask=mask)
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(values, next_values)
                            new_values = nl.where(swap_needed, next_values, values)
                            new_next_values = nl.where(swap_needed, values, next_values)
                            
                            # Store the updated values
                            nl.store(result[row, indices], new_values, mask=mask)
                            nl.store(result[row, next_indices], new_next_values, mask=mask)
        else:
            # For higher dimensions, copy the input to result
            # This is a placeholder - would need more complex handling for higher dimensions
            for i in nl.affine_range(math.ceil(a_tensor.shape[0] / nl.tile_size.pmax)):
                start_idx = i * nl.tile_size.pmax
                # Create indices for current tile
                i_p = nl.arange(nl.tile_size.pmax)[:, None] + start_idx
                i_f = nl.arange(a_tensor.shape[1])[None, :]
                # Load input data
                in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < a_tensor.shape[0]))
                # Store to result
                nl.store(result[i_p, i_f], in_tile, mask=(i_p < a_tensor.shape[0]))
            
            # Note: For higher dimensions, would need to implement sorting along specified dimension
    
    return result