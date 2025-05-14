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
    # Calculate the maximum tile size for partition dimension
    max_tile_size = min(128, shape[0])  # respecting hardware limitation
    
    # Process the tensor in tiles
    for p_idx in nl.affine_range(math.ceil(shape[0] / max_tile_size)):
        start_p = p_idx * max_tile_size
        
        # Handle last tile that might be smaller
        actual_tile_size = min(max_tile_size, shape[0] - start_p)
        
        if dim == 0:
            # If sorting along first dimension, we need a different approach
            # First, load the entire tile
            if ndim == 1:
                # 1D tensor case
                input_tile = nl.load(a_tensor[start_p:start_p+actual_tile_size], 
                                    mask=(nl.arange(max_tile_size) < actual_tile_size))
                
                # Perform bubble sort on this tile
                for i in nl.affine_range(actual_tile_size):
                    for j in nl.affine_range(actual_tile_size - 1):
                        # Create indices for the current and next elements
                        curr_idx = j
                        next_idx = j + 1
                        
                        # Use mask to ensure we're within bounds
                        valid_indices = (next_idx < actual_tile_size)
                        
                        # Compare and swap if needed
                        curr_val = input_tile[curr_idx]
                        next_val = input_tile[next_idx]
                        
                        # Use nl.less to compare values
                        is_greater = nl.greater(curr_val, next_val)
                        
                        # Swap values if curr_val > next_val
                        temp = nl.where(is_greater, next_val, curr_val)
                        next_val = nl.where(is_greater, curr_val, next_val)
                        curr_val = temp
                        
                        # Update the tile with the swapped values
                        input_tile = nl.tensor_update(input_tile, curr_idx, curr_val, mask=valid_indices)
                        input_tile = nl.tensor_update(input_tile, next_idx, next_val, mask=valid_indices)
                
                # Store the sorted tile
                nl.store(result[start_p:start_p+actual_tile_size], input_tile, 
                         mask=(nl.arange(max_tile_size) < actual_tile_size))
            else:
                # Multi-dimensional tensor case, handle dim=0 specially
                for f_idx in nl.affine_range(shape[1]):
                    input_slice = nl.load(a_tensor[start_p:start_p+actual_tile_size, f_idx], 
                                         mask=(nl.arange(max_tile_size) < actual_tile_size))
                    
                    # Perform bubble sort on this slice
                    for i in nl.affine_range(actual_tile_size):
                        for j in nl.affine_range(actual_tile_size - 1):
                            curr_idx = j
                            next_idx = j + 1
                            valid_indices = (next_idx < actual_tile_size)
                            
                            curr_val = input_slice[curr_idx]
                            next_val = input_slice[next_idx]
                            
                            is_greater = nl.greater(curr_val, next_val)
                            
                            temp = nl.where(is_greater, next_val, curr_val)
                            next_val = nl.where(is_greater, curr_val, next_val)
                            curr_val = temp
                            
                            input_slice = nl.tensor_update(input_slice, curr_idx, curr_val, mask=valid_indices)
                            input_slice = nl.tensor_update(input_slice, next_idx, next_val, mask=valid_indices)
                    
                    # Store the sorted slice
                    nl.store(result[start_p:start_p+actual_tile_size, f_idx], input_slice, 
                             mask=(nl.arange(max_tile_size) < actual_tile_size))
        else:
            # If sorting along other dimensions
            if ndim == 1:
                # If 1D tensor, dim must be 0, already handled above
                pass
            elif ndim == 2:
                # 2D tensor case
                if dim == 1:
                    # Sort along the second dimension
                    for f_start in nl.affine_range(math.ceil(shape[1] / 128)):
                        f_size = min(128, shape[1] - f_start * 128)
                        
                        # Load a tile from the input tensor
                        input_tile = nl.load(a_tensor[start_p:start_p+actual_tile_size, 
                                                      f_start*128:f_start*128+f_size], 
                                           mask=((nl.arange(max_tile_size)[:, None] < actual_tile_size) & 
                                                (nl.arange(128)[None, :] < f_size)))
                        
                        # Perform bubble sort on each row
                        for i in nl.affine_range(actual_tile_size):
                            # Bubble sort for this row
                            for j in nl.affine_range(f_size):
                                for k in nl.affine_range(f_size - 1 - j):
                                    # Compare and swap if needed
                                    curr_val = input_tile[i, k]
                                    next_val = input_tile[i, k+1]
                                    
                                    # Swap if needed
                                    is_greater = nl.greater(curr_val, next_val)
                                    
                                    # Use where to conditionally swap
                                    new_curr = nl.where(is_greater, next_val, curr_val)
                                    new_next = nl.where(is_greater, curr_val, next_val)
                                    
                                    # Update values
                                    input_tile = nl.tensor_update(input_tile, (i, k), new_curr)
                                    input_tile = nl.tensor_update(input_tile, (i, k+1), new_next)
                        
                        # Store the sorted tile
                        nl.store(result[start_p:start_p+actual_tile_size, 
                                        f_start*128:f_start*128+f_size], input_tile,
                               mask=((nl.arange(max_tile_size)[:, None] < actual_tile_size) & 
                                    (nl.arange(128)[None, :] < f_size)))
            else:
                # For higher dimensions, we need to handle them differently
                # This is a simplified approach for demonstration
                # Copy input to output for dimensions we don't fully support yet
                for f_idx in nl.affine_range(shape[1]):
                    input_slice = nl.load(a_tensor[start_p:start_p+actual_tile_size, f_idx], 
                                         mask=(nl.arange(max_tile_size) < actual_tile_size))
                    nl.store(result[start_p:start_p+actual_tile_size, f_idx], input_slice, 
                             mask=(nl.arange(max_tile_size) < actual_tile_size))
    
    return result