from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dim index
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Initialize result with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, load the input tensor into the result tensor
    if len(a_tensor.shape) == 1:
        # For 1D tensors, sort the entire array
        sz = a_tensor.shape[0]
        trip_count = math.ceil(sz / nl.tile_size.pmax)
        
        # Copy the input to result first
        for p in nl.affine_range(trip_count):
            # Generate indices for current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < sz))
            
            # Store to result
            nl.store(result[i_p], value=in_tile, mask=(i_p < sz))
        
        # Bubble sort implementation for 1D array
        # We need n-1 passes through the array
        for i in nl.affine_range(sz - 1):
            for p in nl.affine_range(trip_count):
                # For each position in the array (except the last i elements which are already sorted)
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                
                # For each position, compare with the next element
                # We can only load elements that are within bounds
                curr_values = nl.load(result[i_p], mask=(i_p < sz - i - 1))
                next_indices = nl.minimum(i_p + 1, sz - 1)  # Clamp to prevent out-of-bounds
                next_values = nl.load(result[next_indices], mask=(i_p < sz - i - 1))
                
                # Compare and swap if needed
                swap_needed = nl.greater(curr_values, next_values)
                new_curr = nl.where(swap_needed, next_values, curr_values)
                new_next = nl.where(swap_needed, curr_values, next_values)
                
                # Store back the values
                nl.store(result[i_p], value=new_curr, mask=(i_p < sz - i - 1))
                nl.store(result[next_indices], value=new_next, mask=(i_p < sz - i - 1))
    
    else:
        # For multi-dimensional tensors, sort along the specified dimension
        # Get shape information
        tensor_shape = a_tensor.shape
        sort_dim_size = tensor_shape[dim]
        
        # For simplicity, handle 2D tensors explicitly
        if len(tensor_shape) == 2:
            if dim == 0:
                # Sort along rows
                # First, copy the input tensor to result
                for c in nl.affine_range(math.ceil(tensor_shape[1] / nl.tile_size.pmax)):
                    col_start = c * nl.tile_size.pmax
                    col_size = min(nl.tile_size.pmax, tensor_shape[1] - col_start)
                    i_c = col_start + nl.arange(nl.tile_size.pmax)[None, :]
                    
                    for r in nl.affine_range(tensor_shape[0]):
                        i_r = nl.full((nl.tile_size.pmax,), r, dtype=nl.int32)[:, None]
                        
                        # Load input data
                        in_tile = nl.load(a_tensor[i_r, i_c], mask=(i_c < tensor_shape[1]))
                        
                        # Store to result
                        nl.store(result[i_r, i_c], value=in_tile, mask=(i_c < tensor_shape[1]))
                
                # Sort each column
                for i in nl.affine_range(sort_dim_size - 1):
                    for c in nl.affine_range(math.ceil(tensor_shape[1] / nl.tile_size.pmax)):
                        col_start = c * nl.tile_size.pmax
                        col_size = min(nl.tile_size.pmax, tensor_shape[1] - col_start)
                        i_c = col_start + nl.arange(nl.tile_size.pmax)[None, :]
                        
                        for r in nl.affine_range(sort_dim_size - i - 1):
                            i_r1 = nl.full((nl.tile_size.pmax,), r, dtype=nl.int32)[:, None]
                            i_r2 = nl.full((nl.tile_size.pmax,), r + 1, dtype=nl.int32)[:, None]
                            
                            # Load values to compare
                            val1 = nl.load(result[i_r1, i_c], mask=(i_c < tensor_shape[1]))
                            val2 = nl.load(result[i_r2, i_c], mask=(i_c < tensor_shape[1]))
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(val1, val2)
                            new_val1 = nl.where(swap_needed, val2, val1)
                            new_val2 = nl.where(swap_needed, val1, val2)
                            
                            # Store back the values
                            nl.store(result[i_r1, i_c], value=new_val1, mask=(i_c < tensor_shape[1]))
                            nl.store(result[i_r2, i_c], value=new_val2, mask=(i_c < tensor_shape[1]))
            
            else:  # dim == 1
                # Sort along columns
                # First, copy the input tensor to result
                for r in nl.affine_range(math.ceil(tensor_shape[0] / nl.tile_size.pmax)):
                    row_start = r * nl.tile_size.pmax
                    row_size = min(nl.tile_size.pmax, tensor_shape[0] - row_start)
                    i_r = row_start + nl.arange(nl.tile_size.pmax)[:, None]
                    
                    i_c = nl.arange(tensor_shape[1])[None, :]
                    
                    # Load input data
                    in_tile = nl.load(a_tensor[i_r, i_c], mask=(i_r < tensor_shape[0]))
                    
                    # Store to result
                    nl.store(result[i_r, i_c], value=in_tile, mask=(i_r < tensor_shape[0]))
                
                # Sort each row
                for i in nl.affine_range(sort_dim_size - 1):
                    for r in nl.affine_range(math.ceil(tensor_shape[0] / nl.tile_size.pmax)):
                        row_start = r * nl.tile_size.pmax
                        row_size = min(nl.tile_size.pmax, tensor_shape[0] - row_start)
                        i_r = row_start + nl.arange(nl.tile_size.pmax)[:, None]
                        
                        # For each position in the row (except the last i elements which are already sorted)
                        for j in nl.affine_range(sort_dim_size - i - 1):
                            # Current and next column indices
                            i_c1 = nl.full((1, 1), j, dtype=nl.int32)
                            i_c2 = nl.full((1, 1), j + 1, dtype=nl.int32)
                            
                            # Load values to compare
                            val1 = nl.load(result[i_r, i_c1], mask=(i_r < tensor_shape[0]))
                            val2 = nl.load(result[i_r, i_c2], mask=(i_r < tensor_shape[0]))
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(val1, val2)
                            new_val1 = nl.where(swap_needed, val2, val1)
                            new_val2 = nl.where(swap_needed, val1, val2)
                            
                            # Store back the values
                            nl.store(result[i_r, i_c1], value=new_val1, mask=(i_r < tensor_shape[0]))
                            nl.store(result[i_r, i_c2], value=new_val2, mask=(i_r < tensor_shape[0]))
        
        else:
            # For higher dimensions, we can only sort along the last dimension for simplicity
            # Copy input to result first
            tensor_size = 1
            for i in range(len(tensor_shape)):
                if i != dim:
                    tensor_size *= tensor_shape[i]
            
            trip_count = math.ceil(tensor_size / nl.tile_size.pmax)
            
            # Create flattened view for copying
            for p in nl.affine_range(trip_count):
                flat_idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                
                # Convert flat index to multi-dimensional index for each dimension except the sort dimension
                # This is a simplified approach - in a real implementation we would need to calculate
                # the proper multi-dimensional indices
                
                # For now, just copy the entire tensor
                for d in nl.affine_range(sort_dim_size):
                    if dim == 0:
                        in_tile = nl.load(a_tensor[d], mask=(flat_idx < tensor_size))
                        nl.store(result[d], value=in_tile, mask=(flat_idx < tensor_size))
                    elif dim == len(tensor_shape) - 1:
                        in_tile = nl.load(a_tensor[:, :, d], mask=(flat_idx < tensor_size))
                        nl.store(result[:, :, d], value=in_tile, mask=(flat_idx < tensor_size))
    
    return result