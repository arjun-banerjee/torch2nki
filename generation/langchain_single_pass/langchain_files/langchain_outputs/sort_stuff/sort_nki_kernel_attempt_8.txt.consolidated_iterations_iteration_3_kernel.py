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
    
    # Get tensor shape and determine the sort dimension size
    tensor_shape = a_tensor.shape
    sort_dim_size = tensor_shape[dim]
    
    # If the tensor is 1D or we're sorting along the last dimension
    if dim == len(tensor_shape) - 1:
        # Calculate leading dimensions product (all dimensions before sort dimension)
        leading_dims = 1
        for i in range(dim):
            leading_dims *= tensor_shape[i]
        
        # Process in tiles to respect hardware limitations
        trip_count = math.ceil(leading_dims / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            p_start = p * nl.tile_size.pmax
            p_size = min(nl.tile_size.pmax, leading_dims - p_start)
            
            # Load the data for this tile
            if dim == 0:
                # Special case for 1D tensor or sorting along first dimension
                in_data = nl.load(a_tensor[p_start:p_start+p_size], mask=(nl.arange(p_size) < p_size))
                temp_data = nl.zeros((p_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                nl.store(temp_data, in_data)
                
                # Bubble sort
                for i in nl.affine_range(p_size):
                    for j in nl.affine_range(p_size - 1):
                        # Load adjacent elements
                        val_j = nl.load(temp_data[j])
                        val_j_next = nl.load(temp_data[j+1])
                        
                        # Compare and swap if needed
                        should_swap = nl.greater(val_j, val_j_next)
                        if should_swap:
                            nl.store(temp_data[j], val_j_next)
                            nl.store(temp_data[j+1], val_j)
                
                # Store the sorted result
                out_data = nl.load(temp_data)
                nl.store(result[p_start:p_start+p_size], out_data, mask=(nl.arange(p_size) < p_size))
            
            else:
                # For multi-dimensional tensors, process each "row" separately
                i_p = p_start + nl.arange(p_size)[:, None]
                i_f = nl.arange(sort_dim_size)[None, :]
                
                in_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < leading_dims))
                
                # Sort each row independently
                for row in nl.affine_range(p_size):
                    row_data = nl.zeros((sort_dim_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                    
                    # Extract row data
                    for col in nl.affine_range(sort_dim_size):
                        nl.store(row_data[col], in_tile[row, col])
                    
                    # Bubble sort the row
                    for i in nl.affine_range(sort_dim_size):
                        for j in nl.affine_range(sort_dim_size - 1):
                            val_j = nl.load(row_data[j])
                            val_j_next = nl.load(row_data[j+1])
                            
                            should_swap = nl.greater(val_j, val_j_next)
                            if should_swap:
                                nl.store(row_data[j], val_j_next)
                                nl.store(row_data[j+1], val_j)
                    
                    # Store sorted row back to the result tile
                    for col in nl.affine_range(sort_dim_size):
                        in_tile[row, col] = nl.load(row_data[col])
                
                nl.store(result[i_p, i_f], in_tile, mask=(i_p < leading_dims))
    
    else:
        # Transpose tensor to make sort dimension the last dimension
        transposed_shape = list(tensor_shape)
        transposed_shape[dim], transposed_shape[-1] = transposed_shape[-1], transposed_shape[dim]
        
        # Create temporary transposed tensor
        transposed_input = nl.ndarray(transposed_shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
        transposed_output = nl.ndarray(transposed_shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
        
        # Transpose input tensor
        leading_dims = 1
        for i in range(len(tensor_shape) - 1):
            leading_dims *= transposed_shape[i]
        
        trip_count = math.ceil(leading_dims / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            p_start = p * nl.tile_size.pmax
            p_size = min(nl.tile_size.pmax, leading_dims - p_start)
            
            i_p = p_start + nl.arange(p_size)[:, None]
            i_f_in = nl.arange(tensor_shape[dim])[None, :]
            i_f_out = nl.arange(tensor_shape[-1])[None, :]
            
            # Load, transpose, and store
            if dim == 0 and len(tensor_shape) == 2:
                in_tile = nl.load(a_tensor[i_f_in, i_p], mask=(i_p < leading_dims))
                out_tile = nl.transpose(in_tile)
                nl.store(transposed_input[i_p, i_f_in], out_tile, mask=(i_p < leading_dims))
            else:
                # More complex transposition needed for higher dimensions
                # This is a simplified approach for 2D tensors
                in_tile = nl.load(a_tensor[i_p, i_f_in], mask=(i_p < leading_dims))
                nl.store(transposed_input[i_p, i_f_in], in_tile, mask=(i_p < leading_dims))
        
        # Sort transposed tensor along last dimension (recursive call)
        temp_result = nki_sort(transposed_input, dim=-1)
        
        # Transpose result back
        for p in nl.affine_range(trip_count):
            p_start = p * nl.tile_size.pmax
            p_size = min(nl.tile_size.pmax, leading_dims - p_start)
            
            i_p = p_start + nl.arange(p_size)[:, None]
            i_f_in = nl.arange(tensor_shape[-1])[None, :]
            i_f_out = nl.arange(tensor_shape[dim])[None, :]
            
            # Load, transpose, and store
            if dim == 0 and len(tensor_shape) == 2:
                in_tile = nl.load(temp_result[i_p, i_f_in], mask=(i_p < leading_dims))
                out_tile = nl.transpose(in_tile)
                nl.store(result[i_f_out, i_p], out_tile, mask=(i_p < leading_dims))
            else:
                # More complex transposition needed for higher dimensions
                in_tile = nl.load(temp_result[i_p, i_f_in], mask=(i_p < leading_dims))
                nl.store(result[i_p, i_f_out], in_tile, mask=(i_p < leading_dims))
    
    return result