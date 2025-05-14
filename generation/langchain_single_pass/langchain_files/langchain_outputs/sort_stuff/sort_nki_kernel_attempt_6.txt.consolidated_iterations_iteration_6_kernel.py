from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimensions
    if dim < 0:
        dim = len(a_tensor.shape) + dim
        
    # Get shape of input tensor
    shape = a_tensor.shape
    sort_dim_size = shape[dim]
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy the input tensor to result
    if len(shape) == 1:
        # Handle 1D tensor case
        max_tile_size = nl.tile_size.pmax
        trip_count = math.ceil(shape[0] / max_tile_size)
        
        for p in nl.affine_range(trip_count):
            i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
            i_f = nl.zeros((1, 1), dtype=nl.int32)
            
            # Load input data with proper masking
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < shape[0]))
            
            # Store to result
            nl.store(result[i_p], value=x_tile, mask=(i_p < shape[0]))
            
        # Perform bubble sort on the entire array
        for i in nl.affine_range(sort_dim_size):
            for j in nl.affine_range(sort_dim_size - 1):
                for p in nl.affine_range(trip_count):
                    i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
                    
                    # Load current elements with proper masking
                    curr = nl.load(result[i_p], mask=(i_p < shape[0]))
                    
                    # Load next elements with offset and proper masking
                    next_indices = i_p + 1
                    next_vals = nl.load(result[next_indices], mask=(next_indices < shape[0]) & (i_p < shape[0]))
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr, next_vals)
                    
                    # Create swapped values
                    new_curr = nl.where(swap_needed, next_vals, curr)
                    new_next = nl.where(swap_needed, curr, next_vals)
                    
                    # Store back the results with proper masking
                    nl.store(result[i_p], value=new_curr, mask=(i_p < shape[0]))
                    nl.store(result[next_indices], value=new_next, mask=(next_indices < shape[0]) & (i_p < shape[0]))
    else:
        # Handle multi-dimensional tensor case
        # First, copy the input tensor to result
        if dim == 0:
            # Sort along first dimension
            max_tile_size = nl.tile_size.pmax
            trip_count = math.ceil(shape[0] / max_tile_size)
            remaining_dims = shape[1:]
            
            # Create indices for remaining dimensions
            indices_list = []
            for i in range(len(remaining_dims)):
                indices_list.append(nl.arange(remaining_dims[i]))
            
            # Create meshgrid for remaining dimensions
            remaining_indices = nl.meshgrid(*indices_list)
            
            # Copy input to result
            for p in nl.affine_range(trip_count):
                i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
                
                # Load and store with proper masking
                for idx_tuple in remaining_indices:
                    x_tile = nl.load(a_tensor[i_p, idx_tuple], mask=(i_p < shape[0]))
                    nl.store(result[i_p, idx_tuple], value=x_tile, mask=(i_p < shape[0]))
            
            # Perform bubble sort
            for i in nl.affine_range(sort_dim_size):
                for j in nl.affine_range(sort_dim_size - 1):
                    for p in nl.affine_range(trip_count):
                        i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
                        
                        for idx_tuple in remaining_indices:
                            # Load current elements
                            curr = nl.load(result[i_p, idx_tuple], mask=(i_p < shape[0]))
                            
                            # Load next elements with offset
                            next_indices = i_p + 1
                            next_vals = nl.load(result[next_indices, idx_tuple], 
                                               mask=(next_indices < shape[0]) & (i_p < shape[0]))
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr, next_vals)
                            
                            # Create swapped values
                            new_curr = nl.where(swap_needed, next_vals, curr)
                            new_next = nl.where(swap_needed, curr, next_vals)
                            
                            # Store back the results
                            nl.store(result[i_p, idx_tuple], value=new_curr, mask=(i_p < shape[0]))
                            nl.store(result[next_indices, idx_tuple], value=new_next, 
                                    mask=(next_indices < shape[0]) & (i_p < shape[0]))
        else:
            # Sort along non-first dimension
            # This implementation handles 2D tensors with sort along dim=1
            if len(shape) == 2 and dim == 1:
                max_tile_size = nl.tile_size.pmax
                trip_count = math.ceil(shape[0] / max_tile_size)
                
                # Copy input to result
                for p in nl.affine_range(trip_count):
                    i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
                    i_f = nl.arange(shape[1])[None, :]
                    
                    # Load and store with proper masking
                    x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < shape[0]))
                    nl.store(result[i_p, i_f], value=x_tile, mask=(i_p < shape[0]))
                
                # Perform bubble sort along dimension 1
                for i in nl.affine_range(sort_dim_size):
                    for j in nl.affine_range(sort_dim_size - 1):
                        for p in nl.affine_range(trip_count):
                            i_p = p * max_tile_size + nl.arange(max_tile_size)[:, None]
                            i_f = j * nl.ones((max_tile_size, 1), dtype=nl.int32)
                            i_f_next = (j + 1) * nl.ones((max_tile_size, 1), dtype=nl.int32)
                            
                            # Load current elements
                            curr = nl.load(result[i_p, i_f], mask=(i_p < shape[0]) & (i_f < shape[1]))
                            
                            # Load next elements
                            next_vals = nl.load(result[i_p, i_f_next], 
                                              mask=(i_p < shape[0]) & (i_f_next < shape[1]))
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr, next_vals)
                            
                            # Create swapped values
                            new_curr = nl.where(swap_needed, next_vals, curr)
                            new_next = nl.where(swap_needed, curr, next_vals)
                            
                            # Store back the results
                            nl.store(result[i_p, i_f], value=new_curr, 
                                    mask=(i_p < shape[0]) & (i_f < shape[1]))
                            nl.store(result[i_p, i_f_next], value=new_next, 
                                    mask=(i_p < shape[0]) & (i_f_next < shape[1]))
    
    return result