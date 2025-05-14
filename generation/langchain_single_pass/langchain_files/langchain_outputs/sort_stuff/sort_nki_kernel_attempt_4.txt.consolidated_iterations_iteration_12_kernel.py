from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # If dim is negative, convert to positive index
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy the input to result
    if ndim == 1:
        # For 1D tensor
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            input_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            nl.store(result[i_p], input_tile, mask=(i_p < size))
    else:
        # For N-D tensor
        # We'll first create a flattened copy of the input
        total_size = 1
        for i in range(ndim):
            if i != dim:
                total_size *= shape[i]
        
        # Handle different dimension cases
        if dim == 0:
            # Sort along first dimension
            size_dim = shape[0]
            size_rest = total_size // size_dim
            
            # Copy input to result
            for p in nl.affine_range(math.ceil(size_dim / nl.tile_size.pmax)):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                i_f = nl.arange(size_rest)[None, :]
                
                input_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < size_dim))
                nl.store(result[i_p, i_f], input_tile, mask=(i_p < size_dim))
        elif dim == ndim - 1:
            # Sort along last dimension
            size_rest = total_size // shape[dim]
            size_dim = shape[dim]
            
            # Copy input to result
            for p in nl.affine_range(math.ceil(size_rest / nl.tile_size.pmax)):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                i_f = nl.arange(size_dim)[None, :]
                
                input_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < size_rest))
                nl.store(result[i_p, i_f], input_tile, mask=(i_p < size_rest))
        else:
            # For middle dimensions, we need to copy everything
            trip_count = math.ceil(total_size / nl.tile_size.pmax)
            
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                input_tile = nl.load(a_tensor.reshape((-1,))[i_p], mask=(i_p < total_size))
                nl.store(result.reshape((-1,))[i_p], input_tile, mask=(i_p < total_size))
    
    # Now perform bubble sort along the specified dimension
    if ndim == 1:
        # Simple case: 1D tensor
        size = shape[0]
        
        # Bubble sort
        for i in nl.affine_range(size - 1):
            for j in nl.affine_range(size - 1 - i):
                # Load current and next elements
                idx_curr = nl.arange(1)
                idx_next = nl.arange(1) + 1
                
                curr = nl.load(result[j])
                next_val = nl.load(result[j + 1])
                
                # Swap if current > next
                condition = nl.greater(curr, next_val)
                
                # Use where to conditionally swap values
                new_curr = nl.where(condition, next_val, curr)
                new_next = nl.where(condition, curr, next_val)
                
                # Store back
                nl.store(result[j], new_curr)
                nl.store(result[j + 1], new_next)
    else:
        # N-D tensor case
        if dim == ndim - 1:
            # Sort along the last dimension
            size_rest = total_size // shape[dim]
            size_dim = shape[dim]
            
            for p in nl.affine_range(size_rest):
                # Bubble sort each slice
                for i in nl.affine_range(size_dim - 1):
                    for j in nl.affine_range(size_dim - 1 - i):
                        # Load current and next elements
                        curr = nl.load(result[p, j])
                        next_val = nl.load(result[p, j + 1])
                        
                        # Swap if current > next
                        condition = nl.greater(curr, next_val)
                        
                        # Use where to conditionally swap values
                        new_curr = nl.where(condition, next_val, curr)
                        new_next = nl.where(condition, curr, next_val)
                        
                        # Store back
                        nl.store(result[p, j], new_curr)
                        nl.store(result[p, j + 1], new_next)
        elif dim == 0:
            # Sort along the first dimension
            size_dim = shape[0]
            size_rest = total_size // size_dim
            
            for f in nl.affine_range(size_rest):
                # Bubble sort each column
                for i in nl.affine_range(size_dim - 1):
                    for j in nl.affine_range(size_dim - 1 - i):
                        # Load current and next elements
                        curr = nl.load(result[j, f])
                        next_val = nl.load(result[j + 1, f])
                        
                        # Swap if current > next
                        condition = nl.greater(curr, next_val)
                        
                        # Use where to conditionally swap values
                        new_curr = nl.where(condition, next_val, curr)
                        new_next = nl.where(condition, curr, next_val)
                        
                        # Store back
                        nl.store(result[j, f], new_curr)
                        nl.store(result[j + 1, f], new_next)
        else:
            # Middle dimensions - reshape, sort, and reshape back
            # This is a simplified approach for demonstration
            # For production code, would need more complex handling
            sorted_flat = nl.zeros((total_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
            for i in nl.affine_range(total_size):
                sorted_flat[i] = nl.load(result.reshape((-1,))[i])
            
            # Simple bubble sort on flat array
            for i in nl.affine_range(total_size - 1):
                for j in nl.affine_range(total_size - 1 - i):
                    curr = sorted_flat[j]
                    next_val = sorted_flat[j + 1]
                    
                    condition = nl.greater(curr, next_val)
                    sorted_flat[j] = nl.where(condition, next_val, curr)
                    sorted_flat[j + 1] = nl.where(condition, curr, next_val)
            
            # Copy back to result
            for i in nl.affine_range(total_size):
                nl.store(result.reshape((-1,))[i], sorted_flat[i])
    
    return result