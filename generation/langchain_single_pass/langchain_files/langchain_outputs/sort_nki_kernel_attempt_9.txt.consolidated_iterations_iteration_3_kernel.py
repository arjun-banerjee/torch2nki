from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and handle negative dimensions
    tensor_shape = a_tensor.shape
    ndim = len(tensor_shape)
    
    if dim < 0:
        dim = ndim + dim
        
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle 1D case separately for simplicity
    if ndim == 1:
        # Copy input to result first
        sz = tensor_shape[0]
        trip_count = math.ceil(sz / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < sz))
            nl.store(result[i_p], value=in_tile, mask=(i_p < sz))
        
        # Bubble sort implementation for 1D
        for i in range(sz):
            for j in range(0, sz-i-1):
                # Load adjacent elements
                j_idx = nl.full((), j, dtype=nl.int32)
                j_next_idx = nl.full((), j+1, dtype=nl.int32)
                
                val_j = nl.load(result[j_idx])
                val_j_next = nl.load(result[j_next_idx])
                
                # Compare and swap if needed
                swap_needed = nl.greater(val_j, val_j_next)
                
                if swap_needed.item():
                    nl.store(result[j_idx], val_j_next)
                    nl.store(result[j_next_idx], val_j)
    else:
        # For multi-dimensional tensors, we sort along the specified dimension
        # First copy input to result
        # Calculate the total size
        total_size = 1
        for i in range(ndim):
            total_size *= tensor_shape[i]
            
        # Calculate sizes before and after the sort dimension
        outer_size = 1
        for i in range(dim):
            outer_size *= tensor_shape[i]
            
        sort_size = tensor_shape[dim]
        
        inner_size = 1
        for i in range(dim + 1, ndim):
            inner_size *= tensor_shape[i]
            
        # Copy input to result first
        trip_count = math.ceil(total_size / nl.tile_size.pmax)
        flat_tensor_size = total_size
        
        for p in nl.affine_range(trip_count):
            flat_idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Create mask for valid indices
            mask = flat_idx < flat_tensor_size
            
            # Load input data
            in_tile = nl.load(a_tensor.reshape((-1,))[flat_idx], mask=mask)
            
            # Store to result
            nl.store(result.reshape((-1,))[flat_idx], value=in_tile, mask=mask)
        
        # Now sort each slice along the specified dimension
        for outer in range(outer_size):
            for inner in range(inner_size):
                # Perform bubble sort on each slice
                for i in range(sort_size):
                    for j in range(0, sort_size-i-1):
                        # Calculate flat indices for the two elements to compare
                        flat_idx1 = outer * sort_size * inner_size + j * inner_size + inner
                        flat_idx2 = outer * sort_size * inner_size + (j+1) * inner_size + inner
                        
                        idx1 = nl.full((), flat_idx1, dtype=nl.int32)
                        idx2 = nl.full((), flat_idx2, dtype=nl.int32)
                        
                        # Load values
                        val1 = nl.load(result.reshape((-1,))[idx1])
                        val2 = nl.load(result.reshape((-1,))[idx2])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(val1, val2)
                        
                        if swap_needed.item():
                            nl.store(result.reshape((-1,))[idx1], val2)
                            nl.store(result.reshape((-1,))[idx2], val1)
    
    return result