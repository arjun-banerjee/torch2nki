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
        trip_count = math.ceil(shape[0] / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            input_tile = nl.load(a_tensor[idx], mask=(idx < shape[0]))
            nl.store(result[idx], value=input_tile, mask=(idx < shape[0]))
            
        # Bubble sort for 1D tensor
        n = shape[0]
        for i in range(n):
            for j in range(n - i - 1):
                # We need to process in tiles to respect hardware limitations
                trip_count = math.ceil(n / nl.tile_size.pmax)
                
                for p in nl.affine_range(trip_count):
                    idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Only process indices that are within bounds and relevant to current j
                    mask = (idx < n - i - 1) & (idx == j)
                    
                    if j < n - i - 1:  # Static check to avoid unnecessary computations
                        # Load current and next elements
                        curr = nl.load(result[j], mask=(j < n - i - 1))
                        next_elem = nl.load(result[j+1], mask=(j < n - i - 1))
                        
                        # Compare and swap if needed
                        should_swap = nl.greater(curr, next_elem)
                        
                        # Always perform the operations, but use conditional store
                        temp = nl.where(should_swap, next_elem, curr)
                        next_temp = nl.where(should_swap, curr, next_elem)
                        
                        # Store back the values
                        nl.store(result[j], value=temp, mask=(j < n - i - 1))
                        nl.store(result[j+1], value=next_temp, mask=(j < n - i - 1))
    else:
        # For multi-dimensional tensors, we need to sort along the specified dimension
        # Reshape to handle the specified dimension
        
        # Calculate shapes for processing
        pre_dim_size = 1
        for i in range(dim):
            pre_dim_size *= shape[i]
            
        dim_size = shape[dim]
        
        post_dim_size = 1
        for i in range(dim + 1, ndim):
            post_dim_size *= shape[i]
            
        # First copy the input to result
        for pre in range(pre_dim_size):
            for post in range(post_dim_size):
                trip_count = math.ceil(dim_size / nl.tile_size.pmax)
                
                for p in nl.affine_range(trip_count):
                    idx_dim = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Create index tuple for the current slice
                    idx = []
                    idx_flat = pre * dim_size * post_dim_size + idx_dim * post_dim_size + post
                    
                    # Extract the multi-dimensional indices
                    remaining = idx_flat
                    for i in range(ndim):
                        if i == ndim - 1:
                            idx.append(remaining)
                        else:
                            size = 1
                            for j in range(i + 1, ndim):
                                size *= shape[j]
                            idx.append(remaining // size)
                            remaining = remaining % size
                    
                    # Load input data and store to result
                    input_tile = nl.load(a_tensor[tuple(idx)], mask=(idx_dim < dim_size))
                    nl.store(result[tuple(idx)], value=input_tile, mask=(idx_dim < dim_size))
                    
                # Sort along dimension dim for this slice
                for i in range(dim_size):
                    for j in range(dim_size - i - 1):
                        # Create indices for the current and next elements
                        curr_idx = []
                        next_idx = []
                        
                        # Fill indices for dimensions before dim
                        for d in range(dim):
                            curr_idx.append(pre // (math.prod(shape[d+1:dim]) if d < dim - 1 else 1) % shape[d])
                            next_idx.append(pre // (math.prod(shape[d+1:dim]) if d < dim - 1 else 1) % shape[d])
                        
                        # Add indices for dimension dim
                        curr_idx.append(j)
                        next_idx.append(j + 1)
                        
                        # Fill indices for dimensions after dim
                        for d in range(dim + 1, ndim):
                            curr_idx.append(post // (math.prod(shape[d+1:]) if d < ndim - 1 else 1) % shape[d])
                            next_idx.append(post // (math.prod(shape[d+1:]) if d < ndim - 1 else 1) % shape[d])
                        
                        # Compare and swap if needed
                        if j < dim_size - i - 1:  # Static check
                            curr = nl.load(result[tuple(curr_idx)], mask=(j < dim_size - i - 1))
                            next_elem = nl.load(result[tuple(next_idx)], mask=(j < dim_size - i - 1))
                            
                            should_swap = nl.greater(curr, next_elem)
                            
                            # Always perform operations but use conditional store
                            temp = nl.where(should_swap, next_elem, curr)
                            next_temp = nl.where(should_swap, curr, next_elem)
                            
                            nl.store(result[tuple(curr_idx)], value=temp, mask=(j < dim_size - i - 1))
                            nl.store(result[tuple(next_idx)], value=next_temp, mask=(j < dim_size - i - 1))
    
    return result