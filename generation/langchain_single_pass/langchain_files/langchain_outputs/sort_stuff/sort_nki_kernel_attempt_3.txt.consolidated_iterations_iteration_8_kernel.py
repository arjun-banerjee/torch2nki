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
    
    # Handle 1D tensor case
    if len(a_tensor.shape) == 1:
        size = a_tensor.shape[0]
        max_tile_size = nl.tile_size.pmax
        trip_count = math.ceil(size / max_tile_size)
        
        # First copy the input to result
        for p in nl.affine_range(trip_count):
            start_idx = p * max_tile_size
            # Create indices for current tile
            # Load input data with masking to handle boundary
            input_tile = nl.load(a_tensor[start_idx:start_idx+max_tile_size], 
                               mask=(start_idx + nl.arange(max_tile_size) < size))
            # Store to result
            nl.store(result[start_idx:start_idx+max_tile_size], value=input_tile,
                   mask=(start_idx + nl.arange(max_tile_size) < size))
        
        # Bubble sort implementation
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                for p in nl.affine_range(trip_count):
                    start_idx = p * max_tile_size
                    # Load current segment
                    segment = nl.load(result[start_idx:start_idx+max_tile_size],
                                   mask=(start_idx + nl.arange(max_tile_size) < size))
                    
                    # Process each element in the tile
                    for k in nl.affine_range(max_tile_size - 1):
                        idx = start_idx + k
                        if idx < size - 1:  # Ensure we're within bounds
                            # Load the two adjacent elements to compare
                            a = nl.load(result[idx])
                            b = nl.load(result[idx + 1])
                            
                            # Compare and swap if needed
                            condition = nl.greater(a, b)
                            new_a = nl.where(condition, b, a)
                            new_b = nl.where(condition, a, b)
                            
                            # Store the results back
                            nl.store(result[idx], value=new_a)
                            nl.store(result[idx + 1], value=new_b)
    
    # Handle multi-dimensional tensor
    else:
        # Determine the size of the dimension to sort along
        sort_dim_size = a_tensor.shape[dim]
        
        # Calculate the number of slices we need to sort
        slice_shape = list(a_tensor.shape)
        slice_shape.pop(dim)
        total_slices = 1
        for s in slice_shape:
            total_slices *= s
        
        # Copy the input to result first
        for i in nl.affine_range(total_slices):
            for j in nl.affine_range(sort_dim_size):
                # Calculate linear indices
                flat_idx = i * sort_dim_size + j
                
                # Calculate multi-dimensional indices
                multi_idx = []
                temp_idx = flat_idx
                for d in range(len(a_tensor.shape)):
                    if d != dim:
                        dim_size = a_tensor.shape[d]
                        multi_idx.append(temp_idx % dim_size)
                        temp_idx //= dim_size
                
                # Insert the sort dimension index
                multi_idx.insert(dim, j)
                
                # Load value from input and store to result
                val = nl.load(a_tensor[tuple(multi_idx)])
                nl.store(result[tuple(multi_idx)], value=val)
        
        # Bubble sort each slice
        for i in nl.affine_range(total_slices):
            for outer in nl.affine_range(sort_dim_size):
                for inner in nl.affine_range(sort_dim_size - 1):
                    # Calculate multi-dimensional indices for adjacent elements
                    idx1 = []
                    idx2 = []
                    temp_idx = i
                    for d in range(len(a_tensor.shape)):
                        if d != dim:
                            dim_size = a_tensor.shape[d]
                            idx1.append(temp_idx % dim_size)
                            idx2.append(temp_idx % dim_size)
                            temp_idx //= dim_size
                    
                    # Insert the sort dimension indices
                    idx1.insert(dim, inner)
                    idx2.insert(dim, inner + 1)
                    
                    # Load the two adjacent elements
                    a = nl.load(result[tuple(idx1)])
                    b = nl.load(result[tuple(idx2)])
                    
                    # Compare and swap if needed
                    condition = nl.greater(a, b)
                    new_a = nl.where(condition, b, a)
                    new_b = nl.where(condition, a, b)
                    
                    # Store the results back
                    nl.store(result[tuple(idx1)], value=new_a)
                    nl.store(result[tuple(idx2)], value=new_b)
    
    return result