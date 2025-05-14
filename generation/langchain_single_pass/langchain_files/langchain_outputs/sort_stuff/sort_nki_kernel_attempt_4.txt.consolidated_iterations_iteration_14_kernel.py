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
    
    # Copy input to result first
    if ndim == 1:
        # 1D case
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data, with masking for boundary
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            
            # Store to result
            nl.store(result[i_p], value=in_tile, mask=(i_p < size))
    else:
        # Multi-dimensional case
        # Reshape to handle arbitrary dimensions
        outer_size = 1
        for i in range(ndim):
            if i != dim:
                outer_size *= shape[i]
        
        dim_size = shape[dim]
        
        # Iterate through all elements not in the sort dimension
        trip_count_outer = math.ceil(outer_size / nl.tile_size.pmax)
        
        for p_outer in nl.affine_range(trip_count_outer):
            # Generate indices for outer dimensions
            i_p_outer = p_outer * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load and store each slice along the sort dimension
            trip_count_inner = math.ceil(dim_size / nl.tile_size.pmax)
            
            for p_inner in nl.affine_range(trip_count_inner):
                # Generate indices for the inner dimension
                i_p_inner = p_inner * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                
                # Handle different dimension positions
                if dim == 0:
                    in_tile = nl.load(a_tensor[i_p_inner[:, None], i_p_outer[None, :]], 
                                     mask=(i_p_inner[:, None] < dim_size) & (i_p_outer[None, :] < outer_size))
                    nl.store(result[i_p_inner[:, None], i_p_outer[None, :]], value=in_tile, 
                            mask=(i_p_inner[:, None] < dim_size) & (i_p_outer[None, :] < outer_size))
                else:
                    in_tile = nl.load(a_tensor[i_p_outer[:, None], i_p_inner[None, :]], 
                                     mask=(i_p_outer[:, None] < outer_size) & (i_p_inner[None, :] < dim_size))
                    nl.store(result[i_p_outer[:, None], i_p_inner[None, :]], value=in_tile, 
                            mask=(i_p_outer[:, None] < outer_size) & (i_p_inner[None, :] < dim_size))
    
    # Now perform the bubble sort on each slice along the sort dimension
    if ndim == 1:
        # 1D case - sort the entire array
        n = shape[0]
        
        # Bubble sort implementation
        for i in range(n):
            for j in range(0, n-i-1):
                # Process in tiles to respect hardware limitations
                trip_count = math.ceil(1 / nl.tile_size.pmax)  # Just need 1 element at a time
                
                for p in nl.affine_range(trip_count):
                    # Load adjacent elements
                    a = nl.load(result[j])
                    b = nl.load(result[j+1])
                    
                    # Compare and swap if needed
                    condition = nl.greater(a, b)
                    
                    # Use where to conditionally swap values
                    new_a = nl.where(condition, b, a)
                    new_b = nl.where(condition, a, b)
                    
                    # Store the results back
                    nl.store(result[j], value=new_a)
                    nl.store(result[j+1], value=new_b)
    else:
        # Multi-dimensional case
        n = shape[dim]
        
        # Handle different dimension positions for sorting
        if dim == 0:
            # Sort along first dimension
            for i in range(n):
                for j in range(0, n-i-1):
                    # Process in tiles to respect hardware limitations
                    trip_count = math.ceil(shape[1] / nl.tile_size.pmax)
                    
                    for p in nl.affine_range(trip_count):
                        # Generate indices for the current tile
                        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                        
                        # Load adjacent rows, with masking for boundary
                        a = nl.load(result[j, i_p], mask=(i_p < shape[1]))
                        b = nl.load(result[j+1, i_p], mask=(i_p < shape[1]))
                        
                        # Compare and swap if needed
                        condition = nl.greater(a, b)
                        
                        # Use where to conditionally swap values
                        new_a = nl.where(condition, b, a)
                        new_b = nl.where(condition, a, b)
                        
                        # Store the results back
                        nl.store(result[j, i_p], value=new_a, mask=(i_p < shape[1]))
                        nl.store(result[j+1, i_p], value=new_b, mask=(i_p < shape[1]))
        else:
            # Sort along last dimension
            for i in range(n):
                for j in range(0, n-i-1):
                    # Process in tiles to respect hardware limitations
                    trip_count = math.ceil(shape[0] / nl.tile_size.pmax)
                    
                    for p in nl.affine_range(trip_count):
                        # Generate indices for the current tile
                        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                        
                        # Load adjacent columns, with masking for boundary
                        a = nl.load(result[i_p, j], mask=(i_p < shape[0]))
                        b = nl.load(result[i_p, j+1], mask=(i_p < shape[0]))
                        
                        # Compare and swap if needed
                        condition = nl.greater(a, b)
                        
                        # Use where to conditionally swap values
                        new_a = nl.where(condition, b, a)
                        new_b = nl.where(condition, a, b)
                        
                        # Store the results back
                        nl.store(result[i_p, j], value=new_a, mask=(i_p < shape[0]))
                        nl.store(result[i_p, j+1], value=new_b, mask=(i_p < shape[0]))
    
    return result