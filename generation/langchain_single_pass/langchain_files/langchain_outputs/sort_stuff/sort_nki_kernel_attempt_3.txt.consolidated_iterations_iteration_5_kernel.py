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
    
    # Copy input to result first
    if len(a_tensor.shape) == 1:
        # Handle 1D tensor case
        sz = a_tensor.shape[0]
        trip_count = math.ceil(sz / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < sz))
            
            # Store to result
            nl.store(result[i_p], value=in_tile, mask=(i_p < sz))
            
        # Bubble sort implementation for 1D
        for i in nl.affine_range(sz):
            for j in nl.affine_range(sz - 1):
                # Process in tiles to respect hardware limitations
                for p in nl.affine_range(trip_count):
                    # Calculate indices for this tile
                    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Only process valid indices in this tile that correspond to j
                    valid_mask = (i_p < sz - 1) & (i_p == j)
                    
                    if valid_mask.any():
                        # Load current and next element
                        current = nl.load(result[i_p], mask=valid_mask)
                        next_elem = nl.load(result[i_p + 1], mask=valid_mask)
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(current, next_elem)
                        temp = nl.where(swap_needed, next_elem, current)
                        nl.store(result[i_p], value=temp, mask=valid_mask)
                        
                        temp = nl.where(swap_needed, current, next_elem)
                        nl.store(result[i_p + 1], value=temp, mask=valid_mask)
    else:
        # Handle multi-dimensional tensor case
        # Reshape to handle sorting along specified dimension
        tensor_shape = a_tensor.shape
        sort_dim_size = tensor_shape[dim]
        
        # First, copy the input to result
        # Calculate number of elements and trip count
        total_elements = 1
        for s in tensor_shape:
            total_elements *= s
        
        # Determine how many elements to process per tile
        elements_per_tile = nl.tile_size.pmax
        trip_count = math.ceil(total_elements / elements_per_tile)
        
        # Copy input to result first
        for p in nl.affine_range(trip_count):
            # Generate linear indices for the current tile
            linear_idx = p * elements_per_tile + nl.arange(elements_per_tile)
            
            # Create multi-dimensional indices
            multi_idx = []
            remaining_idx = linear_idx
            for i in range(len(tensor_shape) - 1, -1, -1):
                multi_idx.insert(0, remaining_idx % tensor_shape[i])
                remaining_idx = remaining_idx // tensor_shape[i]
            
            # Create mask for valid indices
            valid_mask = linear_idx < total_elements
            
            # Use a different approach for indexing - we'll process each dimension separately
            if len(tensor_shape) == 2:
                # Handle 2D case specifically
                i0_max, i1_max = tensor_shape
                i0 = linear_idx // i1_max
                i1 = linear_idx % i1_max
                
                # Load from input with mask
                in_tile = nl.load(a_tensor[i0, i1], mask=valid_mask & (i0 < i0_max) & (i1 < i1_max))
                
                # Store to result
                nl.store(result[i0, i1], value=in_tile, mask=valid_mask & (i0 < i0_max) & (i1 < i1_max))
            elif len(tensor_shape) == 3:
                # Handle 3D case specifically
                i0_max, i1_max, i2_max = tensor_shape
                i0 = linear_idx // (i1_max * i2_max)
                remainder = linear_idx % (i1_max * i2_max)
                i1 = remainder // i2_max
                i2 = remainder % i2_max
                
                # Load from input with mask
                in_tile = nl.load(a_tensor[i0, i1, i2], mask=valid_mask & (i0 < i0_max) & (i1 < i1_max) & (i2 < i2_max))
                
                # Store to result
                nl.store(result[i0, i1, i2], value=in_tile, mask=valid_mask & (i0 < i0_max) & (i1 < i1_max) & (i2 < i2_max))
        
        # For each "slice" along dimensions other than dim, perform sort
        # We'll handle 2D and 3D cases specifically for simplicity
        if len(tensor_shape) == 2:
            if dim == 0:
                # Sort along rows
                for col in nl.affine_range(tensor_shape[1]):
                    # Bubble sort implementation
                    for i in nl.affine_range(tensor_shape[0]):
                        for j in nl.affine_range(tensor_shape[0] - 1):
                            # Process in tiles
                            trip_count = math.ceil(tensor_shape[0] / nl.tile_size.pmax)
                            for p in nl.affine_range(trip_count):
                                # Calculate indices for this tile
                                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                                
                                # Only process valid indices in this tile that correspond to j
                                valid_mask = (i_p < tensor_shape[0] - 1) & (i_p == j)
                                
                                if valid_mask.any():
                                    # Load current and next element
                                    current = nl.load(result[i_p, col], mask=valid_mask)
                                    next_elem = nl.load(result[i_p + 1, col], mask=valid_mask)
                                    
                                    # Compare and swap if needed
                                    swap_needed = nl.greater(current, next_elem)
                                    temp = nl.where(swap_needed, next_elem, current)
                                    nl.store(result[i_p, col], value=temp, mask=valid_mask)
                                    
                                    temp = nl.where(swap_needed, current, next_elem)
                                    nl.store(result[i_p + 1, col], value=temp, mask=valid_mask)
            else:  # dim == 1
                # Sort along columns
                for row in nl.affine_range(tensor_shape[0]):
                    # Bubble sort implementation
                    for i in nl.affine_range(tensor_shape[1]):
                        for j in nl.affine_range(tensor_shape[1] - 1):
                            # Process in tiles
                            trip_count = math.ceil(tensor_shape[1] / nl.tile_size.pmax)
                            for p in nl.affine_range(trip_count):
                                # Calculate indices for this tile
                                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                                
                                # Only process valid indices in this tile that correspond to j
                                valid_mask = (i_p < tensor_shape[1] - 1) & (i_p == j)
                                
                                if valid_mask.any():
                                    # Load current and next element
                                    current = nl.load(result[row, i_p], mask=valid_mask)
                                    next_elem = nl.load(result[row, i_p + 1], mask=valid_mask)
                                    
                                    # Compare and swap if needed
                                    swap_needed = nl.greater(current, next_elem)
                                    temp = nl.where(swap_needed, next_elem, current)
                                    nl.store(result[row, i_p], value=temp, mask=valid_mask)
                                    
                                    temp = nl.where(swap_needed, current, next_elem)
                                    nl.store(result[row, i_p + 1], value=temp, mask=valid_mask)
    
    return result