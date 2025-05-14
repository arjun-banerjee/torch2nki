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
    
    # First, copy the input tensor to the result tensor
    # We'll need to handle different dimensions differently
    
    # For 1D tensors
    if ndim == 1:
        # Calculate the number of tiles needed
        sz = shape[0]
        tile_size = min(sz, nl.tile_size.pmax)
        trip_count = math.ceil(sz / tile_size)
        
        # Copy data to result tensor
        for p in nl.affine_range(trip_count):
            i_p = p * tile_size + nl.arange(tile_size)
            
            # Load data
            src_tile = nl.load(a_tensor[i_p], mask=(i_p < sz))
            
            # Store data to result
            nl.store(result[i_p], value=src_tile, mask=(i_p < sz))
            
        # Now perform bubble sort on the entire tensor
        # We need to do multiple passes
        for _ in nl.static_range(sz):
            # For each pass, we compare adjacent elements
            for p in nl.affine_range(trip_count):
                i_p = p * tile_size + nl.arange(tile_size)
                
                # Load current chunk
                current_tile = nl.load(result[i_p], mask=(i_p < sz))
                
                # For each element (except the last one), compare with next element
                for i in nl.static_range(tile_size - 1):
                    # Check if we're at a valid position and not at the end of the tensor
                    valid_idx = (i_p[i] < sz - 1)
                    
                    if valid_idx:
                        # Load the next element if it's in a different tile
                        if i == tile_size - 1 and p < trip_count - 1:
                            next_val = nl.load(result[i_p[i] + 1])
                        else:
                            next_val = current_tile[i+1]
                        
                        # Compare and swap if needed
                        current_val = current_tile[i]
                        
                        # Use where to conditionally swap
                        swap_needed = nl.greater(current_val, next_val)
                        if i == tile_size - 1 and p < trip_count - 1:
                            # Cross-tile swap
                            new_current = nl.where(swap_needed, next_val, current_val)
                            new_next = nl.where(swap_needed, current_val, next_val)
                            current_tile = current_tile.at[i].set(new_current)
                            nl.store(result[i_p[i] + 1], value=new_next)
                        else:
                            # Within-tile swap
                            new_current = nl.where(swap_needed, next_val, current_val)
                            new_next = nl.where(swap_needed, current_val, next_val)
                            current_tile = current_tile.at[i].set(new_current)
                            current_tile = current_tile.at[i+1].set(new_next)
                
                # Store updated tile
                nl.store(result[i_p], value=current_tile, mask=(i_p < sz))
    
    # For 2D tensors
    elif ndim == 2:
        sz_0, sz_1 = shape
        
        # Determine which dimension to sort along
        if dim == 0:
            # Sort along first dimension
            for j in nl.affine_range(sz_1):
                # Calculate the number of tiles needed for dimension 0
                tile_size = min(sz_0, nl.tile_size.pmax)
                trip_count = math.ceil(sz_0 / tile_size)
                
                # Copy data to result tensor
                for p in nl.affine_range(trip_count):
                    i_p = p * tile_size + nl.arange(tile_size)
                    
                    # Load data
                    src_tile = nl.load(a_tensor[i_p, j], mask=(i_p < sz_0))
                    
                    # Store data to result
                    nl.store(result[i_p, j], value=src_tile, mask=(i_p < sz_0))
                
                # Now perform bubble sort on this column
                for _ in nl.static_range(sz_0):
                    for p in nl.affine_range(trip_count):
                        i_p = p * tile_size + nl.arange(tile_size)
                        
                        # Load current chunk
                        current_tile = nl.load(result[i_p, j], mask=(i_p < sz_0))
                        
                        # For each element (except the last one), compare with next element
                        for i in nl.static_range(tile_size - 1):
                            # Check if we're at a valid position and not at the end
                            valid_idx = (i_p[i] < sz_0 - 1)
                            
                            if valid_idx:
                                # Load the next element if it's in a different tile
                                if i == tile_size - 1 and p < trip_count - 1:
                                    next_val = nl.load(result[i_p[i] + 1, j])
                                else:
                                    next_val = current_tile[i+1]
                                
                                # Compare and swap if needed
                                current_val = current_tile[i]
                                
                                # Use where to conditionally swap
                                swap_needed = nl.greater(current_val, next_val)
                                if i == tile_size - 1 and p < trip_count - 1:
                                    # Cross-tile swap
                                    new_current = nl.where(swap_needed, next_val, current_val)
                                    new_next = nl.where(swap_needed, current_val, next_val)
                                    current_tile = current_tile.at[i].set(new_current)
                                    nl.store(result[i_p[i] + 1, j], value=new_next)
                                else:
                                    # Within-tile swap
                                    new_current = nl.where(swap_needed, next_val, current_val)
                                    new_next = nl.where(swap_needed, current_val, next_val)
                                    current_tile = current_tile.at[i].set(new_current)
                                    current_tile = current_tile.at[i+1].set(new_next)
                        
                        # Store updated tile
                        nl.store(result[i_p, j], value=current_tile, mask=(i_p < sz_0))
        
        else:  # dim == 1
            # Sort along second dimension
            for i in nl.affine_range(sz_0):
                # Calculate the number of tiles needed for dimension 1
                tile_size = min(sz_1, nl.tile_size.pmax)
                trip_count = math.ceil(sz_1 / tile_size)
                
                # Copy data to result tensor
                for p in nl.affine_range(trip_count):
                    i_p = p * tile_size + nl.arange(tile_size)
                    
                    # Load data
                    src_tile = nl.load(a_tensor[i, i_p], mask=(i_p < sz_1))
                    
                    # Store data to result
                    nl.store(result[i, i_p], value=src_tile, mask=(i_p < sz_1))
                
                # Now perform bubble sort on this row
                for _ in nl.static_range(sz_1):
                    for p in nl.affine_range(trip_count):
                        i_p = p * tile_size + nl.arange(tile_size)
                        
                        # Load current chunk
                        current_tile = nl.load(result[i, i_p], mask=(i_p < sz_1))
                        
                        # For each element (except the last one), compare with next element
                        for j in nl.static_range(tile_size - 1):
                            # Check if we're at a valid position and not at the end
                            valid_idx = (i_p[j] < sz_1 - 1)
                            
                            if valid_idx:
                                # Load the next element if it's in a different tile
                                if j == tile_size - 1 and p < trip_count - 1:
                                    next_val = nl.load(result[i, i_p[j] + 1])
                                else:
                                    next_val = current_tile[j+1]
                                
                                # Compare and swap if needed
                                current_val = current_tile[j]
                                
                                # Use where to conditionally swap
                                swap_needed = nl.greater(current_val, next_val)
                                if j == tile_size - 1 and p < trip_count - 1:
                                    # Cross-tile swap
                                    new_current = nl.where(swap_needed, next_val, current_val)
                                    new_next = nl.where(swap_needed, current_val, next_val)
                                    current_tile = current_tile.at[j].set(new_current)
                                    nl.store(result[i, i_p[j] + 1], value=new_next)
                                else:
                                    # Within-tile swap
                                    new_current = nl.where(swap_needed, next_val, current_val)
                                    new_next = nl.where(swap_needed, current_val, next_val)
                                    current_tile = current_tile.at[j].set(new_current)
                                    current_tile = current_tile.at[j+1].set(new_next)
                        
                        # Store updated tile
                        nl.store(result[i, i_p], value=current_tile, mask=(i_p < sz_1))
    
    return result