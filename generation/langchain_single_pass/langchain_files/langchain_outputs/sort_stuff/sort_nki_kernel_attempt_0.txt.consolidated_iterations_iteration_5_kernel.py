from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and handle negative dimension
    shape = a_tensor.shape
    ndim = len(shape)
    
    if dim < 0:
        dim = ndim + dim
        
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle 1D case
    if ndim == 1:
        # Copy input to result first
        size = shape[0]
        
        # Calculate number of tiles needed
        tile_size = min(size, nl.tile_size.pmax)
        trip_count = math.ceil(size / tile_size)
        
        # First load data into result
        for i in nl.affine_range(trip_count):
            start_idx = i * tile_size
            
            # Generate indices for current tile
            indices = start_idx + nl.arange(tile_size)
            
            # Load input data
            input_tile = nl.load(a_tensor[indices], mask=(indices < size))
            
            # Store to result
            nl.store(result[indices], value=input_tile, mask=(indices < size))
        
        # Bubble sort algorithm
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Load adjacent elements for comparison
                j_idx = nl.arange(size - 1)
                j_plus_one = j_idx + 1
                
                # We need to load in tiles
                for k in nl.affine_range(trip_count):
                    start_idx = k * tile_size
                    
                    # Generate indices for current tile
                    curr_indices = start_idx + nl.arange(tile_size)
                    mask = (curr_indices < (size - 1))
                    
                    # Load current and next values
                    curr_vals = nl.load(result[curr_indices], mask=mask)
                    next_indices = curr_indices + 1
                    next_vals = nl.load(result[next_indices], mask=mask)
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_vals, next_vals)
                    
                    # Where swap is needed, update values
                    new_curr = nl.where(swap_needed, next_vals, curr_vals)
                    new_next = nl.where(swap_needed, curr_vals, next_vals)
                    
                    # Store back
                    nl.store(result[curr_indices], value=new_curr, mask=mask)
                    nl.store(result[next_indices], value=new_next, mask=mask)
    
    # Handle 2D case
    elif ndim == 2:
        # If sorting along dimension 0 (rows)
        if dim == 0:
            # Get dimensions
            rows = shape[0]
            cols = shape[1]
            
            # Calculate tile sizes
            p_tile_size = min(rows, nl.tile_size.pmax)
            f_tile_size = min(cols, 512)  # Using 512 as a typical free dimension size
            
            # Calculate number of tiles needed
            p_trips = math.ceil(rows / p_tile_size)
            f_trips = math.ceil(cols / f_tile_size)
            
            # Copy input to result first
            for p in nl.affine_range(p_trips):
                p_start = p * p_tile_size
                p_indices = p_start + nl.arange(p_tile_size)[:, None]
                
                for f in nl.affine_range(f_trips):
                    f_start = f * f_tile_size
                    f_indices = f_start + nl.arange(f_tile_size)[None, :]
                    
                    # Load data
                    input_tile = nl.load(a_tensor[p_indices, f_indices], 
                                        mask=((p_indices < rows) & (f_indices < cols)))
                    
                    # Store to result
                    nl.store(result[p_indices, f_indices], value=input_tile,
                            mask=((p_indices < rows) & (f_indices < cols)))
            
            # For each column, sort the elements in that column
            for col in nl.affine_range(cols):
                for i in nl.affine_range(rows):
                    for j in nl.affine_range(rows - 1):
                        # We need to load in tiles
                        j_idx = nl.arange(rows - 1)
                        
                        for k in nl.affine_range(p_trips):
                            start_idx = k * p_tile_size
                            
                            # Generate indices for current tile
                            curr_indices = start_idx + nl.arange(p_tile_size)
                            mask = (curr_indices < (rows - 1))
                            
                            # Load current and next values
                            curr_vals = nl.load(result[curr_indices, col], mask=mask)
                            next_indices = curr_indices + 1
                            next_vals = nl.load(result[next_indices, col], mask=mask)
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr_vals, next_vals)
                            
                            # Where swap is needed, update values
                            new_curr = nl.where(swap_needed, next_vals, curr_vals)
                            new_next = nl.where(swap_needed, curr_vals, next_vals)
                            
                            # Store back
                            nl.store(result[curr_indices, col], value=new_curr, mask=mask)
                            nl.store(result[next_indices, col], value=new_next, mask=mask)
                            
        # If sorting along dimension 1 (columns)
        else:  # dim == 1
            # Get dimensions
            rows = shape[0]
            cols = shape[1]
            
            # Calculate tile sizes
            p_tile_size = min(rows, nl.tile_size.pmax)
            f_tile_size = min(cols, 512)  # Using 512 as a typical free dimension size
            
            # Calculate number of tiles needed
            p_trips = math.ceil(rows / p_tile_size)
            f_trips = math.ceil(cols / f_tile_size)
            
            # Copy input to result first
            for p in nl.affine_range(p_trips):
                p_start = p * p_tile_size
                p_indices = p_start + nl.arange(p_tile_size)[:, None]
                
                for f in nl.affine_range(f_trips):
                    f_start = f * f_tile_size
                    f_indices = f_start + nl.arange(f_tile_size)[None, :]
                    
                    # Load data
                    input_tile = nl.load(a_tensor[p_indices, f_indices], 
                                        mask=((p_indices < rows) & (f_indices < cols)))
                    
                    # Store to result
                    nl.store(result[p_indices, f_indices], value=input_tile,
                            mask=((p_indices < rows) & (f_indices < cols)))
            
            # For each row, sort the elements in that row
            for row in nl.affine_range(rows):
                for i in nl.affine_range(cols):
                    for j in nl.affine_range(cols - 1):
                        # We need to load in tiles
                        j_idx = nl.arange(cols - 1)
                        
                        for k in nl.affine_range(f_trips):
                            start_idx = k * f_tile_size
                            
                            # Generate indices for current tile
                            curr_indices = start_idx + nl.arange(f_tile_size)
                            mask = (curr_indices < (cols - 1))
                            
                            # Load current and next values
                            curr_vals = nl.load(result[row, curr_indices], mask=mask)
                            next_indices = curr_indices + 1
                            next_vals = nl.load(result[row, next_indices], mask=mask)
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr_vals, next_vals)
                            
                            # Where swap is needed, update values
                            new_curr = nl.where(swap_needed, next_vals, curr_vals)
                            new_next = nl.where(swap_needed, curr_vals, next_vals)
                            
                            # Store back
                            nl.store(result[row, curr_indices], value=new_curr, mask=mask)
                            nl.store(result[row, next_indices], value=new_next, mask=mask)
    
    return result