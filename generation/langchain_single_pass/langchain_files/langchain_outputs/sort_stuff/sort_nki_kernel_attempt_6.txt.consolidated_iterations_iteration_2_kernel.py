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
    size = shape[dim]
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to result
    # We need to process the tensor in tiles to respect hardware limitations
    if len(shape) == 1:
        # For 1D tensor, sort along the only dimension
        trip_count = math.ceil(shape[0] / nl.tile_size.pmax)
        for p in nl.affine_range(trip_count):
            # Generate indices for the current tile
            idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data from external memory to on-chip memory
            x_tile = nl.load(a_tensor[idx], mask=(idx < shape[0]))
            
            # Store the data to result (before sorting)
            nl.store(result[idx], value=x_tile, mask=(idx < shape[0]))
    
    elif len(shape) == 2:
        # For 2D tensor
        if dim == 0:
            # Sort along dimension 0 (rows)
            trip_count = math.ceil(shape[1] / nl.tile_size.pmax)
            for p in nl.affine_range(trip_count):
                # Generate indices for the current tile
                idx_f = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                idx_p = nl.arange(shape[0])[:, None]
                
                # Load input data from external memory to on-chip memory
                x_tile = nl.load(a_tensor[idx_p, idx_f], mask=(idx_f < shape[1]))
                
                # Store the data to result (before sorting)
                nl.store(result[idx_p, idx_f], value=x_tile, mask=(idx_f < shape[1]))
        else:
            # Sort along dimension 1 (columns)
            trip_count = math.ceil(shape[0] / nl.tile_size.pmax)
            for p in nl.affine_range(trip_count):
                # Generate indices for the current tile
                idx_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                idx_f = nl.arange(shape[1])[None, :]
                
                # Load input data from external memory to on-chip memory
                x_tile = nl.load(a_tensor[idx_p, idx_f], mask=(idx_p < shape[0]))
                
                # Store the data to result (before sorting)
                nl.store(result[idx_p, idx_f], value=x_tile, mask=(idx_p < shape[0]))
    
    # Now perform bubble sort on the result tensor
    if len(shape) == 1:
        # For 1D tensor
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                trip_count = math.ceil(shape[0] / nl.tile_size.pmax)
                for p in nl.affine_range(trip_count):
                    # Generate indices for the current tile
                    idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Load current values
                    current = nl.load(result[idx], mask=(idx < shape[0] - 1))
                    next_val = nl.load(result[idx + 1], mask=(idx < shape[0] - 1))
                    
                    # Compare and swap if needed
                    swap = nl.greater(current, next_val)
                    
                    # Where swap is true, store next_val in current position
                    temp = nl.zeros(current.shape, dtype=current.dtype)
                    temp = nl.where(swap, next_val, current)
                    nl.store(result[idx], value=temp, mask=(idx < shape[0] - 1))
                    
                    # Where swap is true, store current in next position
                    temp = nl.where(swap, current, next_val)
                    nl.store(result[idx + 1], value=temp, mask=(idx < shape[0] - 1))
    
    elif len(shape) == 2:
        if dim == 0:
            # Sort along dimension 0 (rows)
            for i in nl.affine_range(size):
                for j in nl.affine_range(size - 1):
                    trip_count = math.ceil(shape[1] / nl.tile_size.pmax)
                    for p in nl.affine_range(trip_count):
                        # Generate indices for the current tile
                        idx_f = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                        
                        # Load current and next values
                        current = nl.load(result[j, idx_f], mask=(idx_f < shape[1]))
                        next_val = nl.load(result[j + 1, idx_f], mask=(idx_f < shape[1]))
                        
                        # Compare and swap if needed
                        swap = nl.greater(current, next_val)
                        
                        # Where swap is true, store next_val in current position
                        temp = nl.zeros(current.shape, dtype=current.dtype)
                        temp = nl.where(swap, next_val, current)
                        nl.store(result[j, idx_f], value=temp, mask=(idx_f < shape[1]))
                        
                        # Where swap is true, store current in next position
                        temp = nl.where(swap, current, next_val)
                        nl.store(result[j + 1, idx_f], value=temp, mask=(idx_f < shape[1]))
        else:
            # Sort along dimension 1 (columns)
            trip_count = math.ceil(shape[0] / nl.tile_size.pmax)
            for p in nl.affine_range(trip_count):
                # Generate indices for the current tile
                idx_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                idx_f = nl.arange(shape[1])[None, :]
                
                # Load the current tile
                tile = nl.load(result[idx_p, idx_f], mask=(idx_p < shape[0]))
                
                # Perform bubble sort on each row of the tile
                for i in nl.affine_range(size):
                    for j in nl.affine_range(size - 1):
                        # Create a temporary buffer for the current row
                        temp_row = nl.zeros((shape[1],), dtype=a_tensor.dtype)
                        
                        # For each row in the tile
                        for row in nl.affine_range(nl.tile_size.pmax):
                            if row >= shape[0]:
                                continue
                                
                            # Get current row data
                            row_idx = p * nl.tile_size.pmax + row
                            row_data = nl.load(result[row_idx, :])
                            
                            # Compare adjacent elements
                            for col in nl.affine_range(shape[1] - 1):
                                if row_data[col] > row_data[col + 1]:
                                    # Swap elements
                                    temp = row_data[col]
                                    row_data[col] = row_data[col + 1]
                                    row_data[col + 1] = temp
                            
                            # Store the sorted row back
                            nl.store(result[row_idx, :], value=row_data)
    
    return result