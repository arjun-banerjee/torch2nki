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
    if ndim == 1:
        # Handle 1D tensor case
        size = shape[0]
        max_tile_size = nl.tile_size.pmax
        
        # Process the tensor in tiles to respect hardware limitations
        trip_count = math.ceil(size / max_tile_size)
        
        for i in nl.affine_range(trip_count):
            # Calculate start and end indices for this tile
            start_idx = i * max_tile_size
            
            # Load a tile from input tensor
            indices = nl.arange(max_tile_size)
            x_tile = nl.load(a_tensor[start_idx + indices], mask=(start_idx + indices < size))
            
            # Sort the tile using bubble sort
            for j in nl.static_range(max_tile_size):
                for k in nl.static_range(max_tile_size - 1):
                    # Compare adjacent elements
                    cond = nl.less(k + 1, max_tile_size) & nl.less(x_tile[k], x_tile[k+1])
                    # Swap if needed
                    tmp = nl.where(cond, x_tile[k+1], x_tile[k])
                    x_tile = nl.where(cond & nl.equal(nl.arange(max_tile_size), k), x_tile[k], x_tile)
                    x_tile = nl.where(cond & nl.equal(nl.arange(max_tile_size), k+1), tmp, x_tile)
            
            # Store the sorted tile to result
            nl.store(result[start_idx + indices], value=x_tile, mask=(start_idx + indices < size))
            
    elif ndim == 2:
        # Handle 2D tensor case
        if dim == 0:
            # Sort along dimension 0
            rows, cols = shape
            max_tile_size = nl.tile_size.pmax
            
            # Process the tensor in tiles to respect hardware limitations
            row_trips = math.ceil(rows / max_tile_size)
            
            for j in nl.affine_range(cols):
                # For each column, sort all rows
                for i_trip in nl.affine_range(row_trips):
                    start_row = i_trip * max_tile_size
                    
                    # Load a column tile
                    row_indices = nl.arange(max_tile_size)[:, None]
                    col_idx = nl.full((max_tile_size, 1), j, dtype=nl.int32)
                    x_tile = nl.load(a_tensor[start_row + row_indices, col_idx], mask=(start_row + row_indices < rows))
                    
                    # Sort the column tile using bubble sort
                    for p in nl.static_range(max_tile_size):
                        for q in nl.static_range(max_tile_size - 1):
                            cond = nl.less(q + 1, max_tile_size) & nl.less(x_tile[q], x_tile[q+1])
                            # Swap if needed
                            tmp = x_tile[q+1]
                            x_tile = nl.where(cond & nl.equal(nl.arange(max_tile_size)[:, None], q), x_tile[q+1], x_tile)
                            x_tile = nl.where(cond & nl.equal(nl.arange(max_tile_size)[:, None], q+1), x_tile[q], x_tile)
                    
                    # Store the sorted column tile to result
                    nl.store(result[start_row + row_indices, col_idx], value=x_tile, mask=(start_row + row_indices < rows))
                    
        else:  # dim == 1
            # Sort along dimension 1
            rows, cols = shape
            max_tile_size = nl.tile_size.pmax
            
            # Process the tensor in tiles to respect hardware limitations
            row_trips = math.ceil(rows / max_tile_size)
            col_trips = math.ceil(cols / max_tile_size)
            
            for i_trip in nl.affine_range(row_trips):
                start_row = i_trip * max_tile_size
                row_indices = nl.arange(max_tile_size)[:, None]
                
                for j_trip in nl.affine_range(col_trips):
                    start_col = j_trip * max_tile_size
                    col_indices = nl.arange(max_tile_size)[None, :]
                    
                    # Load a row tile
                    x_tile = nl.load(a_tensor[start_row + row_indices, start_col + col_indices], 
                                     mask=((start_row + row_indices < rows) & (start_col + col_indices < cols)))
                    
                    # Sort each row in the tile using bubble sort
                    for i in nl.static_range(max_tile_size):
                        if start_row + i < rows:  # Only process valid rows
                            for p in nl.static_range(max_tile_size):
                                for q in nl.static_range(max_tile_size - 1):
                                    if q + 1 < max_tile_size and start_col + q + 1 < cols:
                                        # Compare adjacent elements in the row
                                        if x_tile[i, q] < x_tile[i, q+1]:
                                            # Swap elements
                                            tmp = x_tile[i, q]
                                            x_tile[i, q] = x_tile[i, q+1]
                                            x_tile[i, q+1] = tmp
                    
                    # Store the sorted row tile to result
                    nl.store(result[start_row + row_indices, start_col + col_indices], value=x_tile,
                             mask=((start_row + row_indices < rows) & (start_col + col_indices < cols)))
    
    # Copy input tensor to result for dimensions we haven't explicitly handled
    else:
        for i in nl.affine_range(math.ceil(a_tensor.size / nl.tile_size.pmax)):
            start = i * nl.tile_size.pmax
            end = min(start + nl.tile_size.pmax, a_tensor.size)
            indices = nl.arange(nl.tile_size.pmax)
            nl.store(result.reshape(-1)[start + indices], 
                     value=nl.load(a_tensor.reshape(-1)[start + indices], 
                                  mask=(start + indices < a_tensor.size)),
                     mask=(start + indices < a_tensor.size))
    
    return result