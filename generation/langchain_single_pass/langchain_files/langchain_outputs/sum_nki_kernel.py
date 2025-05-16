from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sum(a_tensor):
    # Initialize scalar accumulator on psum buffer
    acc_sum = nl.zeros((1,), dtype=a_tensor.dtype, buffer=nl.psum)
    
    # Get tensor shape
    shape = a_tensor.shape
    
    # Handle multi-dimensional tensors by treating them as flattened
    # Calculate the total number of elements
    total_elements = 1
    for dim_size in shape:
        total_elements *= dim_size
    
    # If input is empty, return 0
    if total_elements == 0:
        result = nl.ndarray((1,), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
        nl.store(result, acc_sum)
        return result[0]
    
    # Calculate the number of tiles needed
    tile_size = nl.tile_size.pmax
    trip_count = math.ceil(total_elements / tile_size)
    
    # Reshape tensor conceptually to 1D for processing
    flat_tensor = a_tensor.reshape((total_elements,))
    
    # Process the tensor in tiles
    for i in nl.affine_range(trip_count):
        # Calculate indices for current tile
        start_idx = i * tile_size
        indices = start_idx + nl.arange(tile_size)
        
        # Load data from the current tile
        # Only load indices that are within bounds
        data_tile = nl.load(flat_tensor[indices], mask=(indices < total_elements))
        
        # Sum the tile and add to accumulator
        tile_sum = nl.sum(data_tile, axis=0, mask=(indices < total_elements))
        acc_sum[0] += tile_sum
    
    # Prepare the final result
    result = nl.ndarray((1,), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    nl.store(result, acc_sum)
    
    return result[0]