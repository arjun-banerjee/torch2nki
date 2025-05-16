from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_all(a_tensor):
    # Initialize result as a single boolean value
    result = nl.ndarray((), dtype=nl.bool_, buffer=nl.shared_hbm)
    
    # Initialize accumulator to 1 (True) using int32 for PSUM
    all_true = nl.ones((), dtype=nl.int32, buffer=nl.psum)
    
    # Get tensor shape and flatten for processing
    shape = a_tensor.shape
    total_size = 1
    for dim in shape:
        total_size *= dim
    
    # Calculate the number of tiles needed
    tile_size = nl.tile_size.pmax
    trip_count = math.ceil(total_size / tile_size)
    
    # Process the tensor in tiles
    for i in nl.affine_range(trip_count):
        # Calculate start index for this tile
        start_idx = i * tile_size
        
        # Create indices for this tile, ensuring we don't exceed total_size
        idx = start_idx + nl.arange(min(tile_size, total_size - start_idx))
        
        # Convert flattened indices to tensor indices
        # For simplicity, we'll reshape the tensor to a 1D tensor for processing
        flat_tensor = a_tensor.reshape((-1,))
        
        # Load the current tile
        tile_data = nl.load(flat_tensor[idx], mask=(idx < total_size))
        
        # Convert values to int32 (0 for False, 1 for True)
        tile_int = nl.cast(tile_data, dtype=nl.int32)
        
        # Multiply accumulator by each element (any 0 will make all_true become 0)
        for j in nl.affine_range(min(tile_size, total_size - start_idx)):
            elem_idx = nl.full((1,), j, dtype=nl.int32)
            elem = nl.load(tile_int[elem_idx])
            all_true = nl.multiply(all_true, elem)
            
            # Early termination if we find a False value
            # We can't use conditionals directly, but the multiplication
            # effectively achieves this by making all_true 0 once any element is 0
    
    # Convert final result to boolean and store in result
    final_result = nl.cast(all_true, dtype=nl.bool_)
    nl.store(result, final_result)
    
    return result