from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_vector_add(vec1, vec2, result_tensor):
    # Ensure vectors have same length
    if vec1.shape[0] != vec2.shape[0]:
        raise ValueError("Vectors must be of the same length")
        
    # Calculate number of tiles needed
    tile_size = nl.tile_size.pmax
    trip_count = math.ceil(vec1.shape[0]/tile_size)
    
    # Process vectors in tiles
    for p in nl.affine_range(trip_count):
        # Generate indices for current tile
        i_p = p * tile_size + nl.arange(tile_size)[:, None]
        
        # Load input vectors with masking for last tile
        vec1_tile = nl.load(vec1[i_p], mask=(i_p < vec1.shape[0]))
        vec2_tile = nl.load(vec2[i_p], mask=(i_p < vec2.shape[0]))
        
        # Perform addition
        result_tile = nl.add(vec1_tile, vec2_tile)
        
        # Store result with masking for last tile
        nl.store(result_tensor[i_p], result_tile, mask=(i_p < vec1.shape[0]))
        
    return result_tensor