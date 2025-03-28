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
        # Generate tensor indices for the input/output tensors
        i_p = p * tile_size + nl.arange(tile_size)[:, None]
        
        # Load input vectors from external memory to on-chip memory
        vec1_tile = nl.load(vec1[i_p], mask=(i_p < vec1.shape[0]))
        vec2_tile = nl.load(vec2[i_p], mask=(i_p < vec2.shape[0]))
        
        # Perform vector addition
        result_tile = nl.add(vec1_tile, vec2_tile)
        
        # Store results back to external memory
        nl.store(result_tensor[i_p], result_tile, mask=(i_p < result_tensor.shape[0]))
    
    return result_tensor