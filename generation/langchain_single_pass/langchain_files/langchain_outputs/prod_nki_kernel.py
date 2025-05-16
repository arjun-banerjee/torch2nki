from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_prod(a_tensor):
    # For scalar tensors, just return the input
    if len(a_tensor.shape) == 0:
        return a_tensor
        
    # Get the total number of elements
    total_elements = 1
    for dim in a_tensor.shape:
        total_elements *= dim
    
    # If empty tensor, return 1 as per numpy's behavior
    if total_elements == 0:
        result = nl.ones((1,), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
        return result[0]
    
    # Initialize result as 1 with the same dtype as input
    result_val = nl.ones((1,), dtype=a_tensor.dtype, buffer=nl.psum)
    
    # Maximum batch size to process at once
    max_batch = nl.tile_size.pmax
    
    # Calculate number of batches needed
    num_batches = math.ceil(total_elements / max_batch)
    
    # Process the tensor in batches
    for batch_idx in nl.affine_range(num_batches):
        # Calculate start and end indices for this batch
        start_idx = batch_idx * max_batch
        
        # Create indices for this batch
        indices = start_idx + nl.arange(max_batch)
        
        # Load a batch of elements from the flattened tensor
        # Use mask to handle the last batch which may be smaller
        batch_values = nl.load(a_tensor.reshape((-1,))[indices], mask=(indices < total_elements))
        
        # Compute product for this batch
        batch_prod = nl.ones((1,), dtype=a_tensor.dtype, buffer=nl.psum)
        for i in nl.affine_range(max_batch):
            # Only multiply if the index is valid (for the last batch)
            if start_idx + i < total_elements:
                batch_prod = nl.multiply(batch_prod, batch_values[i])
        
        # Update the overall product
        result_val = nl.multiply(result_val, batch_prod)
    
    # Create the final result array
    result = nl.ndarray((1,), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    nl.store(result, result_val)
    
    return result[0]