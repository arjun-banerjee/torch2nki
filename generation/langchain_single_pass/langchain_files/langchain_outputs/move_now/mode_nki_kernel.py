from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_mode(a_tensor):
    # Initialize result array with int32 dtype
    result = nl.ndarray(a_tensor.shape, dtype=nl.int32, buffer=nl.shared_hbm)
    
    # Create buffer for frequencies with proper rank
    freqs = nl.zeros((a_tensor.shape[0], 1), dtype=nl.int32, buffer=nl.sbuf)
    
    # Create and store frequencies
    for i in nl.affine_range(a_tensor.shape[0]):
        curr_val = nl.load(a_tensor[i])
        # Count matches by comparing with all elements
        matches = nl.equal(a_tensor, curr_val)
        # Sum matches along axis 1 after transpose
        freq = nl.max(nl.transpose(matches), axis=1)
        nl.store(freqs[i], freq)
    
    # Find max frequency
    max_freq = nl.max(freqs, axis=0)
    
    # Find mode (value with max frequency)
    mode_mask = nl.equal(freqs, max_freq)
    mode_val = nl.load(a_tensor[mode_mask])
    
    # Store result
    nl.store(result, mode_val)
    
    return result