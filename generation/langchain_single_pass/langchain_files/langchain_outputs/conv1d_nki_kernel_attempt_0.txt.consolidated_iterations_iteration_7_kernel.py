from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Extract dimensions
    batch_size, in_channels, input_length = x.shape
    out_channels, in_channels_per_group, kernel_size = weight.shape
    
    # Calculate output length
    output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    
    # Initialize result array
    result = nl.ndarray((batch_size, out_channels, output_length), dtype=x.dtype, buffer=nl.shared_hbm)
    
    # Process in tiles to handle hardware limitations
    max_tile_size = nl.tile_size.pmax  # Maximum partition size (128)
    
    # Process batch dimension in tiles
    for b_idx in nl.affine_range(math.ceil(batch_size / max_tile_size)):
        b_start = b_idx * max_tile_size
        b_end = min(batch_size, (b_idx + 1) * max_tile_size)
        b_size = b_end - b_start
        
        # Process output channel dimension in tiles
        for oc_idx in nl.affine_range(math.ceil(out_channels / max_tile_size)):
            oc_start = oc_idx * max_tile_size
            oc_end = min(out_channels, (oc_idx + 1) * max_tile_size)
            oc_size = oc_end - oc_start
            
            # Process output length dimension in tiles
            for ol_idx in nl.affine_range(math.ceil(output_length / max_tile_size)):
                ol_start = ol_idx * max_tile_size
                ol_end = min(output_length, (ol_idx + 1) * max_tile_size)
                ol_size = ol_end - ol_start
                
                # Create a tile to accumulate results for this output segment
                out_tile = nl.zeros((b_size, oc_size, ol_size), dtype=x.dtype, buffer=nl.sbuf)
                
                # Process groups
                for g in nl.affine_range(groups):
                    # Calculate channel ranges for this group
                    ic_start_group = g * in_channels_per_group
                    ic_end_group = (g + 1) * in_channels_per_group
                    oc_start_group = oc_start + g * (out_channels // groups)
                    oc_end_group = min(oc_end, oc_start_group + (out_channels // groups))
                    oc_size_group = oc_end_group - oc_start_group
                    
                    if oc_size_group <= 0:
                        continue
                    
                    # Process input channels in tiles if needed
                    for ic_idx in nl.affine_range(math.ceil(in_channels_per_group / max_tile_size)):
                        ic_start_tile = ic_start_group + ic_idx * max_tile_size
                        ic_end_tile = min(ic_end_group, ic_start_tile + max_tile_size)
                        ic_size_tile = ic_end_tile - ic_start_tile
                        
                        # For each position in the output
                        for k in nl.affine_range(kernel_size):
                            # Calculate corresponding input position with dilation
                            in_pos = ol_start * stride - padding + k * dilation
                            
                            # Skip if outside input bounds
                            if in_pos < 0 or in_pos >= input_length:
                                continue
                            
                            # Load input tile
                            i_p_input = nl.arange(b_size)
                            input_tile = nl.load(
                                x[b_start:b_end, ic_start_tile:ic_end_tile, in_pos:in_pos+1], 
                                mask=(i_p_input < b_size)
                            )
                            
                            # Load weight tile
                            i_p_weight = nl.arange(oc_size_group)
                            weight_tile = nl.load(
                                weight[oc_start_group:oc_end_group, 0:ic_size_tile, k:k+1],
                                mask=(i_p_weight < oc_size_group)
                            )
                            
                            # Compute contribution to output
                            for b in nl.affine_range(b_size):
                                for oc in nl.affine_range(oc_size_group):
                                    for ic in nl.affine_range(ic_size_tile):
                                        for ol in nl.affine_range(ol_size):
                                            # Skip invalid indices
                                            if ol_start + ol >= output_length:
                                                continue
                                                
                                            # Apply convolution
                                            input_val = input_tile[b, ic, 0]
                                            weight_val = weight_tile[oc, ic, 0]
                                            out_tile[b, oc - (oc_start_group - oc_start), ol] += nl.multiply(input_val, weight_val)
                
                # Apply bias if provided
                if bias is not None:
                    for b in nl.affine_range(b_size):
                        for oc in nl.affine_range(oc_size):
                            bias_val = nl.load(bias[oc_start + oc])
                            for ol in nl.affine_range(ol_size):
                                out_tile[b, oc, ol] = nl.add(out_tile[b, oc, ol], bias_val)
                
                # Store result
                i_p_result = nl.arange(b_size)
                nl.store(
                    result[b_start:b_end, oc_start:oc_end, ol_start:ol_end],
                    value=out_tile,
                    mask=(i_p_result < b_size)
                )
    
    return result