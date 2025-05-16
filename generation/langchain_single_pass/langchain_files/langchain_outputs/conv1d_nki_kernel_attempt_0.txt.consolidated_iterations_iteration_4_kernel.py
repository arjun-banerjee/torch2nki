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
    
    # Calculate trips needed to process output channels in chunks of pmax
    oc_trips = math.ceil(out_channels / nl.tile_size.pmax)
    
    # Process each batch
    for b in range(batch_size):
        # Process output channels in chunks
        for oc_idx in nl.affine_range(oc_trips):
            # Calculate the range of output channels for this chunk
            oc_start = oc_idx * nl.tile_size.pmax
            oc_end = min(out_channels, (oc_idx + 1) * nl.tile_size.pmax)
            oc_size = oc_end - oc_start
            
            # Initialize accumulator for convolution results
            output_acc = nl.zeros((oc_size, output_length), dtype=x.dtype, buffer=nl.psum)
            
            # Process input channel groups
            for g in range(groups):
                # Calculate channel ranges for this group
                ic_start_group = g * in_channels_per_group
                ic_end_group = (g + 1) * in_channels_per_group
                oc_start_group = g * (out_channels // groups) + oc_start
                oc_end_group = min(g * (out_channels // groups) + (out_channels // groups), oc_start + oc_size)
                
                if oc_start_group >= oc_end_group:
                    continue
                    
                # Process each input channel in the group
                for ic in range(ic_start_group, ic_end_group):
                    # Load input channel data
                    if padding > 0:
                        # Create padded input
                        padded_length = input_length + 2 * padding
                        input_data = nl.zeros((padded_length,), dtype=x.dtype, buffer=nl.sbuf)
                        
                        # Load actual data into the padded buffer
                        i_in = nl.arange(input_length)
                        input_actual = nl.load(x[b, ic, i_in])
                        
                        # Copy actual data to the middle of padded buffer
                        i_pad = nl.arange(input_length) + padding
                        input_data[i_pad] = input_actual
                    else:
                        # Load without padding
                        i_in = nl.arange(input_length)
                        input_data = nl.load(x[b, ic, i_in])
                    
                    # Calculate which output channels in the current chunk belong to this group
                    rel_oc_start = max(0, oc_start_group - oc_start)
                    rel_oc_end = min(oc_size, oc_end_group - oc_start)
                    
                    # Process each output position
                    for out_pos in range(output_length):
                        # Calculate the start position in the input
                        in_pos = out_pos * stride - padding
                        
                        # Process each filter position
                        for k_pos in range(kernel_size):
                            # Calculate input position with dilation
                            actual_in_pos = in_pos + k_pos * dilation
                            
                            # Skip if outside the input boundaries
                            if padding > 0:
                                if actual_in_pos < 0 or actual_in_pos >= padded_length:
                                    continue
                            else:
                                if actual_in_pos < 0 or actual_in_pos >= input_length:
                                    continue
                            
                            # Load filter value
                            rel_oc_indices = nl.arange(rel_oc_end - rel_oc_start)
                            oc_indices = rel_oc_indices + rel_oc_start
                            filter_values = nl.load(weight[oc_indices + oc_start_group - oc_start, ic - ic_start_group, k_pos])
                            
                            # Get input value at this position
                            if padding > 0:
                                input_val = input_data[actual_in_pos]
                            else:
                                input_val = input_data[actual_in_pos]
                            
                            # Multiply and accumulate
                            output_acc[oc_indices, out_pos] += nl.multiply(input_val, filter_values)
            
            # Add bias if provided
            if bias is not None:
                oc_indices = nl.arange(oc_size)
                bias_values = nl.load(bias[oc_indices + oc_start])
                for out_pos in range(output_length):
                    output_acc[:, out_pos] = nl.add(output_acc[:, out_pos], bias_values)
            
            # Store results
            out_pos_indices = nl.arange(output_length)
            oc_indices = nl.arange(oc_size)
            result_slice = output_acc[oc_indices, out_pos_indices]
            result_oc_indices = oc_indices + oc_start
            nl.store(result[b, result_oc_indices, out_pos_indices], value=result_slice)
    
    return result