from neuronxcc import nki
import neuronxcc.nki.language as nl
import itertools 

@nki.jit
def nki_einsum_kernel_naive(einsum_inputs, contract_str):

    """NKI kernel to compute einsum operation
    """
    i_o_split = contract_str.split("->")
    input_idxs = i_o_split[0].split(",")
    output = i_o_split[1]

    output_chars = list(output)
    input_chars = set("".join(input_idxs))
    assert set(output_chars).issubset(input_chars)

    output_shape = [] 

    sum_idxs = input_chars.difference(output_chars)
    sum_idxs = list(sum_idxs)
    sum_dim_sizes = {}


    for i, idx in enumerate(input_idxs):
        idx = idx.strip()
        
        assert len(einsum_inputs[i].shape) == len(idx)
        for j, var in enumerate(idx):
            if var in output_chars:
                output_shape.append(einsum_inputs[i].shape[j])
            elif var in sum_idxs:
                sum_dim_sizes[var] = einsum_inputs[i].shape[j]
            else: 
                assert False 
    

    sum_dim_sizes = list(sum_dim_sizes.values())
    output = nl.zeros(output_shape, dtype=nl.float32, buffer=nl.shared_hbm)
    output_array = nl.zeros(output_shape, dtype=nl.float32)
    # cache maps for output/sum chars 
    output_char_map = {}
    sum_char_map = {}
    print(output_chars, sum_idxs)
    input()

    for i, char in enumerate(output_chars):
        output_char_map[char] = i

    for i, char in enumerate(sum_idxs):
        sum_char_map[char] = i


    for out_coords in itertools.product(*[range(d) for d in output_shape]):
        print(out_coords)
        #print(out_coords)
        #input()
        #out_coords = [p, q, r, s]
        
        # We will accumulate the sum for output_array[out_coords]
        accum = 0.0
        
        # Inner loop: sum over all possible values of the summation indices
        for sum_coords in itertools.product(*[range(d) for d in sum_dim_sizes]):
            print(sum_coords)
            #input()
            
            # We'll compute the product of all input arrays' relevant entries
            # then add to accum.
            prod_val = 1.0
            
            
            
            for inp, idx_str in zip(einsum_inputs, input_idxs):
                #print(idx_str)
                #input()
                coords_for_this_input = []
                
                for char in idx_str:
                    if char in output_chars:
                        # find which position in output_list
                        pos = output_char_map[char]
                        coords_for_this_input.append(out_coords[pos])
                    else:
                        # must be in sum_list
                        pos = sum_char_map[char]
                        coords_for_this_input.append(sum_coords[pos])
                
                # Now coords_for_this_input is something like [pVal, iVal] if idx_str == "pi"
                # Multiply the corresponding array value
                #print(coords_for_this_input)
                #print(inp.shape, coords_for_this_input)
                print(coords_for_this_input)
                prod_val *= inp[tuple(coords_for_this_input)]
                #print(prod_val)
            print()
            # Add the product to the accumulator
            accum += prod_val
        print("accum done")
        # Finally assign accum to the output array at out_coords
        output_array[out_coords] = accum.item()
    
    nl.store(output, output_array)
    return output 


    

@nki.jit 
def nki_einsum_kernel_optimized(einsum_inputs, contract_str): 
    pass 


