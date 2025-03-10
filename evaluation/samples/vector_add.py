import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def vector_add_kernel(v1, v2):
    # Assume v1 and v2 are 1D tensors of the same size
    size = v1.shape[0]

    print(size)
    # Create an output tensor of the same size
    result = nl.zeros((size, 1), dtype=v1.dtype, buffer=nl.hbm)
    a = nl.load(v1) 
    b = nl.load(v2)
    c = nl.add(a, b)
    nl.store(result[:, None], c)


    return result
