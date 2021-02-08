# Unified memory, data is shared by copying pages

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import timeit
import numpy


mod = SourceModule("""
    __global__ void process_array(int *a, int *result)
    {
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      result[idx] = a[idx] + 1;
    }
    """)

pycuda.driver.init()
dim = 10

a = pycuda.driver.managed_zeros(shape=dim, dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
result = pycuda.driver.managed_zeros(shape=dim, dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)

func = mod.get_function("process_array")
func(a, result, block=(dim, 1, 1), grid=(1, 1, 1))
pycuda.driver.Context.synchronize()

print("result: ", result)



