# 1) Copy data to gpu
# 2) call function on gpu using the data
# 3) use result on gpu as input to function on gpu
# 4) return result

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import timeit
import numpy


mod = SourceModule("""
    __global__ void process_array(unsigned int count, int *a, int *result)
    {
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      if(idx >= count) return;
      result[idx] = a[idx] + 1;
    }
    """)

pycuda.driver.init()
dim = 10

a = pycuda.driver.managed_zeros(shape=dim, dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
result = pycuda.driver.managed_zeros(shape=dim, dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)

func = mod.get_function("process_array")
func(numpy.uint32(dim), a, result, block=(1024, 1, 1), grid=(1, 1, 1))
pycuda.driver.Context.synchronize()

print("result: ", result)



