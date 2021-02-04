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
    __global__ void process_array(int *a, int *result)
    {
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      result[idx] = a[idx] + 1;
    }
    """)


pycuda.driver.init()
dim = 10

a = numpy.array([range(0, dim)], dtype=numpy.int32)      # array size: 10, 32bit !!!
result = numpy.empty(dim, dtype=int)

func = mod.get_function("process_array")
func(cuda.In(a), cuda.Out(result), block=(1024, 1, 1), grid=(1, 1, 1))  # also cuda.InOut()

print("result: ", result)



