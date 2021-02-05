# 1) Copy data to gpu
# 2) call function on gpu using the data
# 3) use result on gpu as input to function on gpu
# 4) return result

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
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

a_on_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)                   # allocate memory on gpu
result_on_gpu = cuda.mem_alloc(result.size * result.dtype.itemsize)    # allocate memory on gpu for result
cuda.memcpy_htod(a_on_gpu, a)                                          # copy to device(gpu) from host(cpu)

func = mod.get_function("process_array")
func(a_on_gpu, result_on_gpu, block=(dim, 1, 1), grid=(1, 1, 1), time_kernel=True)
cuda.memcpy_dtoh(result, result_on_gpu) # copy to device(cpu) from host(gpu)

print("result: ", result)



