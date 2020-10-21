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
    __global__ void process_array(int *a, int *b, int *result)
    {
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      result[idx] = a[idx] + b[idx];
    }
    """)


pycuda.driver.init()

NumberOfRuns = 200
dim = 10

a = numpy.array([range(0,dim)], dtype=numpy.int32)      # 32bit !!!
b = numpy.array([range(dim,2*dim)], dtype=numpy.int32)      # 32bit !!!
result = numpy.zeros(dim, dtype=numpy.int32)      # 32bit !!!

print("a: ", a)
print("b: ", b)

a_on_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)                   # allocate memory on gpu for a
b_on_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)                   # allocate memory on gpu for b
result_on_gpu = cuda.mem_alloc(result.size * result.dtype.itemsize)                   # allocate memory on gpu for result
cuda.memcpy_htod(a_on_gpu, a)                                          # copy to device(gpu) from host(cpu)
cuda.memcpy_htod(b_on_gpu, b)                                          # copy to device(gpu) from host(cpu)

func = mod.get_function("process_array")    # kernel

time_kernel = func(a_on_gpu, b_on_gpu, result_on_gpu, block=(1024, 1, 1), grid=(10, 1), time_kernel=True)  # run kernel
time_kernel = func(result_on_gpu, b_on_gpu, result_on_gpu, block=(1024, 1, 1), grid=(10, 1), time_kernel=True)

cuda.memcpy_dtoh(result, result_on_gpu)   # copy to device(cpu) from host(gpu)

print("result: ", result)



