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
      a[idx] = a[idx] + 1;
      b[idx] = b[idx] + 10;
    }
    """)


pycuda.driver.init()

NumberOfRuns = 200
dim = 10

a = numpy.array([range(0,dim)], dtype=numpy.int32)      # 32bit !!!
b = numpy.array([range(0,dim)], dtype=numpy.int32)      # 32bit !!!

print("a: ", a)
print("b: ", b)

a_on_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)                   # allocate memory on gpu for a
b_on_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)                   # allocate memory on gpu for b
cuda.memcpy_htod(a_on_gpu, a)                                          # copy to device(gpu) from host(cpu)
cuda.memcpy_htod(b_on_gpu, b)                                          # copy to device(gpu) from host(cpu)

func = mod.get_function("process_array")    # kernel

time_kernel = func(a_on_gpu, b_on_gpu, block=(1024, 1, 1), grid=(10, 1), time_kernel=True)  # run kernel
time_kernel = func(a_on_gpu, b_on_gpu, block=(1024, 1, 1), grid=(10, 1), time_kernel=True)  # run kernel

cuda.memcpy_dtoh(a, a_on_gpu)   # copy to device(cpu) from host(gpu)
cuda.memcpy_dtoh(b, b_on_gpu)   # copy to device(cpu) from host(gpu)

print("a: ", a)
print("b: ", b)


