# Sample source code from the Tutorial Introduction in the documentation.

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import timeit
import numpy


pycuda.driver.init()
# print(cuda.Device.count())
# print(pycuda.tools.make_default_context())
# print(pycuda.driver.Device(pycuda.tools.make_default_context()).count())
attributes = pycuda.driver.Device(0).get_attributes()
[print(a, ":", attributes[a]) for a in attributes]

NumberOfRuns = 100
random_dim0 = 4000
random_dim1 = 4000

def func_numpy(a):
    temp = numpy.multiply(a, 2)
    return numpy.add(temp, 1)


a = numpy.random.rand(random_dim0,random_dim1)
a = a.astype(dtype=numpy.float32)

#a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= 2;
      a[idx] += 1;
    }
    """)

func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

a_doubled = numpy.empty_like(a)

result_time = timeit.timeit("cuda.memcpy_dtoh(a_doubled, a_gpu)", globals=globals(), number=NumberOfRuns)
#cuda.memcpy_dtoh(a_doubled, a_gpu)

print(result_time)
# print("original array:")
# print(a)
# print("doubled with kernel:")
# print(a_doubled)

# alternate kernel invocation -------------------------------------------------
#
# #func(cuda.InOut(a), block=(4, 4, 1))
# result_time = timeit.timeit("func(cuda.InOut(a), block=(4, 4, 1))", globals=globals(), number=NumberOfRuns)
# print(result_time)
#
# print("doubled with InOut:")
# print(a)

# part 2 ----------------------------------------------------------------------

# import pycuda.gpuarray as gpuarray
# a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
# a_doubled = (2*a_gpu).get()
#
# print("original array:")
# print(a_gpu)
# print("doubled with gpuarray:")
# print(a_doubled)



result_time = timeit.timeit("func_numpy(a)", globals=globals(), number=NumberOfRuns)
print(result_time)
