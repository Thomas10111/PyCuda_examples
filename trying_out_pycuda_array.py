# Sample source code from the Tutorial Introduction in the documentation.

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import timeit
import numpy


def func_numpy(a):
    temp = numpy.round(a, decimals=1)
    temp = numpy.clip(temp, 0, 80)
    return temp.astype(int)

def copy_and_run(a_gpu, a):
    # See: https://documen.tician.de/pycuda/driver.html?highlight=mem_alloc#pycuda.driver.mem_alloc
    # For a faster (but mildly less convenient) way of invoking kernels, see prepare() and prepared_call().
    cuda.memcpy_htod(a_gpu, a)
    # cuda.memcpy_htod(a_gpu, array('f', a.tolist())) # list needs to be converted to array first

    func = mod.get_function("process_array")
    time_kernel = func(a_gpu, result_gpu, block=(1024, 1, 1), grid=(10, 1), time_kernel=True)

    cuda.memcpy_dtoh(result, result_gpu)

    return result, time_kernel



mod = SourceModule("""
    __global__ void process_array(float *a, int *result)
    {
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      if (a[idx] > 80)  a[idx] = 80;
      result[idx] = int(a[idx] + 0.05);
    }
    """)


pycuda.driver.init()
# attributes = pycuda.driver.Device(0).get_attributes()
# [print(a, ":", attributes[a]) for a in attributes]

NumberOfRuns = 200
random_dim0 = 1024 * 10

a = numpy.random.rand(random_dim0) * 100
print("a: ", a)
result = numpy.empty(random_dim0, dtype=int)

a = a.astype(dtype=numpy.float32)

a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
result_gpu = cuda.mem_alloc(result.size * result.dtype.itemsize)



# # See: https://documen.tician.de/pycuda/driver.html?highlight=mem_alloc#pycuda.driver.mem_alloc
# # For a faster (but mildly less convenient) way of invoking kernels, see prepare() and prepared_call().
# cuda.memcpy_htod(a_gpu, a)
# # cuda.memcpy_htod(a_gpu, array('f', a.tolist())) # list needs to be converted to array first
#
# func = mod.get_function("process_array")
# time_kernel = func(a_gpu, result_gpu, block=(1024, 1, 1), grid=(10,1), time_kernel=True)

#[result, time_kernel]= copy_and_run(a_gpu, a)
#print("time_kernel: ", time_kernel)

# grid = (1, 1)
# block = (1024, 1, 1)
# func.prepare((numpy.float, numpy.int))
# func.prepared_call(grid, block, a_gpu, result_gpu)


#result_time = timeit.timeit("cuda.memcpy_dtoh(result, result_gpu)", globals=globals(), number=NumberOfRuns)

result_time = timeit.timeit("copy_and_run(a_gpu, a)", globals=globals(), number=NumberOfRuns)
print(result_time)
result_time = timeit.timeit("func_numpy(a)", globals=globals(), number=NumberOfRuns)
print(result_time)

print("result_gpu:   ", result)
result_numpy = func_numpy(a)
print("result_numpy: ", result_numpy)

for i, (r1, r2) in enumerate(zip(result, result_numpy)):
    if r1 != r2:
        print("Error: ", r1, " != ", r2, "  a: ", a[i])


