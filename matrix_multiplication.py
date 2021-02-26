import timeit
import random
import numpy
from numba import float32, vectorize, uint32, void, guvectorize, cuda

# PyCUDA
#import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
from pycuda.compiler import SourceModule
import cupy as cp
import timeit
from scipy import sparse




# x = pycuda.driver.managed_zeros(shape=4, dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
# y = pycuda.driver.managed_zeros(shape=4, dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
# x[0:4] = [1, 2, 3, 4]
# y[0:4] = [1, 2, 3, 4]
# l2_gpu = cupy.linalg.norm(y)
# a = cupy.multiply(x,y)
# print(l2_gpu)

# a = cupy.random.rand(44, 20)
# b = cupy.random.rand(20, 1)

pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

sizes = [100, 1000, 2000]
for size in sizes:
    print("---- Size: ",size, " x ", size, " ----")
    a = cp.ones((size, size), dtype=numpy.float32)
    b = cp.ones((size, size), dtype=numpy.float32)

    a[0:size] = 2

    start_time = timeit.default_timer()
    cp.dot(a, b)    # cupy.matmul(a, b, out=None)
    cp.cuda.Device(0).synchronize()
    print("duration gpu: ", timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    for i in range(10):
        a[0:size] = i
        c = cp.dot(a, b)
        cp.cuda.Device(0).synchronize()
    print("avg duration gpu: ", (timeit.default_timer() - start_time)/10.0)


    # numpy
    a_np = numpy.random.random((size, size))
    b_np = numpy.random.random((size, size))
    result_time = timeit.timeit("c_np = numpy.dot(a_np, b_np)", globals=globals(), number=10)
    print("duration numpy: ", result_time / 10.0)


    # # numpy int
    a_np = numpy.random.randint(0, 10, (size, size))
    b_np = numpy.random.randint(0, 10, (size, size))
    result_time = timeit.timeit("c_np = numpy.dot(a_np, b_np)", globals=globals(), number=1)
    print("duration numpy int: ", result_time / 1.0)

    # # numpy int32
    a_np = numpy.random.randint(0, 10, (size, size), dtype=numpy.int32)
    b_np = numpy.random.randint(0, 10, (size, size), dtype=numpy.int32)
    result_time = timeit.timeit("c_np = numpy.dot(a_np, b_np)", globals=globals(), number=1)
    print("duration numpy int32: ", result_time / 1.0)

    # sparse
    M = sparse.random(size, size, .30)
    result_time = timeit.timeit("c_sparse = M.dot(M)", globals=globals(), number=10)
    print("avg duration sparse: ", result_time/10.0)

    print()


# "C:\Program Files\Python36\python.exe" C:/Users/tfischle/Github/PyCuda_examples/matrix_multiplication.py
# ---- Size:  100  x  100  ----
# duration gpu:  0.5852928000000001
# avg duration gpu:  0.00014606999999999815
# duration numpy:  0.0003451800000000005
# duration numpy int:  0.0009719999999999729
# duration numpy int32:  0.0009557000000000038
# avg duration sparse:  0.0009325600000000045
#
# ---- Size:  1000  x  1000  ----
# duration gpu:  0.0015964000000000533
# avg duration gpu:  0.0031100099999999964
# duration numpy:  0.013901710000000001
# duration numpy int:  0.6493503
# duration numpy int32:  0.633626
# avg duration sparse:  0.2823481
#
# ---- Size:  2000  x  2000  ----
# duration gpu:  0.006359999999999921
# avg duration gpu:  0.00414393999999998
# duration numpy:  0.07829159999999993
# duration numpy int:  27.2362269
# duration numpy int32:  27.470195999999994
# avg duration sparse:  2.313961