# Sample source code from the Tutorial Introduction in the documentation.

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import timeit
import numpy
from array import array

NumberOfIndividuals = 3

class Individual:
    def __init__(self):
        self.age = 0
        self._alive = True

    def update(self):
        self.age += 1
        self._alive = self.age < 10
        if self.dead():
            print("Died")

    def dead(self):
        return not self._alive


class Individuals_cuda:
    def __init__(self):
        self.age = numpy.zeros(NumberOfIndividuals, dtype=int)
        self._alive = numpy.zeros(NumberOfIndividuals, dtype=bool)

    def process(self):
        for i in range(len(self.age)):
            if self.dead(i):
                print("Died")

    def dead(self, i):
        return not self._alive[i]

    def __getitem__(self, i):
        individual = Individual()
        individual.age = self.age[i]
        individual._alive = self._alive[i]
        return individual


mod = SourceModule("""
    __global__ void update(int *age, bool *_alive)
    {
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      age[idx]++;
      _alive[idx] = age[idx] < 10;
    }
    """)

if __name__ == '__main__':
    duration = 20
    individuals = [Individual() for i in range(0,NumberOfIndividuals)]
    # for year in range(0, duration):
    #     print("Year: ", year)
    #     for ind in individuals:
    #         ind.update()

    individuals_cuda = Individuals_cuda()
    for i, ind in enumerate(individuals):
        individuals_cuda.age[i] = ind.age
        individuals_cuda._alive[i] = ind._alive



    pycuda.driver.init()
    #

    # result = numpy.empty(random_dim0, dtype=int)
    #
    # a = a.astype(dtype=numpy.float32)
    #
    age_gpu = cuda.mem_alloc(individuals_cuda.age.size * individuals_cuda.age.dtype.itemsize)
    _alive_gpu = cuda.mem_alloc(individuals_cuda._alive.size * individuals_cuda._alive.dtype.itemsize)

    for year in range(duration):
        # # See: https://documen.tician.de/pycuda/driver.html?highlight=mem_alloc#pycuda.driver.mem_alloc
        # # For a faster (but mildly less convenient) way of invoking kernels, see prepare() and prepared_call().
        # cuda.memcpy_htod(a_gpu, array('f', a.tolist()))
        cuda.memcpy_htod(age_gpu, individuals_cuda.age)
        cuda.memcpy_htod(_alive_gpu, individuals_cuda._alive)
        func = mod.get_function("update")
        time_kernel = func(age_gpu, _alive_gpu, block=(1024, 1, 1), grid=(10, 1), time_kernel=True)
        cuda.memcpy_dtoh(individuals_cuda.age, age_gpu)
        cuda.memcpy_dtoh(individuals_cuda._alive, _alive_gpu)

        individuals_cuda.process()
        print("time_kernel: ", time_kernel)

    print("ages: ", individuals_cuda.age)
    print("individual 2:", individuals_cuda[2])



    #
    # # grid = (1, 1)
    # # block = (1024, 1, 1)
    # # func.prepare((numpy.float, numpy.int))
    # # func.prepared_call(grid, block, a_gpu, result_gpu)
    #
    #


    # print(result_time)
    # result_time = timeit.timeit("func_numpy(a)", globals=globals(), number=NumberOfRuns)
    # print(result_time)
    #
    # print("result_gpu:   ", result)
    # result_numpy = func_numpy(a)
    # print("result_numpy: ", result_numpy)
    #
    # for i, (r1, r2) in enumerate(zip(result, result_numpy)):
    #     if r1 != r2:
    #         print("Error: ", r1, " != ", r2, "  a: ", a[i])





# mod_class = SourceModule("""
#     class A
#     {
#         public:
#         A(){};
#         int data;
#     };
#
#     __global__ void process_array(A *a)
#     {
#         a->data = 5;
#     }
#     """)