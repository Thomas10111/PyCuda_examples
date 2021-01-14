# What is faster, using numpy to filter/query/process an array what requires copying the whole array to the cpu
# or syncing threads and collecting all the information on the gpu
#
# The run time on the gpu more or less stays constant with an increasing number of individuals.
# The time needed on the cpu (numpy) increases steadily with more individuals, 11s vs 74s with 65535 individuals
#
# My assumption was that __syncthreads() might take more time than a numpy function.

import timeit
import random
import numpy
import hashlib
import pathlib

# PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
from pycuda.compiler import SourceModule

NumberOfIndividuals = 6

MAX_INDIVIDUALS = (1 << 10)  # 15: 32768, 16: 65536
MAX_HOUSEHOLDS = (1 << 5)
MAX_BARIS = (1 << 5)
MAX_VILLAGES = 2
IDX_START_STOP = 10

MAX_INIVIDUALS_PER_HH = 10
MAX_HH_PER_BARI = 10
BLOCK_SIZE = 1024

mod = """
// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>

extern "C" {
    #define NSTATES (1<<14)

    __global__ void update_individuals(
        unsigned int count,
        unsigned int* added_individuals,
        float dt,
        float* age,
        unsigned int* alive,
        float* death_age,
        int* dead_id,
        unsigned int* give_birth,
        int* sex)
    {
        if ( threadIdx.x == 0 )
        {
            *added_individuals = 0;
        }
        __syncthreads();

        int idx = threadIdx.x + blockIdx.x * blockDim.x;        

        if( idx > count ) return;
        
        dead_id[idx] = -1;
        if( !alive[idx] ) return;

        age[idx] += dt;

        if( age[idx] > death_age[idx] && alive[idx] ) //individual dies only once
        {
            alive[idx] = 0;
        }
        
        if( give_birth[idx] ) //add individual
        {
            const int temp = atomicAdd(added_individuals, 1); // increase index by 1
            alive[count + temp] = 1;
            age[count + temp] = 12345;
            sex[count + temp] = 0; 
        }
        
    }
}"""


def load_cuda_code_individual():
    global _update_individuals_fn

    md5 = hashlib.md5()
    md5.update(mod.encode("utf-8"))
    filename = md5.hexdigest() + ".cubin"
    path = pathlib.Path(__file__).resolve().parent / filename
    if not path.exists():
        try:
            cubin = compiler.compile(mod, no_extern_c=True)
            with open(str(path), "wb") as handle:
                handle.write(cubin)
        except cuda.CompileError as ce:
            print(f"{ce}")

    _cuda_module = cuda.module_from_file(str(path))

    _update_individuals_fn = _cuda_module.get_function("update_individuals")


load_cuda_code_individual()


class SharedObj:
    _next_id = 0

    def __init__(self):
        self._id = None

    @property
    def id(self):
        return self._id

    @classmethod
    def get_next_id(cls):
        _id = cls._next_id
        cls._next_id += 1
        return _id



class Individual(SharedObj):
    # Unified memory
    _alive = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.uint32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _active = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.uint32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _age = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _death_age = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _give_birth = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _dead_id = pycuda.driver.managed_zeros(shape=(MAX_VILLAGES, MAX_HH_PER_BARI, MAX_INIVIDUALS_PER_HH, IDX_START_STOP), dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _dead_id[:] = -1
    _sex = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _sex[:] = -1
    _added_individuals = pycuda.driver.managed_zeros(shape=1, dtype=numpy.uint32, mem_flags=cuda.mem_attach_flags.GLOBAL)

    # array containing all individuals
    all_individuals = numpy.array([None for _ in range(MAX_INDIVIDUALS)])

    def __init__(self):
        super(Individual, self).__init__()
        Individual.all_individuals[self._id] = self  # add individual to list of all individuals

    def init(self):
        self._id = Individual.get_next_id()
        self.age = random.random() * 20.0
        self.death_age = random.randint(60, 80)
        self.is_alive = True

    def set_active(self):
        self._id = Individual.get_next_id()

    @property
    def is_alive(self):
        return Individual._alive[self._id] != 0

    @is_alive.setter
    def is_alive(self, value):
        Individual._alive[self._id] = 1 if value > 0 else 0

    @property
    def age(self):
        return Individual._age[self._id]

    @age.setter
    def age(self, value):
        Individual._age[self._id] = value

    @property
    def death_age(self):
        return Individual._death_age[self._id]

    @death_age.setter
    def death_age(self, value):
        Individual._death_age[self._id] = value

    @property
    def sex(self):
        return Individual._sex[self._id]

    @classmethod
    def update_all_individuals(cls, dt):
        grid_x = (cls._next_id + BLOCK_SIZE - 1) // BLOCK_SIZE

        print("_added_individuals: ", cls._added_individuals)
        _update_individuals_fn(numpy.uint32(cls._next_id),  # unsigned int count
                               numpy.uint32(cls._added_individuals),
                               numpy.float32(dt),  # float dt
                               cls._age,  # float* age
                               cls._alive,  # unsigned int* alive
                               cls._death_age,  # float* death_age
                               cls._dead_id,
                               cls._give_birth,
                               cls._sex,
                               block=(BLOCK_SIZE, 1, 1),
                               grid=(grid_x, 1),
                               time_kernel=True
                               )
        print("_added_individuals: ", cls._added_individuals)


class Female(Individual):
    def __init__(self):
        super(Individual, self).__init__()
        self.sex = 0
        self.give_birth = False

    def birth(self):
        print("birth")

    @property
    def give_birth(self):
        return Individual._give_birth[self._id]

    @give_birth.setter
    def give_birth(self, value):
        Individual._give_birth[self._id] = value


class Male(Individual):
    def __init__(self):
        super(Individual, self).__init__()
        self.sex = 1





if __name__ == '__main__':
    individuals = []

    for ind in range(MAX_INDIVIDUALS):
        individuals.append(Individual())

    for ind_id in range(NumberOfIndividuals):
        individuals[ind_id].init()  # revive individuals
        if random.random() > 0.5:
            individuals[ind_id].__class__ = Female
            individuals[ind_id].give_birth = 1
        else:
            individuals[ind_id].__class__ = Male

    pycuda.driver.init()

    dt = 1 / 365

    number_active = len([ind for ind in individuals if ind.id is not None])
    Individual.update_all_individuals(dt)

    for i in range( number_active, number_active + Individual._added_individuals[0]):
        individuals[i].set_active()
        individuals[i].__class__ = Male if individuals[6].sex else Female
    [print(ind.id, " ", ind.age) for ind in individuals if ind.id is not None]

