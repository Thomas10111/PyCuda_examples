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

NumberOfIndividuals = 10
NumberOfHouseholds = 2
NumberOfBaris = 3
NumberOfIndividualsPerHH = 3
NumberOfHHPerBari = 3

MAX_INDIVIDUALS = (1 << 16) # 15: 32768, 16: 65536
MAX_HOUSEHOLDS = (1 << 5)
MAX_BARIS = (1 << 5)

MAX_INIVIDUALS_PER_HH = 10
MAX_HH_PER_BARI = 10
BLOCK_SIZE = 1024

mod = """
// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>

extern "C" {
    #define NSTATES (1<<14)
    __device__ unsigned int index;
    __device__ unsigned int MAX_HOUSEHOLDS = (1 << 4);
    __device__ unsigned int MAX_INIVIDUALS_PER_HH = 10;
    __device__ unsigned int MAX_HH_PER_BARI = 10;
    __device__ float log_device[NSTATES][10];

     __global__ void update_baris(
        unsigned int count,
        unsigned int* households)
    {
        
        int bari_id = threadIdx.x + blockIdx.x * blockDim.x;
        if( bari_id > count ) return;
        
        if(bari_id == 1)
        {
            // move hh3 from bari
            unsigned int temp = households[bari_id * MAX_HH_PER_BARI + 2];
            households[bari_id * MAX_HH_PER_BARI + 2] = households[(bari_id+1) * MAX_HH_PER_BARI + 2];
            households[(bari_id+1) * MAX_HH_PER_BARI + 2] = temp;
        }       
    }


    
     __global__ void update_households(
        unsigned int count,
        unsigned int* individuals)
    {
        
        int household_id = threadIdx.x + blockIdx.x * blockDim.x;
        if( household_id > count ) return;
       
        if(household_id == 0)
        {
            // exchange individual number 2 between hh0 and hh1
            unsigned int temp = individuals[household_id * MAX_INIVIDUALS_PER_HH + 2];
            individuals[household_id * MAX_INIVIDUALS_PER_HH + 2] = individuals[(household_id+1) * MAX_INIVIDUALS_PER_HH + 2];
            individuals[(household_id+1) * MAX_INIVIDUALS_PER_HH + 2] = temp;
        }        
    }
    
    __global__ void update_individuals(
        unsigned int count,
        float dt,
        float* age,
        unsigned int* alive,
        float* death_age,
        int* dead_id)
    {
        if ( threadIdx.x == 0 )
        {
            index = 0;
        }
        __syncthreads();
        
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        if( idx > count ) return;
        dead_id[idx] = -1;
        if( !alive[idx] ) return;
              
        age[idx] += dt;
        
        if( age[idx] > death_age[idx] )
        {
            alive[idx] = 0;
            const int resultIndex = atomicAdd(&index, 1); // increase index by 1
            dead_id[resultIndex] = idx;
        }
    }
}"""


def load_cuda_code_individual():
    global _update_individuals_fn
    global _update_households_fn
    global _update_baris_fn

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
    _update_households_fn = _cuda_module.get_function("update_households")
    _update_baris_fn = _cuda_module.get_function("update_baris")


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


class Bari(SharedObj):
    _households = pycuda.driver.managed_zeros(shape=MAX_HOUSEHOLDS, dtype=numpy.uint32, mem_flags=cuda.mem_attach_flags.GLOBAL)

    # array containing all baris
    all_baris = numpy.array([None for _ in range(MAX_BARIS)])

    def __init__(self):
        self._id = Bari.get_next_id()
        self.next_available_idx = 0
        Bari.all_baris[self._id] = self

    @classmethod
    def update_all_baris(cls):
        grid_x = (cls._next_id + BLOCK_SIZE - 1) // BLOCK_SIZE

        _update_baris_fn(numpy.uint32(cls._next_id),  # unsigned int count
                         cls._households,
                         block=(BLOCK_SIZE, 1, 1),
                         grid=(grid_x, 1),
                         time_kernel=True
                         )

    def add(self, household):
        idx = self._id * MAX_HH_PER_BARI + self.next_available_idx
        self.next_available_idx += 1
        Bari._households[idx] = household.id

    @property
    def households(self):
        start_idx = self._id * MAX_HH_PER_BARI
        end_idx = self._id * MAX_HH_PER_BARI + self.next_available_idx
        return Bari._households[start_idx:end_idx]


class Household(SharedObj):
    _individuals = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.uint32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    # _hh_ids = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.uint32, mem_flags=cuda.mem_attach_flags.GLOBAL)

    # array containing all households
    all_households = numpy.array([None for _ in range(MAX_HOUSEHOLDS)])

    def __init__(self):
        self._id = Household.get_next_id()
        self.next_available_idx = 0
        Household.all_households[self._id] = self

    @classmethod
    def update_all_households(cls):
        grid_x = (cls._next_id + BLOCK_SIZE - 1) // BLOCK_SIZE

        _update_households_fn(numpy.uint32(cls._next_id),  # unsigned int count
                              cls._individuals,
                              block=(BLOCK_SIZE, 1, 1),
                              grid=(grid_x, 1),
                              time_kernel=True
                              )

    def add(self, individual):
        idx = self._id * MAX_INIVIDUALS_PER_HH + self.next_available_idx
        self.next_available_idx += 1
        Household._individuals[idx] = individual.id

    @property
    def individuals(self):
        start_idx = self._id * MAX_INIVIDUALS_PER_HH
        end_idx = self._id * MAX_INIVIDUALS_PER_HH + self.next_available_idx
        return Household._individuals[start_idx:end_idx]


class Individual(SharedObj):

    # Unified memory
    _alive = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.uint32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _age = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _death_age = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _dead_id = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _dead_id[:] = -1

    # array containing all individuals
    all_individuals = numpy.array([None for _ in range(MAX_INDIVIDUALS)])

    def __init__(self):
        self._id = Individual.get_next_id()
        #self.age = random.randint(10, 20)
        self.age = random.random() * 20.0
        self.death_age = random.randint(60, 80)
        self.is_alive = True
        Individual.all_individuals[self._id] = self # add individual to list of all individuals

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

    @classmethod
    def update_all_individuals(cls, dt):
        grid_x = (cls._next_id + BLOCK_SIZE - 1) // BLOCK_SIZE

        _update_individuals_fn(numpy.uint32(cls._next_id),  # unsigned int count
                               numpy.float32(dt),  # float dt
                               cls._age,  # float* age
                               cls._alive,  # unsigned int* alive
                               cls._death_age,  # float* death_age
                               cls._dead_id,
                               block=(BLOCK_SIZE, 1, 1),
                               grid=(grid_x, 1),
                               time_kernel=True
                               )

if __name__ == '__main__':
    NumberOfRuns = 1
    duration = 365 * 60
    baris = []


    for i in range(0, NumberOfBaris):
        b = Bari()
        for _ in range(0, NumberOfHHPerBari):
            hh = Household()
            [hh.add(Individual()) for i in range(0, NumberOfIndividualsPerHH)]
            b.add(hh)

        baris.append(b)


    for b in baris:
        print("--- Bari: ", b.id, "---")
        for hh in b.households:
            household = Household.all_households[hh]
            print("--- Household: ", household.id, "---")
            for ind in household.individuals:
                print(Individual.all_individuals[ind].id, ": ", Individual.all_individuals[ind].age)

    pycuda.driver.init()

    dt = 1/365

    start_time = timeit.default_timer()
    #for _ in range(duration):
    Individual.update_all_individuals(dt)
    Household.update_all_households()
    Bari.update_all_baris()
    print(timeit.default_timer() - start_time)

    for b in baris:
        print("--- Bari: ", b.id, "---")
        for hh in b.households:
            household = Household.all_households[hh]
            print("--- Household: ", household.id, "---")
            for ind in household.individuals:
                print(Individual.all_individuals[ind].id,": ", Individual.all_individuals[ind].age)


