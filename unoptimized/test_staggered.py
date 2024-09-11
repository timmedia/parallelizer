from mpi4py import MPI
import numpy as np
from time import sleep

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

global_array = np.arange(size * 5 + 1)
local_array = global_array[rank*5:(rank+1)*5+1]
if rank == 0:
    print(global_array)
sleep(0.2)
print(f"{rank=} {local_array=}")
