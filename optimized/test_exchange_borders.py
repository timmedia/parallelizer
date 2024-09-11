import numpy as np
from mpi4py import MPI
from time import sleep

from nmwc_model.parallel import exchange_borders_2d

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
rank_size = comm.Get_size()

gathered_data = np.arange(128).reshape((16, -1))
nb = 2
nx = 3
if rank == 0:
    print(f"data:\n{gathered_data}\n")

sleep(0.1)

rank_data = gathered_data[rank*(nx):(rank+1)*nx + 2 * nb, :]
rank_data[:nb, :] = 0
rank_data[-nb:, :] = 0
print(f"{rank=}\n{rank_data}\n")

exchange_borders_2d(rank_data, 12)
print(f"{rank=}\n{rank_data}\n")
