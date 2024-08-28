# from nmwc_model.solver import main

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# rank_size = comm.Get_size()

if __name__ == '__main__':
    print(f"Starting {rank=}...")
    # main()
