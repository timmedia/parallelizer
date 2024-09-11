from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Dimensions of the local array for each process
rows, cols = 2, 3  # Adjust these as needed for your case

# Create a local 2D array for each process
local_array = np.full((rows, cols), rank, dtype=int)

print(f"{rank=} {local_array=}")

# Root process determines the total number of rows and creates the buffer to gather into
if rank == 0:
    # Total rows will be rows from each process
    total_rows = rows * size
    gather_array = np.empty((total_rows, cols), dtype=int)
else:
    gather_array = None

# Gather all the local arrays to the root process
comm.Gather(local_array, gather_array, root=0)

# Print the result on the root process
if rank == 0:
    print("Gathered array on root process:")
    print(gather_array)
