#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"

#include "mpi.h"
#include "cuda.h"
#include "cuda_runtime.h"

int main(int argc, char **argv)
{

int spin = 0;
int lol = 0;
while(spin) {
    lol = 1;
}



int x = MPI_Init(&argc, &argv);

int size, rank;

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

int i_am_groot = (rank == 0);

char thehost[128];

gethostname(&thehost[0], 127);
thehost[127] = 0x00;

printf("I am rank %i/%i running on host %s. Hello, world!\n", rank, size, thehost);

MPI_Barrier(MPI_COMM_WORLD);

if(i_am_groot) {
	printf("Got to enumerating ranks without dumping, horray!\n");
	printf("Attempting to start CUDA...\n");
}

int ndev;
cudaError_t cuprob = cudaGetDeviceCount(&ndev);

printf("Rank %i/%i has %i devices.\n", rank, size, ndev);

MPI_Finalize();

return 0;
}
