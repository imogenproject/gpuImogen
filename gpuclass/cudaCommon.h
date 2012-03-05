typedef struct {
        double *fluidIn[5];
        double *fluidOut[5];

        double *B[3];

        double *Ptotal;
        double *cFreeze;
        } fluidVarPtrs;


typedef struct {
    int ndims;
    int dim[3];
    int numel;
    } ArrayMetadata;

double **getGPUSourcePointers(const mxArray *prhs[], ArrayMetadata *metaReturn, int fromarg, int toarg);
double **makeGPUDestinationArrays(int64_t *reference, mxArray *retArray[], int howmany);
void cudaLaunchError(cudaError_t E, dim3 blockdim, dim3 griddim, ArrayMetadata *a, int i, char *srcname);

#define FINITEDIFFX_PREAMBLE \
/* Our assumption implicitly is that differencing occurs in the X direction in the local tile */\
int addrX = (threadIdx.x - DIFFEDGE) + blockIdx.x * (TILEDIM_X - 2*DIFFEDGE);\
int addrY = threadIdx.y + blockIdx.y * TILEDIM_Y;\
\
addrX += (addrX < 0)*FD_DIMENSION;\
\
/* Nuke the threads hanging out past the end of the X extent of the array */\
/* addrX is zero indexed, mind */\
if(addrX >= FD_DIMENSION - 1 + DIFFEDGE) return;\
if(addrY >= OTHER_DIMENSION) return;\
\
/* Mask out threads who are near the edges to prevent seg violation upon differencing */\
bool ITakeDerivative = (threadIdx.x >= DIFFEDGE) && (threadIdx.x < (TILEDIM_X - DIFFEDGE)) && (addrX < FD_DIMENSION);\
\
addrX %= FD_DIMENSION; /* Wraparound (circular boundary conditions) */\
\
/* NOTE: This chooses which direction we "actually" take derivatives in\
 *          along with the conditional add a few lines up */\
int globAddr = FD_MEMSTEP * addrX + OTHER_MEMSTEP * addrY;\
\
int tileAddr = threadIdx.x + TILEDIM_X * threadIdx.y + 1;\

#define FINITEDIFFY_PREAMBLE \
/* Our assumption implicitly is that differencing occurs in the X direction in the local tile */\
int addrX = threadIdx.x + blockIdx.x * TILEDIM_X;\
int addrY = (threadIdx.y - DIFFEDGE) + blockIdx.y * (TILEDIM_Y - 2*DIFFEDGE);\
\
addrY += (addrY < 0)*FD_DIMENSION;\
\
/* Nuke the threads hanging out past the end of the X extent of the array */\
/* addrX is zero indexed, mind */\
if(addrY >= FD_DIMENSION - 1 + DIFFEDGE) return;\
if(addrX >= OTHER_DIMENSION) return;\
\
/* Mask out threads who are near the edges to prevent seg violation upon differencing */\
bool ITakeDerivative = (threadIdx.y >= DIFFEDGE) && (threadIdx.y < (TILEDIM_Y - DIFFEDGE)) && (addrY < FD_DIMENSION);\
\
addrY %= FD_DIMENSION; /* Wraparound (circular boundary conditions) */\
\
/* NOTE: This chooses which direction we "actually" take derivatives in\
 *          along with the conditional add a few lines up */\
int globAddr = FD_MEMSTEP * addrY + OTHER_MEMSTEP * addrX;\
\
int tileAddr = threadIdx.y + TILEDIM_Y * threadIdx.x + 1;\


