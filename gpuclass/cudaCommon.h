#define ALFVEN_FACTOR 1

// These define Imogen's equation of state, giving total pressure and square of soundspeed as functions of the other variables

#define PRESS_HD(E, T, gm1fact) ( (gm1fact)*((E)-(T)) )
#define PRESS_MHD(E, T, bsq, gm1fact, alfact) (  (gm1fact)*((E)-(T)) + (alfact)*(bsq)  )
#define CSQ_HD(E, T, rho, gg1fact) ( (gg1fact)*((E) - (T)) / (rho) )
#define CSQ_MHD(E, T, bsq, rhogg1fact, alfact) (  ( (gg1fact)*((E)-(T)) + (alfact)*(bsq) )/(rho)  )


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

void arrayMetadataToTag(ArrayMetadata *meta, int64_t *tag); 
void getTagFromGPUType(const mxArray *gputype, int64_t *tag);

double **getGPUSourcePointers(const mxArray *prhs[], ArrayMetadata *metaReturn, int fromarg, int toarg);
double **makeGPUDestinationArrays(ArrayMetadata *amdRef, mxArray *retArray[], int howmany);
double *replaceGPUArray(const mxArray *prhs[], int target, int *newdims);

void getLaunchForXYCoverage(int *dims, int blkX, int blkY, int nhalo, dim3 *blockdim, dim3 *griddim);

void cudaCheckError(char *where);
void cudaLaunchError(cudaError_t E, dim3 blockdim, dim3 griddim, ArrayMetadata *a, int i, char *srcname);
const char *errorName(cudaError_t E);

void printdim3(char *name, dim3 dim);
void printgputag(char *name, int64_t *tag);

__device__ __inline__ double fluxLimiter_VanLeer(double derivL, double derivR)
{
double r;

r = 2.0 * derivL * derivR;
if(r > 0.0) { return r /(derivL+derivR); }

return 0;
}

__device__ __inline__ double fluxLimiter_Osher(double A, double B)
{
double r = A*B;
if(r <= 0.0) return 0.0;

return r*(A+B)/(A*A+r+B*B);

}

__device__ __inline__ double fluxLimiter_minmod(double derivL, double derivR)
{
if(derivL * derivR < 0) return 0.0;

if(fabs(derivL) > fabs(derivR)) { return derivR; } else { return derivL; }
}

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


